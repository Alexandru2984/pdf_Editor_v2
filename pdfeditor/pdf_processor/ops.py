"""Split, merge, compress, watermark, rotate, page-numbers operations."""

from __future__ import annotations

import difflib
import io
import logging
import os
import shutil
import subprocess
import zipfile
from collections.abc import Callable

import fitz
from PIL import Image as PILImage

from ._common import parse_page_range, processed_dir, safe_basename, timestamp

# Resolve ghostscript binary up-front so PDF/A conversion fails fast with a
# clear message in environments where it isn't installed.
GHOSTSCRIPT_CMD = os.environ.get("GHOSTSCRIPT_CMD") or shutil.which("gs") or "/usr/bin/gs"

# Hard ceiling on the raster a single page may produce. The page-count limit
# (500) bounds how many pages we render, but a page's *physical* size is an
# independent factor: a 10000x10000-inch page at 600 DPI rasterises to a
# multi-GB bitmap that OOM-kills the worker before the file ever hits disk.
# We estimate the uncompressed footprint (w_px * h_px * 4 bytes RGBA) before
# calling get_pixmap and refuse pages that blow past this. Override via env
# for hosts with more headroom.
MAX_PIXMAP_BYTES = int(os.environ.get("PDF_MAX_PIXMAP_BYTES", str(100 * 1024 * 1024)))


def _guard_pixmap_memory(page: fitz.Page, zoom: float, page_index: int) -> None:
    """Reject a page whose rasterised footprint at ``zoom`` would exceed
    ``MAX_PIXMAP_BYTES``, before any large allocation happens.

    ``page_index`` is 1-indexed and only used for the error message.
    """
    rect = page.rect
    width_px = rect.width * zoom
    height_px = rect.height * zoom
    # 4 bytes/pixel is the worst case (RGBA). Using it for RGB too keeps the
    # guard conservative — better to reject slightly early than OOM.
    estimated = width_px * height_px * 4
    if estimated > MAX_PIXMAP_BYTES:
        raise ValueError(
            f"Page {page_index} is too large to render at this resolution "
            f"(~{estimated / (1024 * 1024):.0f} MB, limit "
            f"{MAX_PIXMAP_BYTES // (1024 * 1024)} MB). Lower the DPI or crop the page."
        )


def split_pdf(pdf_path: str, ranges: list[tuple[int, int]]) -> list[str]:
    """ranges: list of (start,end) 1-indexed inclusive."""
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    out_dir = processed_dir()
    base = safe_basename(pdf_path)

    with fitz.open(pdf_path) as doc:
        total = len(doc)
        out_files: list[str] = []

        for idx, (start, end) in enumerate(ranges, 1):
            if start < 1 or end > total or start > end:
                raise ValueError(f"Invalid range: {start}-{end} (PDF has {total} pages)")

            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=start - 1, to_page=end - 1)

            suffix = f"pages_{start}-{end}" if len(ranges) == 1 else f"part{idx}_pages_{start}-{end}"
            out_path = os.path.join(out_dir, f"{base}_{suffix}_{timestamp()}.pdf")

            new_doc.save(out_path, garbage=4, deflate=True, clean=True)
            new_doc.close()
            out_files.append(out_path)

        return out_files


def merge_pdfs(pdf_paths: list[str], output_name: str | None = None) -> str:
    if len(pdf_paths) < 2:
        raise ValueError("At least 2 PDF files are required for merging")
    for p in pdf_paths:
        if not os.path.exists(p):
            raise ValueError(f"PDF file not found: {p}")

    out_dir = processed_dir()
    name = output_name or f"merged_{timestamp()}"
    out_path = os.path.join(out_dir, f"{name}.pdf")

    out = fitz.open()
    try:
        for p in pdf_paths:
            with fitz.open(p) as d:
                out.insert_pdf(d)
        out.save(out_path, garbage=4, deflate=True, clean=True)
        return out_path
    finally:
        out.close()


def compress_pdf(
    pdf_path: str,
    quality: str = "medium",
    output_name: str | None = None,
) -> tuple[str, int, int, float]:
    """Best-effort compression: deflate streams + re-encode images as JPEG."""
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    qmap = {"low": 50, "medium": 75, "high": 90}
    jpg_quality = int(qmap.get(quality, 75))

    out_dir = processed_dir()
    name = output_name or f"compressed_{safe_basename(pdf_path)}_{timestamp()}"
    out_path = os.path.join(out_dir, f"{name}.pdf")

    original_size = os.path.getsize(pdf_path)

    with fitz.open(pdf_path) as doc:
        for page in doc:
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    info = doc.extract_image(xref)
                    img_bytes = info.get("image")
                    if not img_bytes:
                        continue
                    im = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=jpg_quality, optimize=True)
                    doc.update_stream(xref, buf.getvalue())
                except Exception:
                    continue

        doc.save(out_path, garbage=4, deflate=True, clean=True)

    compressed_size = os.path.getsize(out_path)
    ratio = ((original_size - compressed_size) / original_size) * 100 if original_size else 0.0
    return out_path, original_size, compressed_size, ratio


def sign_pdf(
    pdf_path: str,
    p12_bytes: bytes,
    p12_password: str,
    page: int = 1,
    position: str = "bottom-right",
    reason: str = "",
    location: str = "",
    field_name: str = "Signature1",
    tsa_url: str | None = None,
    embed_validation_info: bool = False,
    add_doc_timestamp: bool = False,
) -> str:
    """Apply a cryptographic PKCS#7 signature to a PDF using a user-supplied .p12.

    A visible signature widget is added to ``page`` (1-indexed) at ``position``.
    Returns the path to the signed PDF. ``p12_bytes`` is the raw PKCS#12 archive,
    ``p12_password`` decrypts the private key inside it. When ``tsa_url`` is set,
    pyHanko fetches an RFC 3161 timestamp from that authority and embeds it in the
    signature (``signature-time-stamp`` attribute).

    When ``embed_validation_info`` is True, fetches OCSP/CRL responses for the
    signing chain and embeds them in a Document Security Store (PAdES B-LT).
    Self-signed or otherwise unverifiable certificates will raise ``ValueError``.

    When ``add_doc_timestamp`` is True (and ``tsa_url`` is set), appends a
    document-level RFC 3161 timestamp after the main signature (PAdES B-LTA when
    combined with LTV). Requires ``tsa_url`` — raises ``ValueError`` otherwise.
    """
    from pyhanko.pdf_utils.incremental_writer import IncrementalPdfFileWriter
    from pyhanko.sign import fields, signers
    from pyhanko.sign.fields import SigSeedSubFilter
    from pyhanko.sign.timestamps import HTTPTimeStamper
    from pyhanko_certvalidator.context import ValidationContext

    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")
    if not p12_bytes:
        raise ValueError("p12 archive is empty")

    try:
        signer = signers.SimpleSigner.load_pkcs12_data(
            pkcs12_bytes=p12_bytes,
            other_certs=[],
            passphrase=(p12_password or "").encode("utf-8"),
        )
    except Exception as exc:
        raise ValueError(f"Could not load certificate: {exc}") from exc

    # Compute widget box in PDF coords for the requested page.
    with fitz.open(pdf_path) as doc:
        if doc.is_encrypted:
            raise ValueError("Cannot sign an encrypted PDF — remove the password first")
        if page < 1 or page > len(doc):
            raise ValueError(f"Page {page} out of range (1-{len(doc)})")
        rect = doc[page - 1].rect
        page_w, page_h = float(rect.width), float(rect.height)

    widget_w, widget_h = 200, 60
    margin = 24
    positions = {
        "top-left": (margin, page_h - margin - widget_h, margin + widget_w, page_h - margin),
        "top-right": (
            page_w - margin - widget_w,
            page_h - margin - widget_h,
            page_w - margin,
            page_h - margin,
        ),
        "bottom-left": (margin, margin, margin + widget_w, margin + widget_h),
        "bottom-right": (
            page_w - margin - widget_w,
            margin,
            page_w - margin,
            margin + widget_h,
        ),
        "center": (
            (page_w - widget_w) / 2,
            (page_h - widget_h) / 2,
            (page_w + widget_w) / 2,
            (page_h + widget_h) / 2,
        ),
    }
    box = positions.get(position, positions["bottom-right"])
    box_int: tuple[int, int, int, int] = (
        int(round(box[0])),
        int(round(box[1])),
        int(round(box[2])),
        int(round(box[3])),
    )

    out_dir = processed_dir()
    out_path = os.path.join(out_dir, f"signed_{safe_basename(pdf_path)}_{timestamp()}.pdf")

    with open(pdf_path, "rb") as inf:
        writer = IncrementalPdfFileWriter(inf)
        fields.append_signature_field(
            writer,
            sig_field_spec=fields.SigFieldSpec(
                sig_field_name=field_name,
                on_page=page - 1,
                box=box_int,
            ),
        )
        timestamper = HTTPTimeStamper(tsa_url, timeout=10) if tsa_url else None

        validation_context = None
        meta_kwargs: dict = {
            "field_name": field_name,
            "reason": reason or None,
            "location": location or None,
        }
        if embed_validation_info:
            # PAdES B-LT requires the PAdES subfilter; ValidationContext fetches
            # OCSP/CRL for the signer chain at signing time and we embed it.
            # The chain comes from the user's uploaded .p12, so route every
            # fetch through an SSRF-guarded backend that refuses URLs resolving
            # to internal/loopback/link-local addresses.
            from .ssrf_guard import guarded_fetcher_backend

            validation_context = ValidationContext(
                allow_fetching=True,
                fetcher_backend=guarded_fetcher_backend(),
            )
            meta_kwargs["subfilter"] = SigSeedSubFilter.PADES
            meta_kwargs["embed_validation_info"] = True
            meta_kwargs["validation_context"] = validation_context

        pdf_signer = signers.PdfSigner(
            signers.PdfSignatureMetadata(**meta_kwargs),
            signer=signer,
            timestamper=timestamper,
        )
        from pyhanko_certvalidator.errors import (
            InvalidCertificateError,
            PathBuildingError,
            PathValidationError,
        )

        try:
            with open(out_path, "wb") as outf:
                pdf_signer.sign_pdf(writer, output=outf)
        except (InvalidCertificateError, PathBuildingError, PathValidationError) as exc:
            raise ValueError(
                f"Cannot embed validation info — certificate chain could not be "
                f"verified (likely self-signed or unknown CA): {exc}"
            ) from exc
        except Exception as exc:
            err_msg = str(exc).lower()
            if tsa_url and ("timestamp" in err_msg or "tsa" in err_msg):
                raise ValueError(f"Timestamp authority error: {exc}") from exc
            raise

    if add_doc_timestamp:
        if not tsa_url:
            raise ValueError("Document timestamp requires a TSA URL")
        # Re-open the just-signed PDF for an incremental update that appends
        # a DocTimeStamp signature (PAdES B-LTA).
        ts_out_path = os.path.join(out_dir, f"signed_lta_{safe_basename(pdf_path)}_{timestamp()}.pdf")
        try:
            with open(out_path, "rb") as inf:
                ts_writer = IncrementalPdfFileWriter(inf)
                ts_stamper = signers.PdfTimeStamper(timestamper=HTTPTimeStamper(tsa_url, timeout=10))
                with open(ts_out_path, "wb") as outf:
                    ts_stamper.timestamp_pdf(
                        ts_writer,
                        md_algorithm="sha256",
                        output=outf,
                    )
        except Exception as exc:
            err_msg = str(exc).lower()
            if "timestamp" in err_msg or "tsa" in err_msg:
                raise ValueError(f"Document timestamp authority error: {exc}") from exc
            raise
        # Replace the B-LT output with the B-LTA one.
        import contextlib

        with contextlib.suppress(OSError):
            os.remove(out_path)
        out_path = ts_out_path

    return out_path


def verify_pdf_signatures(pdf_path: str, extra_trust_certs: list[bytes] | None = None) -> list[dict]:
    """Validate every embedded signature in a PDF and return a per-signature report.

    ``extra_trust_certs`` is a list of additional PEM/DER certificate blobs to
    treat as trust anchors — useful for self-signed or test certificates that
    would otherwise be rejected.

    Each report dict contains: ``field_name``, ``signer_name``, ``signing_time``
    (ISO8601 or None), ``has_timestamp``, ``timestamp_valid``, ``trusted``,
    ``intact`` (signature integrity), ``revoked``, ``summary`` (human-readable
    one-liner), and ``errors`` (list of error strings).
    """
    import asyncio

    from asn1crypto import pem, x509
    from pyhanko.pdf_utils.reader import PdfFileReader
    from pyhanko.sign.validation import async_validate_pdf_signature
    from pyhanko_certvalidator.context import ValidationContext

    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    extra_anchors: list[x509.Certificate] = []
    for blob in extra_trust_certs or []:
        try:
            if pem.detect(blob):
                _, _, der = pem.unarmor(blob)
            else:
                der = blob
            extra_anchors.append(x509.Certificate.load(der))
        except Exception:
            # Ignore malformed extras; the report will reflect failures.
            continue

    async def _do() -> list[dict]:
        out: list[dict] = []
        with open(pdf_path, "rb") as f:
            try:
                reader = PdfFileReader(f)
                signatures = list(reader.embedded_signatures)
            except Exception as exc:
                # Corrupted PDF (mid-file mutation can break xref/trailer parsing
                # before pyHanko reaches the signatures). Report a single
                # synthetic failure entry rather than propagating the parse error.
                return [
                    {
                        "field_name": "",
                        "signer_name": "",
                        "signing_time": None,
                        "has_timestamp": False,
                        "timestamp_valid": False,
                        "trusted": False,
                        "intact": False,
                        "revoked": False,
                        "summary": "Could not parse PDF",
                        "errors": [str(exc)],
                    }
                ]
            for sig in signatures:
                report = {
                    "field_name": sig.field_name,
                    "signer_name": "",
                    "signing_time": None,
                    "has_timestamp": False,
                    "timestamp_valid": False,
                    "trusted": False,
                    "intact": False,
                    "revoked": False,
                    "summary": "",
                    "errors": [],
                }
                try:
                    import contextlib

                    cn = ""
                    with contextlib.suppress(Exception):
                        cn = sig.signer_cert.subject.human_friendly
                    report["signer_name"] = cn

                    # allow_fetching=False: this endpoint is unauthenticated and
                    # the signer cert (with its AIA/CRL/OCSP URLs) is fully
                    # attacker-controlled, so live fetching would be an SSRF
                    # vector. Trust + revocation are still checked against the
                    # provided anchors and any revocation info embedded in the
                    # PDF's DSS — we just never reach out to URLs from the cert.
                    vc = ValidationContext(
                        allow_fetching=False,
                        extra_trust_roots=extra_anchors or None,
                        revocation_mode="soft-fail",
                    )
                    status = await async_validate_pdf_signature(sig, signer_validation_context=vc)
                    report["intact"] = bool(status.intact)
                    report["trusted"] = bool(status.trusted)
                    report["revoked"] = bool(getattr(status, "revoked", False))
                    if status.signer_reported_dt:
                        report["signing_time"] = status.signer_reported_dt.isoformat()
                    if status.timestamp_validity is not None:
                        report["has_timestamp"] = True
                        report["timestamp_valid"] = bool(status.timestamp_validity.intact)
                    report["summary"] = (
                        str(status.bottom_line)
                        if status.bottom_line
                        else ("Valid" if (status.intact and status.trusted) else "Invalid")
                    )
                except Exception as exc:
                    report["errors"].append(str(exc))
                    report["summary"] = "Could not validate"
                out.append(report)
        return out

    return asyncio.run(_do())


def protect_pdf(
    pdf_path: str,
    user_password: str,
    owner_password: str | None = None,
) -> str:
    """Encrypt a PDF with AES-256. Without owner_password, owner == user."""
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")
    if not user_password:
        raise ValueError("user_password is required")

    out_dir = processed_dir()
    base = safe_basename(pdf_path)
    out_path = os.path.join(out_dir, f"{base}_protected_{timestamp()}.pdf")

    perm = fitz.PDF_PERM_ACCESSIBILITY | fitz.PDF_PERM_PRINT | fitz.PDF_PERM_COPY | fitz.PDF_PERM_ANNOTATE

    with fitz.open(pdf_path) as doc:
        if doc.is_encrypted:
            raise ValueError("PDF is already encrypted")
        doc.save(
            out_path,
            encryption=fitz.PDF_ENCRYPT_AES_256,
            user_pw=user_password,
            owner_pw=owner_password or user_password,
            permissions=perm,
            garbage=4,
            deflate=True,
            clean=True,
        )

    return out_path


def remove_pdf_password(pdf_path: str, password: str) -> str:
    """Decrypt a PDF and return a copy without password protection.

    ``password`` may be the user or owner password. Raises ``ValueError`` for
    missing files, missing password input, an unencrypted source, or an
    incorrect password.
    """
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")
    if password is None:
        raise ValueError("Password is required")

    out_dir = processed_dir()
    base = safe_basename(pdf_path)
    out_path = os.path.join(out_dir, f"{base}_unprotected_{timestamp()}.pdf")

    with fitz.open(pdf_path) as doc:
        if not doc.is_encrypted:
            raise ValueError("PDF is not password-protected")
        if not doc.authenticate(password or ""):
            raise ValueError("Incorrect password")
        doc.save(
            out_path,
            encryption=fitz.PDF_ENCRYPT_NONE,
            garbage=4,
            deflate=True,
            clean=True,
        )

    return out_path


def convert_pdf_to_docx(pdf_path: str) -> tuple[str, bool]:
    """Convert a PDF to a .docx file using pdf2docx.

    Returns ``(output_path, has_extractable_text)``. When ``has_extractable_text``
    is False, the source PDF is image-only (likely a scan) and the resulting
    .docx will be near-empty — the caller should warn the user to OCR first.
    """
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    has_text = False
    with fitz.open(pdf_path) as doc:
        if doc.is_encrypted:
            raise ValueError("Cannot convert an encrypted PDF — remove the password first")
        for page in doc:
            if page.get_text("text").strip():
                has_text = True
                break

    out_dir = processed_dir()
    base = safe_basename(pdf_path)
    out_path = os.path.join(out_dir, f"{base}_{timestamp()}.docx")

    # pdf2docx writes progress lines via `logging.info(...)` on the root
    # logger, which floods callers' logs. Bump the root threshold while
    # converting and restore it afterwards.
    from pdf2docx import Converter

    root_logger = logging.getLogger()
    prev_level = root_logger.level
    if prev_level < logging.WARNING:
        root_logger.setLevel(logging.WARNING)
    try:
        cv = Converter(pdf_path)
        try:
            cv.convert(out_path)
        finally:
            cv.close()
    finally:
        root_logger.setLevel(prev_level)

    return out_path, has_text


def reorder_pages(pdf_path: str, page_order: list[int]) -> str:
    """Produce a new PDF containing only the pages in ``page_order``.

    ``page_order`` is a list of 1-indexed page numbers in the desired final
    order. Pages omitted from the list are deleted; pages may not be
    duplicated. The list must be non-empty.
    """
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")
    if not page_order:
        raise ValueError("Page order must contain at least one page")
    if len(set(page_order)) != len(page_order):
        raise ValueError("Page order cannot contain duplicates")

    out_dir = processed_dir()
    base = safe_basename(pdf_path)
    out_path = os.path.join(out_dir, f"{base}_reordered_{timestamp()}.pdf")

    with fitz.open(pdf_path) as doc:
        total = len(doc)
        for n in page_order:
            if n < 1 or n > total:
                raise ValueError(f"Invalid page number: {n} (PDF has {total} pages)")

        new_doc = fitz.open()
        try:
            for n in page_order:
                new_doc.insert_pdf(doc, from_page=n - 1, to_page=n - 1)
            new_doc.save(out_path, garbage=4, deflate=True, clean=True)
        finally:
            new_doc.close()

    return out_path


def crop_pages(
    pdf_path: str,
    top: float = 0.0,
    right: float = 0.0,
    bottom: float = 0.0,
    left: float = 0.0,
    page_range: str | None = None,
) -> str:
    """Shrink the visible area of pages by removing percentage margins.

    Each margin is expressed in percent of the page width (left/right) or
    height (top/bottom). Values must be in ``[0, 49]`` and opposite margins
    must sum to less than 100. Cropping is implemented via PyMuPDF's
    ``set_cropbox`` — viewers respect the smaller box but the underlying
    page content is preserved.

    ``page_range`` is the standard ``"1-3,5,7-9"`` syntax; ``None`` or an
    empty string means every page.
    """
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")
    margins = (top, right, bottom, left)
    for m in margins:
        if m is None or m < 0 or m >= 50:
            raise ValueError("Each margin must be in [0, 49] percent")
    if all(m == 0 for m in margins):
        raise ValueError("At least one margin must be greater than zero")

    out_dir = processed_dir()
    base = safe_basename(pdf_path)
    out_path = os.path.join(out_dir, f"{base}_cropped_{timestamp()}.pdf")

    with fitz.open(pdf_path) as doc:
        if doc.is_encrypted:
            raise ValueError("Cannot crop an encrypted PDF — remove the password first")
        total = len(doc)
        if total == 0:
            raise ValueError("PDF has no pages")
        indices = parse_page_range(page_range or "", total)

        for idx in indices:
            page = doc[idx]
            r = page.rect
            new_box = fitz.Rect(
                r.x0 + r.width * left / 100.0,
                r.y0 + r.height * top / 100.0,
                r.x1 - r.width * right / 100.0,
                r.y1 - r.height * bottom / 100.0,
            )
            page.set_cropbox(new_box)

        doc.save(out_path, garbage=4, deflate=True, clean=True)

    return out_path


def flatten_pdf(
    pdf_path: str,
    flatten_annotations: bool = True,
    flatten_forms: bool = True,
) -> str:
    """Bake annotations and/or form widgets into permanent page content.

    After flattening the page looks identical, but interactive form fields
    and annotations are converted to vector text/graphics — they can no
    longer be edited or toggled. Text remains selectable.
    """
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")
    if not flatten_annotations and not flatten_forms:
        raise ValueError("Select at least one of annotations or form fields to flatten")

    out_dir = processed_dir()
    base = safe_basename(pdf_path)
    out_path = os.path.join(out_dir, f"{base}_flattened_{timestamp()}.pdf")

    with fitz.open(pdf_path) as doc:
        if doc.is_encrypted:
            raise ValueError("Cannot flatten an encrypted PDF — remove the password first")
        doc.bake(annots=flatten_annotations, widgets=flatten_forms)
        doc.save(out_path, garbage=4, deflate=True, clean=True)

    return out_path


def redact_text(
    pdf_path: str,
    search_terms: list[str],
    page_range: str | None = None,
) -> tuple[str, int]:
    """Permanently remove every occurrence of each ``search_term``.

    Each match becomes a black rectangle and the underlying text/images are
    stripped from the page content stream — not merely covered. Search is
    case-insensitive (PyMuPDF's default). Returns ``(output_path, total_matches)``.
    """
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")
    cleaned_terms = [t.strip() for t in (search_terms or []) if t and t.strip()]
    if not cleaned_terms:
        raise ValueError("Provide at least one search term")

    out_dir = processed_dir()
    base = safe_basename(pdf_path)
    out_path = os.path.join(out_dir, f"{base}_redacted_{timestamp()}.pdf")

    total_matches = 0
    with fitz.open(pdf_path) as doc:
        if doc.is_encrypted:
            raise ValueError("Cannot redact an encrypted PDF — remove the password first")
        total = len(doc)
        if total == 0:
            raise ValueError("PDF has no pages")
        indices = parse_page_range(page_range or "", total)

        for idx in indices:
            page = doc[idx]
            page_had_match = False
            for term in cleaned_terms:
                rects = page.search_for(term) or []
                for rect in rects:
                    page.add_redact_annot(rect, fill=(0, 0, 0))
                    total_matches += 1
                    page_had_match = True
            if page_had_match:
                page.apply_redactions()

        doc.save(out_path, garbage=4, deflate=True, clean=True)

    return out_path, total_matches


_PDFA_VERSION_TO_GS_LEVEL = {"1b": 1, "2b": 2}


def convert_to_pdfa(pdf_path: str, version: str = "2b") -> tuple[str, str]:
    """Convert a PDF to PDF/A-1b or PDF/A-2b for long-term archival.

    Uses ghostscript with ``-dPDFA=N -dPDFACompatibilityPolicy=1``, which
    fails the conversion if the source contains anything that can't be
    represented in PDF/A (e.g. transparency on 1b) rather than silently
    dropping it. Color is normalised through device-independent space so
    embedded profiles aren't required. Returns ``(output_path, version)``.
    """
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")
    if version not in _PDFA_VERSION_TO_GS_LEVEL:
        raise ValueError("Unsupported PDF/A version. Choose '1b' or '2b'.")

    with fitz.open(pdf_path) as doc:
        if doc.is_encrypted:
            raise ValueError("Cannot convert an encrypted PDF — remove the password first")
        if len(doc) == 0:
            raise ValueError("PDF has no pages")

    if not (GHOSTSCRIPT_CMD and os.path.exists(GHOSTSCRIPT_CMD)):
        raise ValueError("ghostscript binary not found — install ghostscript on the host")

    level = _PDFA_VERSION_TO_GS_LEVEL[version]
    out_dir = processed_dir()
    base = safe_basename(pdf_path)
    out_path = os.path.join(out_dir, f"{base}_pdfa{version}_{timestamp()}.pdf")

    cmd = [
        GHOSTSCRIPT_CMD,
        # -dSAFER sandboxes the interpreter: it blocks the PostScript file
        # operators that a malicious PDF could use to read host files (e.g.
        # .env) or run commands — Ghostscript's classic RCE/file-disclosure
        # vector. GS >=9.50 enables it by default, but the host fallback gs
        # at GHOSTSCRIPT_CMD may be older, so we never rely on that default.
        "-dSAFER",
        "-dPDFA=" + str(level),
        "-dBATCH",
        "-dNOPAUSE",
        "-dQUIET",
        "-dPDFACompatibilityPolicy=1",
        "-sColorConversionStrategy=UseDeviceIndependentColor",
        "-sDEVICE=pdfwrite",
        "-sOutputFile=" + out_path,
        pdf_path,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("PDF/A conversion timed out") from exc
    except FileNotFoundError as exc:
        raise ValueError("ghostscript binary not found") from exc

    if proc.returncode != 0 or not os.path.exists(out_path):
        stderr = proc.stderr.decode("utf-8", errors="replace")[:500] if proc.stderr else ""
        raise ValueError(f"PDF/A conversion failed: {stderr.strip() or 'ghostscript error'}")

    return out_path, version


def read_pdf_outline(pdf_path: str) -> list[dict[str, int | str]]:
    """Return the existing bookmarks/outline as a flat list of
    ``{"level": int, "title": str, "page": int}`` dicts (page is 1-indexed).
    Returns an empty list if the PDF has no TOC."""
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")
    with fitz.open(pdf_path) as doc:
        if doc.is_encrypted:
            raise ValueError("Cannot read outline of an encrypted PDF — remove the password first")
        toc = doc.get_toc(simple=True) or []
    return [{"level": int(lvl), "title": str(title), "page": int(page)} for lvl, title, page in toc]


def set_pdf_outline(pdf_path: str, entries: list[dict[str, int | str]]) -> str:
    """Replace the PDF outline with ``entries`` and save a new copy.

    Each entry must have integer ``level`` (>=1), non-empty ``title``, and
    integer ``page`` within the document's page range. Levels must form a
    valid tree — the first entry must be level 1, and levels cannot jump
    (e.g. cannot go from 1 directly to 3). An empty list clears the TOC.
    """
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    out_dir = processed_dir()
    base = safe_basename(pdf_path)
    out_path = os.path.join(out_dir, f"{base}_outline_{timestamp()}.pdf")

    with fitz.open(pdf_path) as doc:
        if doc.is_encrypted:
            raise ValueError("Cannot edit outline of an encrypted PDF — remove the password first")
        total = len(doc)
        if total == 0:
            raise ValueError("PDF has no pages")

        toc: list[list[int | str]] = []
        prev_level = 0
        for idx, entry in enumerate(entries, 1):
            try:
                level = int(entry["level"])
                title = str(entry["title"]).strip()
                page = int(entry["page"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Entry {idx} is malformed: {exc}") from exc
            if level < 1:
                raise ValueError(f"Entry {idx} has invalid level {level} (must be ≥ 1)")
            if not title:
                raise ValueError(f"Entry {idx} has an empty title")
            if page < 1 or page > total:
                raise ValueError(f"Entry {idx} page {page} is out of range (PDF has {total} pages)")
            if prev_level == 0 and level != 1:
                raise ValueError(f"Entry {idx} must start at level 1")
            if level > prev_level + 1:
                raise ValueError(
                    f"Entry {idx} jumps from level {prev_level} to {level} — levels must increase by 1"
                )
            toc.append([level, title, page])
            prev_level = level

        doc.set_toc(toc)
        doc.save(out_path, garbage=4, deflate=True, clean=True)

    return out_path


def _page_text_lines(page: fitz.Page) -> list[str]:
    """Extract page text and split into trimmed, non-empty lines."""
    raw = page.get_text() or ""
    return [ln.rstrip() for ln in raw.splitlines() if ln.strip()]


def _diff_pages(lines_a: list[str], lines_b: list[str]) -> tuple[list[tuple[str, str]], bool]:
    """Return (diff_lines, has_changes). Each diff line is (tag, text) where
    tag is one of "same", "del", "add"."""
    matcher = difflib.SequenceMatcher(a=lines_a, b=lines_b, autojunk=False)
    out: list[tuple[str, str]] = []
    has_changes = False
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for line in lines_a[i1:i2]:
                out.append(("same", line))
        elif op == "delete":
            has_changes = True
            for line in lines_a[i1:i2]:
                out.append(("del", line))
        elif op == "insert":
            has_changes = True
            for line in lines_b[j1:j2]:
                out.append(("add", line))
        elif op == "replace":
            has_changes = True
            for line in lines_a[i1:i2]:
                out.append(("del", line))
            for line in lines_b[j1:j2]:
                out.append(("add", line))
    return out, has_changes


_COMPARE_PAGE_WIDTH = 595.0
_COMPARE_PAGE_HEIGHT = 842.0
_COMPARE_MARGIN_X = 40.0
_COMPARE_MARGIN_Y = 50.0
_COMPARE_LINE_HEIGHT = 12.0
_COMPARE_TAG_COLORS = {
    "same": (0.25, 0.25, 0.25),
    "del": (0.75, 0.0, 0.0),
    "add": (0.0, 0.55, 0.1),
}
_COMPARE_TAG_PREFIX = {"same": "  ", "del": "- ", "add": "+ "}


def _wrap_text(text: str, max_chars: int = 95) -> list[str]:
    """Crude wrap: split on max_chars while preserving original whitespace.
    Good enough for monospace-style diff output rendered by PyMuPDF."""
    if not text:
        return [""]
    chunks = []
    while len(text) > max_chars:
        chunks.append(text[:max_chars])
        text = text[max_chars:]
    chunks.append(text)
    return chunks


def _emit_diff_lines(
    out_doc: fitz.Document,
    header: str,
    diff_lines: list[tuple[str, str]],
) -> None:
    """Render a header + colored diff lines, paginating as needed."""
    page = out_doc.new_page(width=_COMPARE_PAGE_WIDTH, height=_COMPARE_PAGE_HEIGHT)
    page.insert_text((_COMPARE_MARGIN_X, _COMPARE_MARGIN_Y), header, fontsize=13, color=(0, 0, 0))
    y = _COMPARE_MARGIN_Y + 24
    max_y = _COMPARE_PAGE_HEIGHT - _COMPARE_MARGIN_Y

    for tag, text in diff_lines:
        color = _COMPARE_TAG_COLORS[tag]
        prefix = _COMPARE_TAG_PREFIX[tag]
        for wrapped in _wrap_text(prefix + text):
            if y > max_y:
                page = out_doc.new_page(width=_COMPARE_PAGE_WIDTH, height=_COMPARE_PAGE_HEIGHT)
                page.insert_text(
                    (_COMPARE_MARGIN_X, _COMPARE_MARGIN_Y),
                    header + " (continued)",
                    fontsize=11,
                    color=(0.4, 0.4, 0.4),
                )
                y = _COMPARE_MARGIN_Y + 20
            page.insert_text(
                (_COMPARE_MARGIN_X, y),
                wrapped,
                fontsize=9,
                fontname="courier",
                color=color,
            )
            y += _COMPARE_LINE_HEIGHT


def compare_pdfs(pdf_a_path: str, pdf_b_path: str) -> tuple[str, dict[str, int]]:
    """Compare two PDFs page-by-page and emit a diff report PDF.

    Text from each page is extracted and run through ``difflib.SequenceMatcher``
    to identify added / removed / changed lines. The report starts with a
    summary page (page counts, pages identical / changed / added / removed)
    followed by one or more diff pages per non-identical page. Returns
    ``(out_path, stats)`` where stats contains ``pages_a``, ``pages_b``,
    ``identical``, ``changed``, ``added``, ``removed``.
    """
    for p in (pdf_a_path, pdf_b_path):
        if not os.path.exists(p):
            raise ValueError(f"PDF file not found: {p}")
    if os.path.abspath(pdf_a_path) == os.path.abspath(pdf_b_path):
        raise ValueError("Cannot compare a PDF against itself — choose two different files")

    out_dir = processed_dir()
    base_a = safe_basename(pdf_a_path)
    base_b = safe_basename(pdf_b_path)
    out_path = os.path.join(out_dir, f"{base_a}_vs_{base_b}_diff_{timestamp()}.pdf")

    with fitz.open(pdf_a_path) as doc_a, fitz.open(pdf_b_path) as doc_b:
        if doc_a.is_encrypted or doc_b.is_encrypted:
            raise ValueError("Cannot compare encrypted PDFs — remove the password first")

        pages_a = len(doc_a)
        pages_b = len(doc_b)
        if pages_a == 0 or pages_b == 0:
            raise ValueError("Both PDFs must have at least one page")

        text_a = [_page_text_lines(doc_a[i]) for i in range(pages_a)]
        text_b = [_page_text_lines(doc_b[i]) for i in range(pages_b)]

    stats = {
        "pages_a": pages_a,
        "pages_b": pages_b,
        "identical": 0,
        "changed": 0,
        "added": 0,
        "removed": 0,
    }
    per_page_reports: list[tuple[str, list[tuple[str, str]]]] = []
    common = min(pages_a, pages_b)

    for i in range(common):
        diff_lines, has_changes = _diff_pages(text_a[i], text_b[i])
        if has_changes:
            stats["changed"] += 1
            per_page_reports.append((f"Page {i + 1} — changed", diff_lines))
        else:
            stats["identical"] += 1

    for i in range(common, pages_a):
        stats["removed"] += 1
        per_page_reports.append((f"Page {i + 1} — removed in revised", [("del", line) for line in text_a[i]]))
    for i in range(common, pages_b):
        stats["added"] += 1
        per_page_reports.append((f"Page {i + 1} — added in revised", [("add", line) for line in text_b[i]]))

    out_doc = fitz.open()
    try:
        summary = out_doc.new_page(width=_COMPARE_PAGE_WIDTH, height=_COMPARE_PAGE_HEIGHT)
        y = _COMPARE_MARGIN_Y
        summary.insert_text(
            (_COMPARE_MARGIN_X, y),
            "PDF Comparison Report",
            fontsize=18,
            color=(0, 0, 0),
        )
        y += 32
        for label, value in [
            ("Original", f"{os.path.basename(pdf_a_path)} ({pages_a} pages)"),
            ("Revised", f"{os.path.basename(pdf_b_path)} ({pages_b} pages)"),
            ("Pages identical", str(stats["identical"])),
            ("Pages changed", str(stats["changed"])),
            ("Pages added in revised", str(stats["added"])),
            ("Pages removed in revised", str(stats["removed"])),
        ]:
            summary.insert_text(
                (_COMPARE_MARGIN_X, y), f"{label}: {value}", fontsize=11, color=(0.15, 0.15, 0.15)
            )
            y += 18

        for header, diff_lines in per_page_reports:
            _emit_diff_lines(out_doc, header, diff_lines)

        if not per_page_reports:
            note = out_doc.new_page(width=_COMPARE_PAGE_WIDTH, height=_COMPARE_PAGE_HEIGHT)
            note.insert_text(
                (_COMPARE_MARGIN_X, _COMPARE_MARGIN_Y),
                "No textual differences found.",
                fontsize=14,
                color=(0, 0.45, 0.1),
            )

        out_doc.save(out_path, garbage=4, deflate=True, clean=True)
    finally:
        out_doc.close()

    return out_path, stats


PAGE_SIZES_PT: dict[str, tuple[float, float]] = {
    # 1 inch = 72 points
    "a4": (595.0, 842.0),  # 210 x 297 mm
    "letter": (612.0, 792.0),  # 8.5 x 11 in
}


def convert_images_to_pdf(
    image_paths: list[str],
    page_size: str = "auto",
    fit_mode: str = "fit",
    output_name: str | None = None,
) -> tuple[str, int]:
    """Build a single PDF with one image per page.

    ``page_size`` is ``"auto"`` (page matches image dimensions), or ``"a4"`` /
    ``"letter"`` (fixed page; orientation chosen automatically per image based
    on aspect ratio). ``fit_mode`` is ``"fit"`` (preserve aspect ratio,
    letterbox on white) or ``"fill"`` (stretch to fill page, may distort).

    Returns ``(output_path, image_count)``.
    """
    if not image_paths:
        raise ValueError("At least one image is required")
    page_size = (page_size or "auto").lower()
    if page_size not in ("auto", "a4", "letter"):
        raise ValueError(f"Unsupported page size: {page_size}")
    fit_mode = (fit_mode or "fit").lower()
    if fit_mode not in ("fit", "fill"):
        raise ValueError(f"Unsupported fit mode: {fit_mode}")

    for p in image_paths:
        if not os.path.exists(p):
            raise ValueError(f"Image not found: {p}")

    out_dir = processed_dir()
    name = output_name or f"images_to_pdf_{timestamp()}"
    out_path = os.path.join(out_dir, f"{name}.pdf")

    out = fitz.open()
    try:
        for img_path in image_paths:
            try:
                im: PILImage.Image = PILImage.open(img_path)
            except Exception as exc:
                raise ValueError(f"Could not open image: {os.path.basename(img_path)} ({exc})") from exc

            # Flatten alpha onto a white background and normalise to RGB so
            # PyMuPDF doesn't refuse exotic modes (P, RGBA, LA, ...).
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                bg = PILImage.new("RGB", im.size, (255, 255, 255))
                rgba = im.convert("RGBA")
                bg.paste(rgba, mask=rgba.split()[3])
                im = bg
            elif im.mode != "RGB":
                im = im.convert("RGB")

            img_w_px, img_h_px = im.size

            if page_size == "auto":
                page_w, page_h = float(img_w_px), float(img_h_px)
            else:
                base_w, base_h = PAGE_SIZES_PT[page_size]
                # Auto-orient page: landscape image -> landscape page.
                if img_w_px > img_h_px:
                    page_w, page_h = max(base_w, base_h), min(base_w, base_h)
                else:
                    page_w, page_h = min(base_w, base_h), max(base_w, base_h)

            page = out.new_page(width=page_w, height=page_h)

            if fit_mode == "fill" or page_size == "auto":
                rect = fitz.Rect(0, 0, page_w, page_h)
            else:
                # Preserve aspect ratio; centre with letterbox on white.
                scale = min(page_w / img_w_px, page_h / img_h_px)
                draw_w = img_w_px * scale
                draw_h = img_h_px * scale
                offset_x = (page_w - draw_w) / 2
                offset_y = (page_h - draw_h) / 2
                rect = fitz.Rect(offset_x, offset_y, offset_x + draw_w, offset_y + draw_h)

            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=92, optimize=True)
            page.insert_image(rect, stream=buf.getvalue())

        out.save(out_path, garbage=4, deflate=True, clean=True)
    finally:
        out.close()

    return out_path, len(image_paths)


def convert_pdf_to_images(
    pdf_path: str,
    fmt: str = "png",
    dpi: int = 150,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[str, int]:
    """Render every page as an image and bundle the results into a ZIP.

    ``fmt`` is one of ``"png"`` or ``"jpg"``. ``dpi`` controls render resolution
    (72 = screen, 150 = balanced, 300 = print). Returns ``(zip_path, page_count)``.

    ``progress_cb(rendered, total)`` is called after each page is written
    to the archive. Optional — sync callers can omit it; the async Celery
    task uses it to surface live progress to SSE subscribers.
    """
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")
    fmt = (fmt or "png").lower()
    if fmt not in ("png", "jpg", "jpeg"):
        raise ValueError(f"Unsupported image format: {fmt}")
    if fmt == "jpeg":
        fmt = "jpg"
    if dpi < 36 or dpi > 600:
        raise ValueError("dpi must be between 36 and 600")

    out_dir = processed_dir()
    base = safe_basename(pdf_path)
    zip_path = os.path.join(out_dir, f"{base}_images_{fmt}_{dpi}dpi_{timestamp()}.zip")
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    page_count = 0
    with fitz.open(pdf_path) as doc:
        if doc.is_encrypted:
            raise ValueError("Cannot render an encrypted PDF — remove the password first")
        total = len(doc)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for idx, page in enumerate(doc, 1):
                _guard_pixmap_memory(page, zoom, idx)
                pix = page.get_pixmap(matrix=matrix, alpha=(fmt == "png"))
                if fmt == "png":
                    data = pix.tobytes("png")
                else:
                    # PyMuPDF can emit JPEG directly via Pillow.
                    img = PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=90, optimize=True)
                    data = buf.getvalue()
                zf.writestr(f"{base}_page_{idx:03d}.{fmt}", data)
                page_count += 1
                if progress_cb is not None:
                    progress_cb(page_count, total)

    return zip_path, page_count


METADATA_FIELDS: tuple[str, ...] = (
    "title",
    "author",
    "subject",
    "keywords",
    "creator",
    "producer",
)


def _parse_pdf_date(raw: str | None) -> str | None:
    """Convert a PDF date string (``D:YYYYMMDDHHmmSS+TZ``) to ISO ``YYYY-MM-DD HH:MM``.

    Returns ``None`` when ``raw`` is empty or unparseable.
    """
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("D:"):
        s = s[2:]
    # Strip trailing timezone marker like ``+02'00'`` or ``Z``.
    for marker in ("+", "-", "Z"):
        idx = s.find(marker)
        if idx >= 14:
            s = s[:idx]
            break
    if len(s) < 8 or not s[:8].isdigit():
        return None
    y, mo, d = s[0:4], s[4:6], s[6:8]
    h = s[8:10] if len(s) >= 10 and s[8:10].isdigit() else "00"
    mi = s[10:12] if len(s) >= 12 and s[10:12].isdigit() else "00"
    return f"{y}-{mo}-{d} {h}:{mi}"


def read_pdf_metadata(pdf_path: str) -> dict[str, str | None]:
    """Return the editable metadata fields plus formatted creation/mod dates.

    The result always contains keys: ``title``, ``author``, ``subject``,
    ``keywords``, ``creator``, ``producer``, ``creation_date``, ``mod_date``.
    Missing values are returned as ``None`` (or empty string for text fields).
    """
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")
    with fitz.open(pdf_path) as doc:
        meta = doc.metadata or {}
    return {
        "title": meta.get("title", "") or "",
        "author": meta.get("author", "") or "",
        "subject": meta.get("subject", "") or "",
        "keywords": meta.get("keywords", "") or "",
        "creator": meta.get("creator", "") or "",
        "producer": meta.get("producer", "") or "",
        "creation_date": _parse_pdf_date(meta.get("creationDate")),
        "mod_date": _parse_pdf_date(meta.get("modDate")),
    }


def edit_pdf_metadata(pdf_path: str, metadata: dict[str, str], clear_dates: bool = False) -> str:
    """Apply new metadata to ``pdf_path`` and write a new PDF.

    ``metadata`` may contain any of: ``title``, ``author``, ``subject``,
    ``keywords``, ``creator``, ``producer``. Empty strings clear the field;
    keys not present are left untouched. When ``clear_dates`` is true, both
    ``creationDate`` and ``modDate`` are wiped.
    """
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    out_dir = processed_dir()
    base = safe_basename(pdf_path)
    out_path = os.path.join(out_dir, f"{base}_metadata_{timestamp()}.pdf")

    with fitz.open(pdf_path) as doc:
        if doc.is_encrypted:
            raise ValueError("Cannot edit metadata on an encrypted PDF — remove the password first")
        current = dict(doc.metadata or {})
        for key in METADATA_FIELDS:
            if key in metadata:
                current[key] = metadata[key] or ""
        if clear_dates:
            current["creationDate"] = ""
            current["modDate"] = ""
        doc.set_metadata(current)
        doc.save(out_path, garbage=4, deflate=True, clean=True)

    return out_path


def render_page_thumbnail(pdf_path: str, page_number: int, max_width: int = 200) -> bytes:
    """Render a single page as a PNG thumbnail. ``page_number`` is 1-indexed."""
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    with fitz.open(pdf_path) as doc:
        total = len(doc)
        if page_number < 1 or page_number > total:
            raise ValueError(f"Invalid page number: {page_number} (PDF has {total} pages)")
        page = doc[page_number - 1]
        rect = page.rect
        zoom = max_width / rect.width if rect.width > 0 else 1.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        return bytes(pix.tobytes("png"))


def _calculate_position(
    position: str,
    page_width: float,
    page_height: float,
    content_width: float,
    content_height: float,
) -> tuple[float, float]:
    positions = {
        "top-left": (20, 20 + content_height),
        "top-center": ((page_width - content_width) / 2, 20 + content_height),
        "top-right": (page_width - content_width - 20, 20 + content_height),
        "center-left": (20, (page_height + content_height) / 2),
        "center": ((page_width - content_width) / 2, (page_height + content_height) / 2),
        "center-right": (page_width - content_width - 20, (page_height + content_height) / 2),
        "bottom-left": (20, page_height - 20),
        "bottom-center": ((page_width - content_width) / 2, page_height - 20),
        "bottom-right": (page_width - content_width - 20, page_height - 20),
    }
    return positions.get(position, positions["center"])


def add_watermark(
    pdf_path: str,
    watermark_type: str,
    watermark_content: str,
    options: dict | None = None,
) -> str:
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    options = options or {}
    position = options.get("position", "center")
    opacity = float(options.get("opacity", 0.3))
    rotation = int(options.get("rotation", 45))
    font_size = int(options.get("font_size", 48))

    out_dir = processed_dir()
    out_path = os.path.join(out_dir, f"watermarked_{safe_basename(pdf_path)}_{timestamp()}.pdf")

    temp_img_path: str | None = None
    wm_w = wm_h = 0

    with fitz.open(pdf_path) as doc:
        if watermark_type == "image":
            if not os.path.exists(watermark_content):
                raise ValueError(f"Watermark image not found: {watermark_content}")

            root, _ = os.path.splitext(watermark_content)
            temp_img_path = f"{root}_watermark_tmp_{timestamp()}.png"

            im = PILImage.open(watermark_content).convert("RGBA")

            max_size = 800
            if im.width > max_size or im.height > max_size:
                r = min(max_size / im.width, max_size / im.height)
                im = im.resize((int(im.width * r), int(im.height * r)), PILImage.Resampling.LANCZOS)

            if rotation:
                im = im.rotate(-rotation, expand=True, resample=PILImage.Resampling.BICUBIC)

            alpha = im.split()[3].point(lambda p: int(p * opacity))
            im.putalpha(alpha)

            im.save(temp_img_path, "PNG", optimize=True, compress_level=9)
            wm_w, wm_h = im.size

        for page in doc:
            pw, ph = page.rect.width, page.rect.height

            if watermark_type == "text":
                text = watermark_content
                tw = len(text) * font_size * 0.6
                th = font_size
                x, y = _calculate_position(position, pw, ph, tw, th)
                # PyMuPDF's insert_text only accepts rotations in {0, 90, 180, 270}.
                text_rotation = round(rotation / 90) * 90 % 360
                page.insert_text(
                    (x, y),
                    text,
                    fontsize=font_size,
                    fontname="hebo",
                    color=(0.65, 0.65, 0.65),
                    rotate=text_rotation,
                    overlay=True,
                )
            elif watermark_type == "image":
                max_w, max_h = pw * 0.3, ph * 0.3
                s = min(max_w / wm_w, max_h / wm_h, 1.0)
                w, h = wm_w * s, wm_h * s
                x, y = _calculate_position(position, pw, ph, w, h)
                page.insert_image(fitz.Rect(x, y, x + w, y + h), filename=temp_img_path, overlay=True)

        doc.save(out_path, garbage=4, deflate=True, clean=True)

    if temp_img_path and os.path.exists(temp_img_path):
        os.remove(temp_img_path)

    return out_path


def rotate_pages(pdf_path: str, rotation_angle: int, page_range: str | None = None) -> str:
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")
    if rotation_angle not in (90, 180, 270):
        raise ValueError("Rotation angle must be 90, 180, or 270 degrees")

    out_dir = processed_dir()
    out_path = os.path.join(out_dir, f"rotated_{safe_basename(pdf_path)}_{timestamp()}.pdf")

    with fitz.open(pdf_path) as doc:
        pages = parse_page_range(page_range, len(doc)) if page_range else list(range(len(doc)))
        for idx in pages:
            doc[idx].set_rotation(rotation_angle)
        doc.save(out_path, garbage=4, deflate=True, clean=True)

    return out_path


def add_page_numbers(pdf_path: str, options: dict | None = None) -> str:
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    options = options or {}
    position = options.get("position", "bottom-center")
    format_type = options.get("format", "number")
    font_size = int(options.get("font_size", 12))
    start_page = int(options.get("start_page", 1))

    out_dir = processed_dir()
    out_path = os.path.join(out_dir, f"numbered_{safe_basename(pdf_path)}_{timestamp()}.pdf")

    with fitz.open(pdf_path) as doc:
        total = len(doc)

        for page_idx, page in enumerate(doc):
            if page_idx < start_page - 1:
                continue

            pw, ph = page.rect.width, page.rect.height
            n = page_idx + 1

            if format_type == "page_number":
                text = f"Page {n}"
            elif format_type == "of_total":
                text = f"{n} of {total}"
            else:
                text = str(n)

            margin = 30
            tw = len(text) * font_size * 0.6

            coords = {
                "bottom-center": ((pw - tw) / 2, ph - margin),
                "bottom-left": (margin, ph - margin),
                "bottom-right": (pw - tw - margin, ph - margin),
                "top-center": ((pw - tw) / 2, margin + font_size),
                "top-left": (margin, margin + font_size),
                "top-right": (pw - tw - margin, margin + font_size),
            }
            x, y = coords.get(position, coords["bottom-center"])

            page.insert_text((x, y), text, fontsize=font_size, fontname="helv", color=(0, 0, 0), overlay=True)

        doc.save(out_path, garbage=4, deflate=True, clean=True)

    return out_path
