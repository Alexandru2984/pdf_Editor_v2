"""Split, merge, compress, watermark, rotate, page-numbers operations."""

from __future__ import annotations

import io
import logging
import os
import zipfile

import fitz
from PIL import Image as PILImage

from ._common import parse_page_range, processed_dir, safe_basename, timestamp


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
            validation_context = ValidationContext(allow_fetching=True)
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

                    vc = ValidationContext(
                        allow_fetching=True,
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


def convert_pdf_to_images(pdf_path: str, fmt: str = "png", dpi: int = 150) -> tuple[str, int]:
    """Render every page as an image and bundle the results into a ZIP.

    ``fmt`` is one of ``"png"`` or ``"jpg"``. ``dpi`` controls render resolution
    (72 = screen, 150 = balanced, 300 = print). Returns ``(zip_path, page_count)``.
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
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for idx, page in enumerate(doc, 1):
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

    return zip_path, page_count


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
