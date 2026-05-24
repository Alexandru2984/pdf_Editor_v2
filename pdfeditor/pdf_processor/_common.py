"""Shared helpers: paths, page ranges, font/color mapping."""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from datetime import datetime
from typing import Any

import fitz
from django.conf import settings
from django.utils.translation import gettext as _

logger = logging.getLogger(__name__)

# Document-catalog keys that make a PDF *do something* the moment a viewer
# opens it. /OpenAction and /AA (additional-actions) can fire embedded
# JavaScript or /Launch an external program with no user interaction —
# the active-content vector we refuse to persist for arbitrary uploads.
_AUTO_EXEC_CATALOG_KEYS = ("OpenAction", "AA")


def processed_dir() -> str:
    out_dir = os.path.join(settings.MEDIA_ROOT, "processed")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def pdf_page_count(pdf_path: str) -> int | None:
    """Return the page count or None if the file can't be opened.

    Used to gate sync vs async dispatch for heavy ops; cheap on typical
    PDFs (single-digit ms even for 100-page docs since fitz.open just
    parses the trailer + xref).
    """
    try:
        with fitz.open(pdf_path) as doc:
            return len(doc)
    except Exception:  # noqa: BLE001 — caller decides on None
        return None


def sanitize_pdf(pdf_path: str) -> bool:
    """Strip executable/auto-run structures from an uploaded PDF, in place.

    Removes embedded JavaScript and document-level auto-action hooks
    (``/OpenAction``, ``/AA``), then rewrites the file with full garbage
    collection so the xref table is rebuilt from scratch and any bytes
    appended after ``%%EOF`` (a classic polyglot trick) plus now-unreferenced
    objects are dropped. Page content, links, form fields and metadata are
    preserved — this targets only the active-content vectors, not the
    document the user actually uploaded.

    Returns ``True`` if the file was rewritten, ``False`` if it was skipped
    (e.g. password-protected, so we can't safely rewrite it) or failed.
    Best-effort: any failure leaves the original file untouched so a sanitize
    hiccup never blocks an otherwise-valid upload.
    """
    tmp_path: str | None = None
    try:
        with fitz.open(pdf_path) as doc:
            if doc.needs_pass:
                logger.info("Skipping sanitization of encrypted PDF %s", pdf_path)
                return False

            # 1. Drop all JavaScript (doc-level Names/JavaScript + annotation
            #    actions) while leaving content, links, metadata and filled
            #    form values intact.
            try:
                doc.scrub(
                    javascript=True,
                    attached_files=False,
                    clean_pages=False,
                    embedded_files=False,
                    hidden_text=False,
                    metadata=False,
                    redactions=False,
                    remove_links=False,
                    reset_fields=False,
                    reset_responses=False,
                    thumbnails=False,
                    xml_metadata=False,
                )
            except Exception as exc:  # noqa: BLE001 — scrub kwargs drift across versions
                logger.warning("scrub(javascript) failed on %s: %s", pdf_path, exc)

            # 2. Neutralise auto-execute hooks in the document catalog. Null'ing
            #    Names/JavaScript orphans the doc-level JS objects so the GC
            #    pass below drops them entirely rather than leaving empty shells.
            catalog = doc.pdf_catalog()
            for key in (*_AUTO_EXEC_CATALOG_KEYS, "Names/JavaScript"):
                # Key may simply not be present — that's fine.
                with contextlib.suppress(Exception):
                    doc.xref_set_key(catalog, key, "null")

            # 3. Rewrite with full GC: rebuilds the xref and discards trailing
            #    junk after %%EOF and any unreferenced objects.
            fd, tmp_path = tempfile.mkstemp(suffix=".pdf", dir=os.path.dirname(pdf_path))
            os.close(fd)
            doc.save(tmp_path, garbage=4, deflate=True, clean=True)
    except Exception as exc:  # noqa: BLE001 — never block upload on a sanitize failure
        logger.warning("Sanitization failed for %s: %s", pdf_path, exc)
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False

    os.replace(tmp_path, pdf_path)
    return True


def parse_page_range(range_string: str, total_pages: int) -> list[int]:
    """Parse "1-3,5,7-9" -> 0-indexed page list. Empty -> all pages."""
    if not range_string or not range_string.strip():
        return list(range(total_pages))

    pages: set[int] = set()
    for part in range_string.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start_idx, end_idx = int(start_s) - 1, int(end_s) - 1
            if start_idx < 0 or end_idx >= total_pages or start_idx > end_idx:
                raise ValueError(f"Invalid page range: {part}")
            pages.update(range(start_idx, end_idx + 1))
        else:
            p = int(part) - 1
            if p < 0 or p >= total_pages:
                raise ValueError(f"Page {part} out of range (1-{total_pages})")
            pages.add(p)

    return sorted(pages)


def check_pdf_has_text(pdf_path: str) -> tuple[bool, str]:
    if not os.path.exists(pdf_path):
        return False, _("PDF not found: %(path)s") % {"path": pdf_path}
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                if page.get_text().strip():
                    return True, _("PDF contains selectable text.")
        return False, _("PDF does not contain selectable text (possibly a scanned image-only document).")
    except Exception as e:
        return False, _("Error while checking PDF: %(err)s") % {"err": e}


BASE14_FONTS = {
    "helv",
    "hebo",
    "heit",
    "hebi",
    "tiro",
    "tibo",
    "tiri",
    "tibi",
    "cour",
    "cobo",
    "coit",
    "cobi",
    "symb",
    "zadb",
}


def convert_color(color_int: Any) -> tuple[float, float, float]:
    if isinstance(color_int, int):
        r = (color_int >> 16) & 0xFF
        g = (color_int >> 8) & 0xFF
        b = color_int & 0xFF
        return (r / 255.0, g / 255.0, b / 255.0)
    return (0.0, 0.0, 0.0)


def map_font_name(original_font: str) -> str:
    clean = (original_font or "").lower()
    if "+" in clean:
        clean = clean.split("+", 1)[1]
    clean2 = clean.replace("-", "").replace(" ", "")

    is_bold = any(x in clean2 for x in ("bold", "bd", "heavy", "black"))
    is_italic = any(x in clean2 for x in ("italic", "it", "oblique", "slant"))

    if any(x in clean2 for x in ("times", "timesnewroman", "timesroman")):
        if is_bold and is_italic:
            return "tibi"
        if is_bold:
            return "tibo"
        if is_italic:
            return "tiri"
        return "tiro"

    if any(x in clean2 for x in ("helvetica", "arial")):
        if is_bold and is_italic:
            return "hebi"
        if is_bold:
            return "hebo"
        if is_italic:
            return "heit"
        return "helv"

    if any(x in clean2 for x in ("courier", "couriernew", "mono")):
        if is_bold and is_italic:
            return "cobi"
        if is_bold:
            return "cobo"
        if is_italic:
            return "coit"
        return "cour"

    if "symbol" in clean2:
        return "symb"
    if "zapf" in clean2 or "dingbat" in clean2:
        return "zadb"

    return "helv"
