"""Shared helpers: paths, page ranges, font/color mapping."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import fitz
from django.conf import settings


def processed_dir() -> str:
    out_dir = os.path.join(settings.MEDIA_ROOT, "processed")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


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
        return False, f"PDF not found: {pdf_path}"
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                if page.get_text().strip():
                    return True, "PDF-ul conține text selectabil."
        return False, "PDF-ul nu conține text selectabil (posibil scanat doar cu imagini)."
    except Exception as e:
        return False, f"Eroare la verificarea PDF-ului: {e}"


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
