"""
PDF Processing Module - robust core logic for PDF operations + text editing.

Key goals:
- Stable output (SAFE mode) for text replacement based on a selected rectangle (bbox).
- Optional "FLOW mode": best-effort Word-like reflow inside a detected paragraph/column container,
  including shifting text blocks below in the same column.

Uses PyMuPDF (fitz). NOTE: PDF is not a flow document; reflow is heuristic.
"""

from __future__ import annotations

import os
import io
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Iterable, Any

import fitz  # PyMuPDF
from PIL import Image as PILImage
from django.conf import settings


# =========================
# Helpers: paths, pages, misc
# =========================

def _processed_dir() -> str:
    out_dir = os.path.join(settings.MEDIA_ROOT, "processed")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def parse_page_range(range_string: str, total_pages: int) -> List[int]:
    """
    Parse "1-3,5,7-9" => 0-indexed list of pages.
    Empty => all pages.
    """
    if not range_string or not range_string.strip():
        return list(range(total_pages))

    pages: set[int] = set()
    parts = range_string.replace(" ", "").split(",")

    for part in parts:
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            start_idx = start - 1
            end_idx = end - 1
            if start_idx < 0 or end_idx >= total_pages or start_idx > end_idx:
                raise ValueError(f"Invalid page range: {part}")
            pages.update(range(start_idx, end_idx + 1))
        else:
            p = int(part) - 1
            if p < 0 or p >= total_pages:
                raise ValueError(f"Page {part} out of range (1-{total_pages})")
            pages.add(p)

    return sorted(pages)


def check_pdf_has_text(pdf_path: str) -> Tuple[bool, str]:
    if not os.path.exists(pdf_path):
        return False, f"PDF not found: {pdf_path}"

    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                if page.get_text().strip():
                    return True, "PDF-ul conține text selectabil."
        return False, "PDF-ul nu conține text selectabil (posibil scanat doar cu imagini)."
    except Exception as e:
        return False, f"Eroare la verificarea PDF-ului: {str(e)}"


# =========================
# Split / Merge
# =========================

def split_pdf(pdf_path: str, ranges: List[Tuple[int, int]]) -> List[str]:
    """
    ranges: list of (start,end) 1-indexed inclusive
    """
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    out_dir = _processed_dir()
    base = _safe_basename(pdf_path)

    with fitz.open(pdf_path) as doc:
        total = len(doc)
        out_files: List[str] = []

        for idx, (start, end) in enumerate(ranges, 1):
            if start < 1 or end > total or start > end:
                raise ValueError(f"Invalid range: {start}-{end} (PDF has {total} pages)")

            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=start - 1, to_page=end - 1)

            suffix = f"pages_{start}-{end}" if len(ranges) == 1 else f"part{idx}_pages_{start}-{end}"
            out_path = os.path.join(out_dir, f"{base}_{suffix}_{_ts()}.pdf")

            new_doc.save(out_path, garbage=4, deflate=True, clean=True)
            new_doc.close()
            out_files.append(out_path)

        return out_files


def merge_pdfs(pdf_paths: List[str], output_name: Optional[str] = None) -> str:
    if len(pdf_paths) < 2:
        raise ValueError("At least 2 PDF files are required for merging")
    for p in pdf_paths:
        if not os.path.exists(p):
            raise ValueError(f"PDF file not found: {p}")

    out_dir = _processed_dir()
    if not output_name:
        output_name = f"merged_{_ts()}"
    out_path = os.path.join(out_dir, f"{output_name}.pdf")

    out = fitz.open()
    try:
        for p in pdf_paths:
            with fitz.open(p) as d:
                out.insert_pdf(d)
        out.save(out_path, garbage=4, deflate=True, clean=True)
        return out_path
    finally:
        out.close()


# =========================
# Compression (best-effort)
# =========================

def compress_pdf(pdf_path: str, quality: str = "medium", output_name: Optional[str] = None):
    """
    Best-effort compression:
    - deflate streams
    - try re-encode images to JPEG with requested quality (may not work for all xrefs)
    """
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    qmap = {"low": 50, "medium": 75, "high": 90}
    jpg_quality = int(qmap.get(quality, 75))

    out_dir = _processed_dir()
    if not output_name:
        output_name = f"compressed_{_safe_basename(pdf_path)}_{_ts()}"
    out_path = os.path.join(out_dir, f"{output_name}.pdf")

    original_size = os.path.getsize(pdf_path)

    with fitz.open(pdf_path) as doc:
        for page in doc:
            for img in page.get_images(full=True):
                xref = img[0]
                # Skip soft masks etc. (heuristic): if smask exists, often img[1] != 0
                # PyMuPDF tuple layout varies; safest: try/except.
                try:
                    info = doc.extract_image(xref)
                    img_bytes = info.get("image")
                    if not img_bytes:
                        continue

                    # Convert to JPEG
                    im = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=jpg_quality, optimize=True)
                    new_bytes = buf.getvalue()

                    doc.update_stream(xref, new_bytes)
                except Exception:
                    continue

        doc.save(out_path, garbage=4, deflate=True, clean=True)

    compressed_size = os.path.getsize(out_path)
    ratio = ((original_size - compressed_size) / original_size) * 100 if original_size else 0.0
    return out_path, original_size, compressed_size, ratio


# =========================
# Watermark / Rotate / Page Numbers
# =========================

def _calculate_position(position: str, page_width: float, page_height: float, content_width: float, content_height: float):
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


def add_watermark(pdf_path: str, watermark_type: str, watermark_content: str, options=None) -> str:
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    options = options or {}
    position = options.get("position", "center")
    opacity = float(options.get("opacity", 0.3))
    rotation = int(options.get("rotation", 45))
    font_size = int(options.get("font_size", 48))

    out_dir = _processed_dir()
    out_path = os.path.join(out_dir, f"watermarked_{_safe_basename(pdf_path)}_{_ts()}.pdf")

    temp_img_path: Optional[str] = None
    wm_w = wm_h = 0

    with fitz.open(pdf_path) as doc:
        if watermark_type == "image":
            if not os.path.exists(watermark_content):
                raise ValueError(f"Watermark image not found: {watermark_content}")

            root, _ = os.path.splitext(watermark_content)
            temp_img_path = f"{root}_watermark_tmp_{_ts()}.png"

            im = PILImage.open(watermark_content).convert("RGBA")

            max_size = 800
            if im.width > max_size or im.height > max_size:
                r = min(max_size / im.width, max_size / im.height)
                im = im.resize((int(im.width * r), int(im.height * r)), PILImage.LANCZOS)

            if rotation:
                im = im.rotate(-rotation, expand=True, resample=PILImage.BICUBIC)

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
                page.insert_text(
                    (x, y),
                    text,
                    fontsize=font_size,
                    fontname="hebo",
                    color=(0.65, 0.65, 0.65),
                    rotate=rotation,
                    overlay=True,
                )

            elif watermark_type == "image":
                max_w, max_h = pw * 0.3, ph * 0.3
                s = min(max_w / wm_w, max_h / wm_h, 1.0)
                w, h = wm_w * s, wm_h * s
                x, y = _calculate_position(position, pw, ph, w, h)
                rect = fitz.Rect(x, y, x + w, y + h)
                page.insert_image(rect, filename=temp_img_path, overlay=True)

        doc.save(out_path, garbage=4, deflate=True, clean=True)

    if temp_img_path and os.path.exists(temp_img_path):
        os.remove(temp_img_path)

    return out_path


def rotate_pages(pdf_path: str, rotation_angle: int, page_range: Optional[str] = None) -> str:
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")
    if rotation_angle not in (90, 180, 270):
        raise ValueError("Rotation angle must be 90, 180, or 270 degrees")

    out_dir = _processed_dir()
    out_path = os.path.join(out_dir, f"rotated_{_safe_basename(pdf_path)}_{_ts()}.pdf")

    with fitz.open(pdf_path) as doc:
        pages = parse_page_range(page_range, len(doc)) if page_range else list(range(len(doc)))
        for idx in pages:
            doc[idx].set_rotation(rotation_angle)
        doc.save(out_path, garbage=4, deflate=True, clean=True)

    return out_path


def add_page_numbers(pdf_path: str, options: Optional[dict] = None) -> str:
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    options = options or {}
    position = options.get("position", "bottom-center")
    format_type = options.get("format", "number")
    font_size = int(options.get("font_size", 12))
    start_page = int(options.get("start_page", 1))

    out_dir = _processed_dir()
    out_path = os.path.join(out_dir, f"numbered_{_safe_basename(pdf_path)}_{_ts()}.pdf")

    with fitz.open(pdf_path) as doc:
        total = len(doc)

        for page_idx, page in enumerate(doc):
            if page_idx < start_page - 1:
                continue

            pw, ph = page.rect.width, page.rect.height
            n = page_idx + 1

            if format_type == "number":
                text = str(n)
            elif format_type == "page_number":
                text = f"Page {n}"
            elif format_type == "of_total":
                text = f"{n} of {total}"
            else:
                text = str(n)

            margin = 30
            tw = len(text) * font_size * 0.6

            if position == "bottom-center":
                x, y = (pw - tw) / 2, ph - margin
            elif position == "bottom-left":
                x, y = margin, ph - margin
            elif position == "bottom-right":
                x, y = pw - tw - margin, ph - margin
            elif position == "top-center":
                x, y = (pw - tw) / 2, margin + font_size
            elif position == "top-left":
                x, y = margin, margin + font_size
            elif position == "top-right":
                x, y = pw - tw - margin, margin + font_size
            else:
                x, y = (pw - tw) / 2, ph - margin

            page.insert_text((x, y), text, fontsize=font_size, fontname="helv", color=(0, 0, 0), overlay=True)

        doc.save(out_path, garbage=4, deflate=True, clean=True)

    return out_path


# =========================
# Extract / OCR
# =========================

def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    with fitz.open(pdf_path) as doc:
        chunks: List[str] = []
        for i, page in enumerate(doc, 1):
            t = page.get_text()
            if t.strip():
                chunks.append(f"=== Page {i} ===\n{t}\n")

    if not chunks:
        return "No text found in PDF. This might be a scanned document - try OCR instead."

    return "\n".join(chunks)


def ocr_pdf_to_text(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:
        raise Exception("pytesseract not installed. Run: pip install pytesseract") from e

    with fitz.open(pdf_path) as doc:
        out: List[str] = []
        for i, page in enumerate(doc, 1):
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)
            if text.strip():
                out.append(f"=== Page {i} ===\n{text}\n")

    if not out:
        return "No text could be extracted via OCR. The document might be blank or poor quality."

    return "\n".join(out)


# =========================
# Font + Color mapping
# =========================

BASE14_FONTS = {
    "helv", "hebo", "heit", "hebi",
    "tiro", "tibo", "tiri", "tibi",
    "cour", "cobo", "coit", "cobi",
    "symb", "zadb",
}

def convert_color(color_int: Any) -> Tuple[float, float, float]:
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
        if is_bold and is_italic: return "tibi"
        if is_bold: return "tibo"
        if is_italic: return "tiri"
        return "tiro"

    if any(x in clean2 for x in ("helvetica", "arial")):
        if is_bold and is_italic: return "hebi"
        if is_bold: return "hebo"
        if is_italic: return "heit"
        return "helv"

    if any(x in clean2 for x in ("courier", "couriernew", "mono")):
        if is_bold and is_italic: return "cobi"
        if is_bold: return "cobo"
        if is_italic: return "coit"
        return "cour"

    if "symbol" in clean2:
        return "symb"
    if "zapf" in clean2 or "dingbat" in clean2:
        return "zadb"

    return "helv"


# =========================
# Text model for FLOW mode
# =========================

@dataclass
class SpanInfo:
    rect: fitz.Rect
    text: str
    font: str
    size: float
    color: Tuple[float, float, float]
    flags: int
    line_rect: fitz.Rect
    block_rect: fitz.Rect


@dataclass
class LineInfo:
    rect: fitz.Rect
    spans: List[SpanInfo]


@dataclass
class BlockInfo:
    rect: fitz.Rect
    lines: List[LineInfo]


def _iter_text_blocks(page: fitz.Page) -> List[BlockInfo]:
    d = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    blocks: List[BlockInfo] = []

    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        block_rect = fitz.Rect(b["bbox"])
        lines_out: List[LineInfo] = []

        for ln in b.get("lines", []):
            line_rect = fitz.Rect(ln["bbox"])
            spans_out: List[SpanInfo] = []

            for sp in ln.get("spans", []):
                t = sp.get("text", "")
                if not t or not t.strip():
                    continue
                sp_rect = fitz.Rect(sp["bbox"])
                spans_out.append(
                    SpanInfo(
                        rect=sp_rect,
                        text=t,
                        font=map_font_name(str(sp.get("font", "helv"))),
                        size=float(sp.get("size", 11)),
                        color=convert_color(sp.get("color", 0)),
                        flags=int(sp.get("flags", 0)),
                        line_rect=line_rect,
                        block_rect=block_rect,
                    )
                )

            if spans_out:
                lines_out.append(LineInfo(rect=line_rect, spans=spans_out))

        if lines_out:
            blocks.append(BlockInfo(rect=block_rect, lines=lines_out))

    return blocks


def _pick_style_near_rect(page: fitz.Page, rect: fitz.Rect) -> Tuple[str, float, Tuple[float, float, float]]:
    # Best effort: pick first span intersecting rect
    blocks = _iter_text_blocks(page)
    for b in blocks:
        for ln in b.lines:
            if not ln.rect.intersects(rect):
                continue
            for sp in ln.spans:
                if sp.rect.intersects(rect):
                    font = sp.font if sp.font in BASE14_FONTS else "helv"
                    size = max(6.0, min(48.0, sp.size))
                    return font, size, sp.color
    return "helv", 11.0, (0.0, 0.0, 0.0)


# =========================
# Coordinate conversion (frontend bbox)
# =========================

def rect_from_frontend_bbox(page: fitz.Page, bbox_bl: Dict[str, float]) -> fitz.Rect:
    """
    Frontend sends bbox in PDF standard bottom-left origin (BL) at scale=1.
    PyMuPDF uses top-left origin (TL). Convert BL -> TL.
    bbox_bl: {"x0","y0","x1","y1"} where y0<y1 in BL coordinates.
    """
    ph = page.rect.height
    x0 = float(bbox_bl["x0"])
    x1 = float(bbox_bl["x1"])
    y0_bl = float(bbox_bl["y0"])
    y1_bl = float(bbox_bl["y1"])
    y0_tl = ph - y1_bl
    y1_tl = ph - y0_bl
    return fitz.Rect(x0, y0_tl, x1, y1_tl)


# =========================
# SAFE mode: replace inside a rectangle (stable)
# =========================

def replace_in_rect_safe(
    page: fitz.Page,
    rect: fitz.Rect,
    new_text: str,
    align: int = fitz.TEXT_ALIGN_LEFT,
    shrink_steps: int = 6,
) -> List[str]:
    """
    Stable method:
    - redact rect
    - insert_textbox in same rect
    - shrink font if overflow
    """
    warnings: List[str] = []
    new_text = (new_text or "").strip()

    font, size, color = _pick_style_near_rect(page, rect)

    page.add_redact_annot(rect, fill=(1, 1, 1))
    page.apply_redactions()

    if not new_text:
        warnings.append("Replacement text was empty.")
        return warnings

    ok = False
    s = size
    for _ in range(max(1, shrink_steps)):
        rc = page.insert_textbox(
            rect,
            new_text,
            fontname=font,
            fontsize=s,
            color=color,
            align=align,
        )
        if rc >= 0:
            ok = True
            break
        s = max(6.0, s - 1.0)

    if not ok:
        warnings.append("Text did not fit in the selected area (even after shrinking).")

    return warnings


def rephrase_with_coordinates(
    pdf_path: str,
    page_number: int,
    bounding_box_bl: Dict[str, float],
    replace_text: str,
    mode: str = "flow",  # "safe" or "flow"
    original_text: str = "",  # optional, used by flow mode
) -> Tuple[str, int, List[str]]:
    """
    Entry point for UI selection replacement.
    mode:
      - "safe": replace only inside selection rect
      - "flow": best-effort reflow inside paragraph/column container and shift below content
    """
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    out_dir = _processed_dir()
    out_path = os.path.join(out_dir, f"{_safe_basename(pdf_path)}_rephrased_{mode}_{_ts()}.pdf")
    warnings: List[str] = []

    with fitz.open(pdf_path) as doc:
        if page_number < 0 or page_number >= len(doc):
            raise ValueError(f"Invalid page number: {page_number}")

        page = doc[page_number]
        sel_rect = rect_from_frontend_bbox(page, bounding_box_bl)

        if mode == "flow":
            w = replace_with_flow(page, sel_rect, replace_text, original_text=original_text)
            warnings.extend(w)
        else:
            warnings.extend(replace_in_rect_safe(page, sel_rect, replace_text))

        doc.save(out_path, garbage=4, deflate=True, clean=True)

    return out_path, 1, warnings


# =========================
# FLOW mode: paragraph/column container reflow + shift below
# =========================

def _line_gap(a: fitz.Rect, b: fitz.Rect) -> float:
    # gap between lines vertically (positive if b below a)
    return b.y0 - a.y1


def _get_line_size(line: LineInfo) -> float:
    if not line.spans:
        return 0.0
    # Return size of the first span (usually representative)
    return line.spans[0].size


def _group_paragraph_lines(lines: List[LineInfo], seed_idx: int, max_gap: float) -> List[int]:
    """
    Grow paragraph by including neighboring lines with small vertical gaps
    AND similar font sizes.
    """
    chosen = {seed_idx}
    seed_size = _get_line_size(lines[seed_idx])
    size_tolerance = 2.0  # allow small variations (e.g. bold vs regular might differ slightly)

    # Expand upwards
    i = seed_idx - 1
    while i >= 0:
        gap_ok = _line_gap(lines[i].rect, lines[i + 1].rect) <= max_gap
        size_ok = abs(_get_line_size(lines[i]) - seed_size) < size_tolerance
        
        # Check for list items (bullets) to avoid merging them into the paragraph
        text = lines[i].spans[0].text.strip() if lines[i].spans else ""
        is_list_item = text.startswith(("•", "-", "*", "1.", "2.", "3.", "a.", "b."))

        if gap_ok and size_ok and not is_list_item:
            chosen.add(i)
            i -= 1
        else:
            break

    # Expand downwards
    i = seed_idx + 1
    while i < len(lines):
        gap_ok = _line_gap(lines[i - 1].rect, lines[i].rect) <= max_gap
        size_ok = abs(_get_line_size(lines[i]) - seed_size) < size_tolerance
        
        text = lines[i].spans[0].text.strip() if lines[i].spans else ""
        is_list_item = text.startswith(("•", "-", "*", "1.", "2.", "3.", "a.", "b."))

        if gap_ok and size_ok and not is_list_item:
            chosen.add(i)
            i += 1
        else:
            break

    return sorted(chosen)


def _flatten_lines_in_reading_order(blocks: List[BlockInfo]) -> List[LineInfo]:
    """
    Flatten lines, roughly top->bottom, left->right.
    """
    lines: List[LineInfo] = []
    for b in blocks:
        lines.extend(b.lines)
    lines.sort(key=lambda ln: (round(ln.rect.y0, 2), round(ln.rect.x0, 2)))
    return lines


def _detect_container_for_selection(page: fitz.Page, sel_rect: fitz.Rect) -> Optional[Dict[str, Any]]:
    """
    Detect a 'flow container' (paragraph/column) around selection.

    Returns dict with:
    - container_rect (fitz.Rect)
    - lines (List[LineInfo]) included
    - style (font,size,color) from first intersecting span
    - align (left/center/right -> fitz align)
    """
    blocks = _iter_text_blocks(page)
    lines = _flatten_lines_in_reading_order(blocks)
    if not lines:
        return None

    # find seed line index intersecting selection (largest overlap)
    seed = None
    max_overlap = 0.0

    for i, ln in enumerate(lines):
        # intersection rect
        x0 = max(ln.rect.x0, sel_rect.x0)
        y0 = max(ln.rect.y0, sel_rect.y0)
        x1 = min(ln.rect.x1, sel_rect.x1)
        y1 = min(ln.rect.y1, sel_rect.y1)

        if x1 > x0 and y1 > y0:
            area = (x1 - x0) * (y1 - y0)
            if area > max_overlap:
                max_overlap = area
                seed = i

    if seed is None:
        return None

    # estimate typical line height and gap threshold
    lh = max(6.0, lines[seed].rect.height)
    max_gap = lh * 0.5  # heuristic: reduced from 0.9 to 0.5 to avoid merging separate paragraphs

    para_idxs = _group_paragraph_lines(lines, seed, max_gap=max_gap)

    para_lines = [lines[i] for i in para_idxs]
    # container rect = union of paragraph lines, but "columnized" a bit:
    x0 = min(ln.rect.x0 for ln in para_lines)
    x1 = max(ln.rect.x1 for ln in para_lines)
    y0 = min(ln.rect.y0 for ln in para_lines)
    y1 = max(ln.rect.y1 for ln in para_lines)
    container_rect = fitz.Rect(x0, y0, x1, y1)

    # Determine column-ish margins: expand slightly to match column width
    # Use page text blocks that overlap x-range to estimate consistent column.
    page_w = page.rect.width
    col_left = x0
    col_right = x1

    # expand toward nearest consistent margins in that x band (very conservative)
    col_left = max(0, col_left - 2)
    col_right = min(page_w, col_right + 2)
    container_rect = fitz.Rect(col_left, y0, col_right, y1)

    # style from seed line (most representative)
    seed_line = lines[seed]
    font = "helv"
    size = 11.0
    color = (0, 0, 0)
    if seed_line.spans:
        sp = seed_line.spans[0]
        font = sp.font if sp.font in BASE14_FONTS else "helv"
        size = sp.size
        color = sp.color

    # alignment heuristic based on line starts/ends
    # We analyze the variance of x0 and x1 across lines
    if not para_lines:
        align = fitz.TEXT_ALIGN_LEFT
    else:
        x0s = [ln.rect.x0 for ln in para_lines]
        x1s = [ln.rect.x1 for ln in para_lines]
        
        # Tolerance for "same position"
        tol = 2.0
        
        # Check if all start at same x0 (approx)
        min_x0 = min(x0s)
        starts_same = all(abs(x - min_x0) < tol for x in x0s)
        
        # Check if all end at same x1 (approx) - usually only true for justified text or single block
        # But for justified text, the last line often ends earlier. So we check if *most* lines end at same x1.
        max_x1 = max(x1s)
        # We consider "ends same" if > 80% of lines end near max_x1 (excluding last line if multiple)
        if len(para_lines) > 1:
            ends_same_count = sum(1 for x in x1s[:-1] if abs(x - max_x1) < tol)
            ends_same = (ends_same_count / (len(para_lines) - 1)) > 0.8
        else:
            ends_same = True # Single line
            
        if starts_same and ends_same:
            align = fitz.TEXT_ALIGN_JUSTIFY
        elif starts_same:
            align = fitz.TEXT_ALIGN_LEFT
        elif ends_same:
            align = fitz.TEXT_ALIGN_RIGHT
        else:
            # If neither, maybe centered?
            # Check if midpoints are aligned
            mids = [(ln.rect.x0 + ln.rect.x1)/2 for ln in para_lines]
            avg_mid = sum(mids) / len(mids)
            mids_same = all(abs(m - avg_mid) < tol for m in mids)
            
            if mids_same:
                align = fitz.TEXT_ALIGN_CENTER
            else:
                # Default to left if chaotic
                align = fitz.TEXT_ALIGN_LEFT

    return {
        "container_rect": container_rect,
        "para_lines": para_lines,
        "style": (font, size, color),
        "align": align,
    }


def _extract_text_from_lines(para_lines: List[LineInfo]) -> str:
    """
    Reconstruct paragraph text from lines/spans.
    We preserve intra-line span text; between lines we join with '\n'.
    """
    out_lines: List[str] = []
    for ln in para_lines:
        line_text = "".join(sp.text for sp in ln.spans)
        # normalize weird whitespace but keep line breaks
        out_lines.append(re.sub(r"[ \t]+", " ", line_text).rstrip())
    return "\n".join(out_lines).strip()


def _replace_substring_best_effort(text: str, old: str, new: str) -> Tuple[str, bool]:
    """
    Replace old->new once (best-effort). If old empty or not found, no replace.
    """
    old = (old or "").strip()
    if not old:
        return text, False

    if old in text:
        return text.replace(old, new, 1), True

    # Try whitespace-normalized match
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    t_norm = norm(text)
    o_norm = norm(old)
    if o_norm and o_norm in t_norm:
        # naive: replace in normalized then return normalized text (layout changes but better than nothing)
        return t_norm.replace(o_norm, norm(new), 1), True

    return text, False


def _erase_rect(page: fitz.Page, rect: fitz.Rect):
    page.add_redact_annot(rect, fill=(1, 1, 1))
    page.apply_redactions()


def _text_blocks_below_in_same_column(page: fitz.Page, container_rect: fitz.Rect) -> List[BlockInfo]:
    """
    Select text blocks below container that overlap container x-range significantly.
    (We avoid touching non-text blocks by design.)
    """
    blocks = _iter_text_blocks(page)
    out: List[BlockInfo] = []

    for b in blocks:
        if b.rect.y0 < container_rect.y1 - 1:
            continue  # not below
        # overlap in X
        overlap = min(b.rect.x1, container_rect.x1) - max(b.rect.x0, container_rect.x0)
        if overlap <= 0:
            continue
        overlap_ratio = overlap / max(1.0, min(b.rect.width, container_rect.width))
        if overlap_ratio >= 0.55:
            out.append(b)

    # sort top->bottom
    out.sort(key=lambda bb: (bb.rect.y0, bb.rect.x0))
    return out


def _shift_text_blocks(page: fitz.Page, blocks: List[BlockInfo], dy: float) -> List[str]:
    """
    Shift blocks by dy by:
    - erasing their union area
    - re-inserting their text with insert_textbox per block rect shifted

    WARNING: This is still heuristic; we keep it conservative.
    """
    warnings: List[str] = []
    if not blocks or abs(dy) < 0.5:
        return warnings

    # erase old blocks individually (safer than union, preserves graphics between blocks)
    for b in blocks:
        page.add_redact_annot(b.rect, fill=(1, 1, 1))
    page.apply_redactions()

    # re-insert each block using insert_text (exact positioning)
    # This avoids "text does not fit" errors for small blocks (like bullets)
    for b in blocks:
        for ln in b.lines:
            for sp in ln.spans:
                # Shift origin
                # sp.origin is (x, y) of the baseline
                # We need to reconstruct the point. 
                # PyMuPDF span doesn't give 'origin' directly in the dict output of get_text("dict"), 
                # but we can approximate or use bbox.
                # Wait, get_text("dict") spans have 'bbox' and 'origin'.
                # Let's check SpanInfo definition. It has 'rect' but not 'origin'.
                # We need to check how SpanInfo is populated in _iter_text_blocks.
                
                # Looking at _iter_text_blocks (lines 497+), it uses page.get_text("dict").
                # The raw span dict has "origin". We should probably store it in SpanInfo if we want to use it.
                # But we don't have it in SpanInfo currently.
                
                # Fallback: Use insert_textbox with MASSIVE expansion for small blocks?
                # Or better: Use insert_text at (x0, y1 - descent).
                # Since we don't have exact baseline, insert_textbox is safer for alignment IF it fits.
                # But it fails for small blocks.
                
                # Let's use insert_textbox but with a much larger rect, 
                # BUT we must ensure it doesn't wrap.
                # Actually, for a single line/span, we can just make the rect very wide.
                
                # Alternative: We can use the 'rect' (bbox) to estimate position.
                # Text is usually drawn starting near (x0, y1 - descent).
                # If we use insert_text(point, text, ...), point is usually bottom-left of the first char.
                
                # Let's try to stick with insert_textbox but make the rect HUGE in width/height 
                # to guarantee it fits, but rely on the fact that the text is short.
                # No, if we make it huge, it might overlap others? 
                # No, we are inserting into the page. Overlap is fine (it's just drawing).
                # The issue with insert_textbox is it might wrap if width is too small.
                
                # Let's try to use insert_text with an estimated baseline.
                # Baseline approx: y1 - (size * 0.2) ?
                # Or just use insert_textbox with a guaranteed sufficient width/height.
                
                # Let's try the "Guaranteed Fit" insert_textbox approach:
                # 1. Create a rect at the new position.
                # 2. Expand it significantly (e.g. width + 100, height + 50).
                # 3. Use render_mode? No.
                
                # Let's go with the "Huge Rect" strategy for each LINE.
                # We iterate lines, not blocks, to be safer.
                
                # Shift rect
                r = sp.rect
                new_r = fitz.Rect(r.x0, r.y0 + dy, r.x1 + 200, r.y1 + dy + 50) # Massive expansion
                
                font = sp.font if sp.font in BASE14_FONTS else "helv"
                size = sp.size
                color = sp.color
                
                rc = page.insert_textbox(
                    new_r,
                    sp.text,
                    fontname=font,
                    fontsize=size,
                    color=color,
                    align=fitz.TEXT_ALIGN_LEFT
                )
                
                if rc < 0:
                     warnings.append(f"Failed to shift text: '{sp.text[:10]}...'")

    return warnings


def replace_with_flow(page: fitz.Page, sel_rect: fitz.Rect, replace_text: str, original_text: str = "") -> List[str]:
    """
    FLOW mode:
    - detect paragraph/container around selection
    - reconstruct container text
    - replace substring (prefer original_text if provided)
    - erase container
    - re-insert container text via textbox (wrap)
    - shift blocks below in same column by delta height (best effort)
    """
    warnings: List[str] = []

    detected = _detect_container_for_selection(page, sel_rect)
    if not detected:
        # fallback to safe
        warnings.append("FLOW mode: could not detect a paragraph/container; falling back to SAFE.")
        warnings.extend(replace_in_rect_safe(page, sel_rect, replace_text))
        return warnings

    container: fitz.Rect = detected["container_rect"]
    para_lines: List[LineInfo] = detected["para_lines"]
    font, size, color = detected["style"]
    align = detected["align"]

    # old text from container
    old_text = _extract_text_from_lines(para_lines)

    # choose which 'old substring' to replace:
    # if original_text is given from selection, use it; else attempt using selection extraction
def _measure_text_height(
    width: float,
    text: str,
    fontname: str,
    fontsize: float,
    align: int = 0
) -> float:
    """
    Measure height required for text in a given width using a temporary page.
    """
    # Create a temp doc/page large enough
    temp_doc = fitz.open()
    temp_page = temp_doc.new_page(width=1000, height=5000)
    
    # Define a rect with target width and ample height
    # x0=0 is fine for measurement
    rect = fitz.Rect(0, 0, width, 5000)
    
    rc = temp_page.insert_textbox(
        rect,
        text,
        fontname=fontname,
        fontsize=fontsize,
        align=align
    )
    
    temp_doc.close()
    
    if rc < 0:
        # Should not happen with height=5000 unless text is massive
        return 5000.0
        
    # rc is unused vertical space
    used_height = 5000.0 - rc
    return used_height


def replace_with_flow(
    page: fitz.Page,
    sel_rect: fitz.Rect,
    replace_text: str,
    original_text: str = "",
) -> List[str]:
    """
    Flow mode:
    1. Detect container (paragraph/column)
    2. Measure new text height
    3. Shift content below by delta
    4. Insert new text
    """
    warnings: List[str] = []
    
    # 1. Detect container
    container_info = _detect_container_for_selection(page, sel_rect)
    if not container_info:
        warnings.append("FLOW mode: Could not detect text container. Falling back to SAFE mode.")
        warnings.extend(replace_in_rect_safe(page, sel_rect, replace_text))
        return warnings

    container = container_info["container_rect"]
    font, size, color = container_info["style"]
    align = container_info["align"]

    # 2. Measure new text height
    # Use the detected font/size to see how much space we NEED
    needed_height = _measure_text_height(
        width=container.width,
        text=replace_text,
        fontname=font if font in BASE14_FONTS else "helv",
        fontsize=size,
        align=align
    )
    
    # Add a small buffer (padding)
    needed_height += 2.0
    
    old_h = container.height
    delta = needed_height - old_h
    
    # 3. Shift content below
    # We shift if delta is significant (positive or negative)
    # If delta > 0, we push down. If delta < 0, we pull up (or just leave whitespace if safer).
    # Let's support both for true reflow.
    
    if abs(delta) >= 1.0:
        blocks_below = _text_blocks_below_in_same_column(page, container)
        if len(blocks_below) > 50: # increased limit
             warnings.append("FLOW mode: too many blocks below; skipping shifting to avoid breaking layout.")
        else:
             # Check if we need to resize the page height
             if delta > 0 and blocks_below:
                 # Find the lowest point of the blocks below
                 max_y = max(b.rect.y1 for b in blocks_below)
                 new_max_y = max_y + delta
                 
                 # If new bottom exceeds page height, resize page
                 if new_max_y > page.rect.height:
                     # Increase height
                     new_height = new_max_y + 36.0 # add margin
                     page.set_mediabox(fitz.Rect(0, 0, page.rect.width, new_height))
                     warnings.append(f"FLOW mode: Extended page height to {new_height:.1f}pt.")

             warnings.extend(_shift_text_blocks(page, blocks_below, dy=delta))
             warnings.append(f"FLOW mode: shifted content by {delta:.1f}pt.")

    # 4. Erase & Insert
    # We erase the ORIGINAL container area (plus any area we expanded into? No, just original is enough to clear old text)
    # Wait, if we shift blocks down, we clear the path?
    # _shift_text_blocks handles moving the blocks.
    # We just need to clear the OLD text in the container.
    _erase_rect(page, container)
    
    # Define the NEW container rect
    new_container = fitz.Rect(container.x0, container.y0, container.x1, container.y0 + needed_height)
    
    # Insert text
    rc = page.insert_textbox(
        new_container,
        replace_text,
        fontname=font if font in BASE14_FONTS else "helv",
        fontsize=size,
        color=color,
        align=align,
    )
    
    if rc < 0:
        warnings.append("FLOW mode: Text insertion reported overflow (unexpected).")

    return warnings


# =========================
# Text search replace (document-wide) - stable version
# =========================

def find_and_replace_text(
    pdf_path: str,
    search_text: str,
    replace_text: str,
    case_sensitive: bool = True,
    page_range: Optional[str] = None,
    mode: str = "safe",  # safe or flow
) -> Tuple[str, int, List[str]]:
    """
    Document-wide search & replace:
    - Locates matches using search_for (best for same-line matches).
    - For each match: apply SAFE replacement in a union rect (or FLOW on container).
    """
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    warnings: List[str] = []
    replacements = 0

    out_dir = _processed_dir()
    out_path = os.path.join(out_dir, f"{_safe_basename(pdf_path)}_findreplace_{mode}_{_ts()}.pdf")

    with fitz.open(pdf_path) as doc:
        pages = parse_page_range(page_range, len(doc)) if page_range else list(range(len(doc)))

        for pno in pages:
            page = doc[pno]

            # PyMuPDF doesn't provide perfect ignore-case search flags.
            # We'll handle case-insensitive by scanning page text and using search_for on exact casing
            # as a best-effort fallback.
            if case_sensitive:
                rects = page.search_for(search_text)
            else:
                # best-effort: try direct, then try common casing variants
                rects = page.search_for(search_text)
                if not rects:
                    rects = page.search_for(search_text.lower()) or page.search_for(search_text.upper())

            if not rects:
                continue

            # process from bottom to top
            rects_sorted = sorted(rects, key=lambda r: (r.y0, r.x0), reverse=True)

            for r in rects_sorted:
                # Expand a bit
                rr = fitz.Rect(r.x0 - 1, r.y0 - 1, r.x1 + 1, r.y1 + 1)

                if mode == "flow":
                    warnings.extend(replace_with_flow(page, rr, replace_text, original_text=search_text))
                else:
                    warnings.extend(replace_in_rect_safe(page, rr, replace_text))

                replacements += 1

        doc.save(out_path, garbage=4, deflate=True, clean=True)

    return out_path, replacements, warnings


def rephrase_text_in_pdf(
    pdf_path: str,
    search_text: str,
    replace_text: str,
    case_sensitive: bool = True,
    page_range: Optional[str] = None,
    mode: str = "flow",
) -> Tuple[str, int, List[str]]:
    """
    Alias that prefers FLOW mode for multi-line text,
    but will fall back to SAFE when container detection fails.
    """
    return find_and_replace_text(
        pdf_path=pdf_path,
        search_text=search_text,
        replace_text=replace_text,
        case_sensitive=case_sensitive,
        page_range=page_range,
        mode=mode,
    )
