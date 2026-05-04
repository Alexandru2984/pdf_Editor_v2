"""Split, merge, compress, watermark, rotate, page-numbers operations."""

from __future__ import annotations

import io
import os

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
