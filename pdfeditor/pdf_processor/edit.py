"""Text editing: find/replace and coordinate-based rephrase (SAFE + FLOW modes)."""

from __future__ import annotations

import os

import fitz

from ._common import BASE14_FONTS, parse_page_range, processed_dir, safe_basename, timestamp
from ._layout import (
    detect_container_for_selection,
    erase_rect,
    measure_text_height,
    pick_style_near_rect,
    rect_from_frontend_bbox,
    shift_text_blocks,
    text_blocks_below_in_same_column,
)


def replace_in_rect_safe(
    page: fitz.Page,
    rect: fitz.Rect,
    new_text: str,
    align: int = fitz.TEXT_ALIGN_LEFT,
    shrink_steps: int = 6,
) -> list[str]:
    """SAFE mode: redact the rect, insert new text at the same location, shrinking if needed."""
    warnings: list[str] = []
    new_text = (new_text or "").strip()
    font, size, color = pick_style_near_rect(page, rect)

    page.add_redact_annot(rect, fill=(1, 1, 1))
    page.apply_redactions()

    if not new_text:
        warnings.append("Replacement text was empty.")
        return warnings

    s = size
    for _ in range(max(1, shrink_steps)):
        rc = page.insert_textbox(rect, new_text, fontname=font, fontsize=s, color=color, align=align)
        if rc >= 0:
            return warnings
        s = max(6.0, s - 1.0)

    warnings.append("Text did not fit in the selected area (even after shrinking).")
    return warnings


def replace_with_flow(
    page: fitz.Page,
    sel_rect: fitz.Rect,
    replace_text: str,
    original_text: str = "",
) -> list[str]:
    """FLOW mode: detect paragraph container, reflow text, shift content below by delta."""
    warnings: list[str] = []

    info = detect_container_for_selection(page, sel_rect)
    if not info:
        warnings.append("FLOW mode: could not detect a paragraph/container; falling back to SAFE.")
        warnings.extend(replace_in_rect_safe(page, sel_rect, replace_text))
        return warnings

    container: fitz.Rect = info["container_rect"]
    font, size, color = info["style"]
    align = info["align"]
    safe_font = font if font in BASE14_FONTS else "helv"

    needed_height = (
        measure_text_height(
            width=container.width,
            text=replace_text,
            fontname=safe_font,
            fontsize=size,
            align=align,
        )
        + 2.0
    )  # padding

    delta = needed_height - container.height

    if abs(delta) >= 1.0:
        blocks_below = text_blocks_below_in_same_column(page, container)
        if len(blocks_below) > 50:
            warnings.append("FLOW mode: too many blocks below; skipping shifting to avoid breaking layout.")
        else:
            if delta > 0 and blocks_below:
                max_y = max(b.rect.y1 for b in blocks_below)
                new_max_y = max_y + delta
                if new_max_y > page.rect.height:
                    new_height = new_max_y + 36.0
                    page.set_mediabox(fitz.Rect(0, 0, page.rect.width, new_height))
                    warnings.append(f"FLOW mode: Extended page height to {new_height:.1f}pt.")

            warnings.extend(shift_text_blocks(page, blocks_below, dy=delta))
            warnings.append(f"FLOW mode: shifted content by {delta:.1f}pt.")

    erase_rect(page, container)

    new_container = fitz.Rect(container.x0, container.y0, container.x1, container.y0 + needed_height)
    rc = page.insert_textbox(
        new_container,
        replace_text,
        fontname=safe_font,
        fontsize=size,
        color=color,
        align=align,
    )
    if rc < 0:
        warnings.append("FLOW mode: Text insertion reported overflow (unexpected).")

    return warnings


def rephrase_with_coordinates(
    pdf_path: str,
    page_number: int,
    bounding_box_bl: dict[str, float],
    replace_text: str,
    mode: str = "flow",
    original_text: str = "",
) -> tuple[str, int, list[str]]:
    """Apply a coordinate-based text replacement from a UI selection."""
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    out_dir = processed_dir()
    out_path = os.path.join(out_dir, f"{safe_basename(pdf_path)}_rephrased_{mode}_{timestamp()}.pdf")
    warnings: list[str] = []

    with fitz.open(pdf_path) as doc:
        if page_number < 0 or page_number >= len(doc):
            raise ValueError(f"Invalid page number: {page_number}")

        page = doc[page_number]
        sel_rect = rect_from_frontend_bbox(page, bounding_box_bl)

        if mode == "flow":
            warnings.extend(replace_with_flow(page, sel_rect, replace_text, original_text=original_text))
        else:
            warnings.extend(replace_in_rect_safe(page, sel_rect, replace_text))

        doc.save(out_path, garbage=4, deflate=True, clean=True)

    return out_path, 1, warnings


def find_and_replace_text(
    pdf_path: str,
    search_text: str,
    replace_text: str,
    case_sensitive: bool = True,
    page_range: str | None = None,
    mode: str = "safe",
) -> tuple[str, int, list[str]]:
    """Document-wide search & replace using PyMuPDF's search_for + SAFE/FLOW replacement."""
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    warnings: list[str] = []
    replacements = 0

    out_dir = processed_dir()
    out_path = os.path.join(out_dir, f"{safe_basename(pdf_path)}_findreplace_{mode}_{timestamp()}.pdf")

    with fitz.open(pdf_path) as doc:
        pages = parse_page_range(page_range, len(doc)) if page_range else list(range(len(doc)))

        for pno in pages:
            page = doc[pno]

            rects = page.search_for(search_text)
            if not rects and not case_sensitive:
                rects = page.search_for(search_text.lower()) or page.search_for(search_text.upper())

            if not rects:
                continue

            rects_sorted = sorted(rects, key=lambda r: (r.y0, r.x0), reverse=True)

            for r in rects_sorted:
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
    page_range: str | None = None,
    mode: str = "flow",
) -> tuple[str, int, list[str]]:
    """Alias of find_and_replace_text that defaults to FLOW mode."""
    return find_and_replace_text(
        pdf_path=pdf_path,
        search_text=search_text,
        replace_text=replace_text,
        case_sensitive=case_sensitive,
        page_range=page_range,
        mode=mode,
    )
