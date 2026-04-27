"""Text layout model + container/paragraph detection for FLOW mode replacement."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import fitz

from ._common import BASE14_FONTS, convert_color, map_font_name


@dataclass
class SpanInfo:
    rect: fitz.Rect
    text: str
    font: str
    size: float
    color: tuple[float, float, float]
    flags: int
    line_rect: fitz.Rect
    block_rect: fitz.Rect


@dataclass
class LineInfo:
    rect: fitz.Rect
    spans: list[SpanInfo]


@dataclass
class BlockInfo:
    rect: fitz.Rect
    lines: list[LineInfo]


def iter_text_blocks(page: fitz.Page) -> list[BlockInfo]:
    d = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    blocks: list[BlockInfo] = []

    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        block_rect = fitz.Rect(b["bbox"])
        lines_out: list[LineInfo] = []

        for ln in b.get("lines", []):
            line_rect = fitz.Rect(ln["bbox"])
            spans_out: list[SpanInfo] = []

            for sp in ln.get("spans", []):
                t = sp.get("text", "")
                if not t or not t.strip():
                    continue
                spans_out.append(
                    SpanInfo(
                        rect=fitz.Rect(sp["bbox"]),
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


def pick_style_near_rect(page: fitz.Page, rect: fitz.Rect) -> tuple[str, float, tuple[float, float, float]]:
    for b in iter_text_blocks(page):
        for ln in b.lines:
            if not ln.rect.intersects(rect):
                continue
            for sp in ln.spans:
                if sp.rect.intersects(rect):
                    font = sp.font if sp.font in BASE14_FONTS else "helv"
                    size = max(6.0, min(48.0, sp.size))
                    return font, size, sp.color
    return "helv", 11.0, (0.0, 0.0, 0.0)


def rect_from_frontend_bbox(page: fitz.Page, bbox_bl: dict[str, float]) -> fitz.Rect:
    """Convert PDF bottom-left origin bbox to PyMuPDF top-left origin rect."""
    ph = page.rect.height
    x0 = float(bbox_bl["x0"])
    x1 = float(bbox_bl["x1"])
    y0_bl = float(bbox_bl["y0"])
    y1_bl = float(bbox_bl["y1"])
    return fitz.Rect(x0, ph - y1_bl, x1, ph - y0_bl)


def _line_gap(a: fitz.Rect, b: fitz.Rect) -> float:
    return float(b.y0 - a.y1)


def _get_line_size(line: LineInfo) -> float:
    return line.spans[0].size if line.spans else 0.0


_LIST_PREFIXES = ("•", "-", "*", "1.", "2.", "3.", "a.", "b.")


def _is_list_item(line: LineInfo) -> bool:
    if not line.spans:
        return False
    return line.spans[0].text.strip().startswith(_LIST_PREFIXES)


def _group_paragraph_lines(lines: list[LineInfo], seed_idx: int, max_gap: float) -> list[int]:
    """Grow paragraph from seed line, stopping at large gaps, size changes, or list items."""
    chosen = {seed_idx}
    seed_size = _get_line_size(lines[seed_idx])
    size_tolerance = 2.0

    i = seed_idx - 1
    while i >= 0:
        if (
            _line_gap(lines[i].rect, lines[i + 1].rect) <= max_gap
            and abs(_get_line_size(lines[i]) - seed_size) < size_tolerance
            and not _is_list_item(lines[i])
        ):
            chosen.add(i)
            i -= 1
        else:
            break

    i = seed_idx + 1
    while i < len(lines):
        if (
            _line_gap(lines[i - 1].rect, lines[i].rect) <= max_gap
            and abs(_get_line_size(lines[i]) - seed_size) < size_tolerance
            and not _is_list_item(lines[i])
        ):
            chosen.add(i)
            i += 1
        else:
            break

    return sorted(chosen)


def _flatten_lines_in_reading_order(blocks: list[BlockInfo]) -> list[LineInfo]:
    lines: list[LineInfo] = []
    for b in blocks:
        lines.extend(b.lines)
    lines.sort(key=lambda ln: (round(ln.rect.y0, 2), round(ln.rect.x0, 2)))
    return lines


def _detect_alignment(para_lines: list[LineInfo]) -> int:
    if not para_lines:
        return int(fitz.TEXT_ALIGN_LEFT)

    x0s = [ln.rect.x0 for ln in para_lines]
    x1s = [ln.rect.x1 for ln in para_lines]
    tol = 2.0

    starts_same = all(abs(x - min(x0s)) < tol for x in x0s)
    max_x1 = max(x1s)
    if len(para_lines) > 1:
        ends_same_count = sum(1 for x in x1s[:-1] if abs(x - max_x1) < tol)
        ends_same = (ends_same_count / (len(para_lines) - 1)) > 0.8
    else:
        ends_same = True

    if starts_same and ends_same:
        return int(fitz.TEXT_ALIGN_JUSTIFY)
    if starts_same:
        return int(fitz.TEXT_ALIGN_LEFT)
    if ends_same:
        return int(fitz.TEXT_ALIGN_RIGHT)

    mids = [(ln.rect.x0 + ln.rect.x1) / 2 for ln in para_lines]
    avg_mid = sum(mids) / len(mids)
    if all(abs(m - avg_mid) < tol for m in mids):
        return int(fitz.TEXT_ALIGN_CENTER)

    return int(fitz.TEXT_ALIGN_LEFT)


def detect_container_for_selection(page: fitz.Page, sel_rect: fitz.Rect) -> dict[str, Any] | None:
    """Detect paragraph/column container around a selection rectangle."""
    blocks = iter_text_blocks(page)
    lines = _flatten_lines_in_reading_order(blocks)
    if not lines:
        return None

    seed = None
    max_overlap = 0.0
    for i, ln in enumerate(lines):
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

    lh = max(6.0, lines[seed].rect.height)
    max_gap = lh * 0.5

    para_idxs = _group_paragraph_lines(lines, seed, max_gap=max_gap)
    para_lines = [lines[i] for i in para_idxs]

    x0 = max(0, min(ln.rect.x0 for ln in para_lines) - 2)
    x1 = min(page.rect.width, max(ln.rect.x1 for ln in para_lines) + 2)
    y0 = min(ln.rect.y0 for ln in para_lines)
    y1 = max(ln.rect.y1 for ln in para_lines)
    container_rect = fitz.Rect(x0, y0, x1, y1)

    seed_line = lines[seed]
    font = "helv"
    size = 11.0
    color = (0.0, 0.0, 0.0)
    if seed_line.spans:
        sp = seed_line.spans[0]
        font = sp.font if sp.font in BASE14_FONTS else "helv"
        size = sp.size
        color = sp.color

    return {
        "container_rect": container_rect,
        "para_lines": para_lines,
        "style": (font, size, color),
        "align": _detect_alignment(para_lines),
    }


def extract_text_from_lines(para_lines: list[LineInfo]) -> str:
    out_lines: list[str] = []
    for ln in para_lines:
        line_text = "".join(sp.text for sp in ln.spans)
        out_lines.append(re.sub(r"[ \t]+", " ", line_text).rstrip())
    return "\n".join(out_lines).strip()


def erase_rect(page: fitz.Page, rect: fitz.Rect) -> None:
    page.add_redact_annot(rect, fill=(1, 1, 1))
    page.apply_redactions()


def text_blocks_below_in_same_column(page: fitz.Page, container_rect: fitz.Rect) -> list[BlockInfo]:
    """Return text blocks below container that share >= 55% horizontal overlap."""
    out: list[BlockInfo] = []
    for b in iter_text_blocks(page):
        if b.rect.y0 < container_rect.y1 - 1:
            continue
        overlap = min(b.rect.x1, container_rect.x1) - max(b.rect.x0, container_rect.x0)
        if overlap <= 0:
            continue
        overlap_ratio = overlap / max(1.0, min(b.rect.width, container_rect.width))
        if overlap_ratio >= 0.55:
            out.append(b)
    out.sort(key=lambda bb: (bb.rect.y0, bb.rect.x0))
    return out


def shift_text_blocks(page: fitz.Page, blocks: list[BlockInfo], dy: float) -> list[str]:
    """Shift blocks vertically by dy (erase + re-insert). Heuristic, conservative."""
    warnings: list[str] = []
    if not blocks or abs(dy) < 0.5:
        return warnings

    for b in blocks:
        page.add_redact_annot(b.rect, fill=(1, 1, 1))
    page.apply_redactions()

    for b in blocks:
        for ln in b.lines:
            for sp in ln.spans:
                r = sp.rect
                new_r = fitz.Rect(r.x0, r.y0 + dy, r.x1 + 200, r.y1 + dy + 50)
                font = sp.font if sp.font in BASE14_FONTS else "helv"

                rc = page.insert_textbox(
                    new_r,
                    sp.text,
                    fontname=font,
                    fontsize=sp.size,
                    color=sp.color,
                    align=fitz.TEXT_ALIGN_LEFT,
                )
                if rc < 0:
                    warnings.append(f"Failed to shift text: '{sp.text[:10]}...'")

    return warnings


def measure_text_height(width: float, text: str, fontname: str, fontsize: float, align: int = 0) -> float:
    """Measure height required to render `text` at given width via a throwaway page."""
    temp_doc = fitz.open()
    try:
        temp_page = temp_doc.new_page(width=1000, height=5000)
        rect = fitz.Rect(0, 0, width, 5000)
        rc = temp_page.insert_textbox(rect, text, fontname=fontname, fontsize=fontsize, align=align)
        if rc < 0:
            return 5000.0
        return float(5000.0 - rc)
    finally:
        temp_doc.close()
