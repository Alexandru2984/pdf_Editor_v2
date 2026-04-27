"""Tests for pdf_processor._layout — paragraph/container detection + reflow geometry."""
import os
import tempfile

import fitz
from django.test import TestCase

from .pdf_processor._layout import (
    LineInfo,
    SpanInfo,
    _detect_alignment,
    _group_paragraph_lines,
    _is_list_item,
    detect_container_for_selection,
    extract_text_from_lines,
    iter_text_blocks,
    measure_text_height,
    pick_style_near_rect,
    rect_from_frontend_bbox,
    shift_text_blocks,
    text_blocks_below_in_same_column,
)


def _make_span(x0, y0, x1, y1, text="x", size=11.0, font="helv"):
    rect = fitz.Rect(x0, y0, x1, y1)
    return SpanInfo(
        rect=rect,
        text=text,
        font=font,
        size=size,
        color=(0.0, 0.0, 0.0),
        flags=0,
        line_rect=rect,
        block_rect=rect,
    )


def _make_line(x0, y0, x1, y1, text="x", size=11.0):
    sp = _make_span(x0, y0, x1, y1, text=text, size=size)
    return LineInfo(rect=fitz.Rect(x0, y0, x1, y1), spans=[sp])


def _make_pdf_paragraph(text_lines, x=72.0, y=100.0, line_h=14.0, size=11.0, font="helv"):
    """Build a single-page PDF with `text_lines` rendered at known coordinates."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    for i, line in enumerate(text_lines):
        page.insert_text((x, y + i * line_h), line, fontsize=size, fontname=font)
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    doc.save(path)
    doc.close()
    return path


class RectFromFrontendBboxTests(TestCase):
    def test_simple_conversion_inverts_y_axis(self):
        doc = fitz.open()
        page = doc.new_page(width=500, height=800)
        bbox_bl = {"x0": 50, "y0": 100, "x1": 150, "y1": 200}
        rect = rect_from_frontend_bbox(page, bbox_bl)

        # Top-left origin: y_top = page.height - y_bl_top
        self.assertAlmostEqual(rect.x0, 50)
        self.assertAlmostEqual(rect.x1, 150)
        self.assertAlmostEqual(rect.y0, 800 - 200)  # 600
        self.assertAlmostEqual(rect.y1, 800 - 100)  # 700
        doc.close()

    def test_full_page_round_trip(self):
        doc = fitz.open()
        page = doc.new_page(width=400, height=600)
        bbox_bl = {"x0": 0, "y0": 0, "x1": 400, "y1": 600}
        rect = rect_from_frontend_bbox(page, bbox_bl)
        self.assertEqual((rect.x0, rect.y0, rect.x1, rect.y1), (0, 0, 400, 600))
        doc.close()


class IsListItemTests(TestCase):
    def test_bullet_marker(self):
        self.assertTrue(_is_list_item(_make_line(0, 0, 10, 10, text="• item one")))

    def test_dash_marker(self):
        self.assertTrue(_is_list_item(_make_line(0, 0, 10, 10, text="- a thing")))

    def test_numeric_marker(self):
        self.assertTrue(_is_list_item(_make_line(0, 0, 10, 10, text="1. step one")))
        self.assertTrue(_is_list_item(_make_line(0, 0, 10, 10, text="  2. indented step")))

    def test_alpha_marker(self):
        self.assertTrue(_is_list_item(_make_line(0, 0, 10, 10, text="a. alpha")))

    def test_plain_text_is_not_list(self):
        self.assertFalse(_is_list_item(_make_line(0, 0, 10, 10, text="ordinary paragraph text")))

    def test_empty_line_is_not_list(self):
        empty = LineInfo(rect=fitz.Rect(0, 0, 0, 0), spans=[])
        self.assertFalse(_is_list_item(empty))


class DetectAlignmentTests(TestCase):
    def test_empty_returns_left(self):
        self.assertEqual(_detect_alignment([]), fitz.TEXT_ALIGN_LEFT)

    def test_left_aligned(self):
        # Same x0 across lines, varying x1.
        lines = [
            _make_line(50, 100, 200, 110),
            _make_line(50, 115, 180, 125),
            _make_line(50, 130, 220, 140),
        ]
        self.assertEqual(_detect_alignment(lines), fitz.TEXT_ALIGN_LEFT)

    def test_right_aligned(self):
        # Varying x0, same x1.
        lines = [
            _make_line(80, 100, 300, 110),
            _make_line(110, 115, 300, 125),
            _make_line(60, 130, 300, 140),
        ]
        self.assertEqual(_detect_alignment(lines), fitz.TEXT_ALIGN_RIGHT)

    def test_justified(self):
        # All lines start AND end at the same x.
        lines = [
            _make_line(50, 100, 300, 110),
            _make_line(50, 115, 300, 125),
            _make_line(50, 130, 300, 140),
        ]
        self.assertEqual(_detect_alignment(lines), fitz.TEXT_ALIGN_JUSTIFY)

    def test_centered(self):
        # Different x0/x1 but same midpoint.
        lines = [
            _make_line(100, 100, 300, 110),  # mid=200
            _make_line(120, 115, 280, 125),  # mid=200
            _make_line(140, 130, 260, 140),  # mid=200
        ]
        self.assertEqual(_detect_alignment(lines), fitz.TEXT_ALIGN_CENTER)

    def test_single_line_treated_as_justified(self):
        # starts_same trivially true, ends_same defaults true → justify.
        self.assertEqual(
            _detect_alignment([_make_line(50, 100, 300, 110)]),
            fitz.TEXT_ALIGN_JUSTIFY,
        )


class GroupParagraphLinesTests(TestCase):
    def test_grows_until_large_gap(self):
        # Three close-spaced lines, then a large gap, then more lines.
        lines = [
            _make_line(50, 100, 300, 112),
            _make_line(50, 115, 300, 127),
            _make_line(50, 130, 300, 142),
            _make_line(50, 200, 300, 212),  # large gap above
            _make_line(50, 215, 300, 227),
        ]
        # Seed in middle of first cluster, max_gap small.
        idxs = _group_paragraph_lines(lines, seed_idx=1, max_gap=10.0)
        self.assertEqual(idxs, [0, 1, 2])

    def test_size_change_breaks_paragraph(self):
        # Heading at top (larger), body below (smaller).
        lines = [
            _make_line(50, 100, 300, 120, size=18),  # heading
            _make_line(50, 130, 300, 142, size=11),  # body line 1
            _make_line(50, 145, 300, 157, size=11),  # body line 2
        ]
        idxs = _group_paragraph_lines(lines, seed_idx=1, max_gap=20.0)
        self.assertNotIn(0, idxs)  # heading excluded
        self.assertIn(1, idxs)
        self.assertIn(2, idxs)

    def test_list_item_breaks_paragraph(self):
        lines = [
            _make_line(50, 100, 300, 112, text="paragraph above"),
            _make_line(50, 115, 300, 127, text="• list item"),
            _make_line(50, 130, 300, 142, text="paragraph below seed"),
            _make_line(50, 145, 300, 157, text="continuation"),
        ]
        idxs = _group_paragraph_lines(lines, seed_idx=2, max_gap=10.0)
        self.assertEqual(idxs, [2, 3])  # blocked by list item at idx 1


class ExtractTextFromLinesTests(TestCase):
    def test_joins_spans_within_lines(self):
        line = LineInfo(
            rect=fitz.Rect(0, 0, 100, 10),
            spans=[
                _make_span(0, 0, 50, 10, text="Hello "),
                _make_span(50, 0, 100, 10, text="World"),
            ],
        )
        self.assertEqual(extract_text_from_lines([line]), "Hello World")

    def test_collapses_consecutive_whitespace(self):
        line = LineInfo(
            rect=fitz.Rect(0, 0, 100, 10),
            spans=[_make_span(0, 0, 100, 10, text="too   many    spaces")],
        )
        self.assertEqual(extract_text_from_lines([line]), "too many spaces")

    def test_joins_lines_with_newline_and_strips(self):
        l1 = LineInfo(rect=fitz.Rect(0, 0, 50, 10), spans=[_make_span(0, 0, 50, 10, text="line one  ")])
        l2 = LineInfo(rect=fitz.Rect(0, 12, 50, 22), spans=[_make_span(0, 12, 50, 22, text="line two")])
        self.assertEqual(extract_text_from_lines([l1, l2]), "line one\nline two")

    def test_empty_input(self):
        self.assertEqual(extract_text_from_lines([]), "")


class IterTextBlocksAndPickStyleTests(TestCase):
    def setUp(self):
        self.path = _make_pdf_paragraph(
            ["First line of paragraph.", "Second line continues.", "Third line ends it."],
            x=72, y=100, line_h=14, size=11,
        )

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_iter_text_blocks_extracts_lines_in_order(self):
        with fitz.open(self.path) as doc:
            blocks = iter_text_blocks(doc[0])

        self.assertGreaterEqual(len(blocks), 1)
        all_lines = [ln for b in blocks for ln in b.lines]
        self.assertEqual(len(all_lines), 3)
        # Lines are top-to-bottom; PDF coords increase downward.
        ys = [ln.rect.y0 for ln in all_lines]
        self.assertEqual(ys, sorted(ys))

    def test_iter_text_blocks_skips_empty_spans(self):
        # Sanity: every emitted span has non-empty stripped text.
        with fitz.open(self.path) as doc:
            blocks = iter_text_blocks(doc[0])
        for b in blocks:
            for ln in b.lines:
                for sp in ln.spans:
                    self.assertTrue(sp.text.strip())

    def test_pick_style_near_rect_returns_actual_size(self):
        with fitz.open(self.path) as doc:
            page = doc[0]
            blocks = iter_text_blocks(page)
            sample = blocks[0].lines[0].spans[0].rect
            font, size, color = pick_style_near_rect(page, sample)

        self.assertEqual(size, 11.0)
        self.assertEqual(color, (0.0, 0.0, 0.0))
        # Font may be remapped but must be a base-14 short code.
        self.assertEqual(len(font), 4)

    def test_pick_style_default_when_rect_misses(self):
        with fitz.open(self.path) as doc:
            page = doc[0]
            font, size, color = pick_style_near_rect(page, fitz.Rect(0, 0, 1, 1))

        self.assertEqual((font, size, color), ("helv", 11.0, (0.0, 0.0, 0.0)))


class DetectContainerTests(TestCase):
    def setUp(self):
        # Single paragraph of 4 closely-spaced lines.
        self.lines = [
            "This is the opening line of the paragraph.",
            "Here is a second line that continues it.",
            "And a third line further along.",
            "Final line wrapping it up.",
        ]
        self.path = _make_pdf_paragraph(self.lines, x=72, y=100, line_h=14, size=11)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_container_groups_all_paragraph_lines(self):
        with fitz.open(self.path) as doc:
            page = doc[0]
            # Pick a selection rectangle inside the second line.
            sel = fitz.Rect(80, 110, 200, 120)
            info = detect_container_for_selection(page, sel)

        self.assertIsNotNone(info)
        self.assertEqual(len(info["para_lines"]), 4)
        # Container should span the paragraph vertically.
        self.assertLess(info["container_rect"].y0, 100)  # at/above first line
        self.assertGreater(info["container_rect"].y1, 100 + 3 * 14)  # past last line

    def test_container_with_no_text_returns_none(self):
        doc = fitz.open()
        page = doc.new_page(width=500, height=500)
        info = detect_container_for_selection(page, fitz.Rect(10, 10, 100, 100))
        self.assertIsNone(info)
        doc.close()

    def test_selection_far_from_text_returns_none(self):
        with fitz.open(self.path) as doc:
            page = doc[0]
            # Bottom of the page, no text there.
            info = detect_container_for_selection(page, fitz.Rect(10, 800, 100, 830))
        self.assertIsNone(info)

    def test_style_extracted_from_seed(self):
        with fitz.open(self.path) as doc:
            page = doc[0]
            info = detect_container_for_selection(page, fitz.Rect(80, 110, 200, 120))
        font, size, color = info["style"]
        self.assertEqual(size, 11.0)
        self.assertEqual(color, (0.0, 0.0, 0.0))
        self.assertEqual(len(font), 4)


class MeasureTextHeightTests(TestCase):
    def test_more_text_needs_more_height(self):
        short = measure_text_height(width=200, text="hello", fontname="helv", fontsize=12)
        long = measure_text_height(
            width=200,
            text=("hello " * 200).strip(),
            fontname="helv",
            fontsize=12,
        )
        self.assertGreater(long, short)
        self.assertGreater(short, 0)

    def test_narrower_box_needs_more_height(self):
        text = "the quick brown fox jumps over the lazy dog " * 5
        wide = measure_text_height(width=400, text=text, fontname="helv", fontsize=11)
        narrow = measure_text_height(width=100, text=text, fontname="helv", fontsize=11)
        self.assertGreater(narrow, wide)

    def test_larger_font_needs_more_height(self):
        text = "the quick brown fox jumps over the lazy dog " * 3
        small = measure_text_height(width=200, text=text, fontname="helv", fontsize=8)
        large = measure_text_height(width=200, text=text, fontname="helv", fontsize=18)
        self.assertGreater(large, small)


class BlocksBelowAndShiftTests(TestCase):
    def setUp(self):
        # Top paragraph + bottom paragraph in the same column (same x range).
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), "Top paragraph line.", fontsize=11)
        page.insert_text((72, 200), "Bottom paragraph line one.", fontsize=11)
        page.insert_text((72, 215), "Bottom paragraph line two.", fontsize=11)
        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        doc.save(path)
        doc.close()
        self.path = path

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_blocks_below_finds_bottom_paragraph(self):
        with fitz.open(self.path) as doc:
            page = doc[0]
            top_block = iter_text_blocks(page)[0]
            below = text_blocks_below_in_same_column(page, top_block.rect)

        # Top paragraph is the seed; we should find at least one block below it.
        self.assertGreaterEqual(len(below), 1)
        for b in below:
            self.assertGreater(b.rect.y0, top_block.rect.y1 - 1)

    def test_blocks_below_excludes_non_overlapping_column(self):
        # Add a block off to the right that shouldn't be considered "below in same column".
        doc = fitz.open(self.path)
        page = doc[0]
        page.insert_text((400, 200), "right column text", fontsize=11)
        fd, path2 = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        doc.save(path2)
        doc.close()

        try:
            with fitz.open(path2) as d:
                page = d[0]
                blocks = iter_text_blocks(page)
                # The leftmost block at y≈100 is our top paragraph.
                top = min(blocks, key=lambda b: b.rect.y0)
                below = text_blocks_below_in_same_column(page, top.rect)

            xs = [b.rect.x0 for b in below]
            # No "right column" block (x0≈400) should be in the result.
            self.assertTrue(all(x < 200 for x in xs))
        finally:
            os.remove(path2)

    def test_shift_text_blocks_no_op_for_tiny_dy(self):
        with fitz.open(self.path) as doc:
            page = doc[0]
            blocks = iter_text_blocks(page)
            warnings = shift_text_blocks(page, blocks, dy=0.1)
        self.assertEqual(warnings, [])

    def test_shift_text_blocks_does_not_crash(self):
        with fitz.open(self.path) as doc:
            page = doc[0]
            top = iter_text_blocks(page)[0]
            below = text_blocks_below_in_same_column(page, top.rect)
            # Shouldn't throw and should return a list.
            warnings = shift_text_blocks(page, below, dy=20.0)
        self.assertIsInstance(warnings, list)
