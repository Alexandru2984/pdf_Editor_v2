"""Tests for pdf_processor.edit — SAFE + FLOW replacement, find_and_replace, rephrase_with_coordinates."""
import os
import shutil
import tempfile

import fitz
from django.test import TestCase, override_settings

from .pdf_processor.edit import (
    find_and_replace_text,
    rephrase_with_coordinates,
    replace_in_rect_safe,
    replace_with_flow,
)


def _normalize(text: str) -> str:
    """Strip whitespace so PyMuPDF's line-wrapping doesn't break substring checks."""
    return "".join(text.split())


def _read_page_text(path: str, page_idx: int = 0) -> str:
    with fitz.open(path) as doc:
        return doc[page_idx].get_text()


class _MediaRootMixin:
    """Redirects processed_dir() output to an isolated temp dir per test class."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._media_tmp = tempfile.mkdtemp(prefix="pdfedit_test_media_")
        cls._media_override = override_settings(MEDIA_ROOT=cls._media_tmp)
        cls._media_override.enable()

    @classmethod
    def tearDownClass(cls):
        cls._media_override.disable()
        shutil.rmtree(cls._media_tmp, ignore_errors=True)
        super().tearDownClass()


def _make_pdf(lines, x=72.0, y=100.0, line_h=14.0, size=11.0):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for i, line in enumerate(lines):
        page.insert_text((x, y + i * line_h), line, fontsize=size)
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    doc.save(path)
    doc.close()
    return path


class ReplaceInRectSafeTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.src = _make_pdf(["Original sentence here."])
        # Operate on a copy so the source is preserved.
        fd, self.work = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        shutil.copy(self.src, self.work)

    def tearDown(self):
        for p in (self.src, self.work):
            if os.path.exists(p):
                os.remove(p)

    def test_basic_replacement_writes_new_text(self):
        with fitz.open(self.work) as doc:
            page = doc[0]
            rect = fitz.Rect(70, 90, 350, 110)
            warnings = replace_in_rect_safe(page, rect, "Replacement done.")
            doc.saveIncr()

        self.assertEqual(warnings, [])
        text = _read_page_text(self.work)
        self.assertIn("Replacement", text)
        self.assertNotIn("Original", text)

    def test_empty_text_warns_but_redacts(self):
        with fitz.open(self.work) as doc:
            page = doc[0]
            rect = fitz.Rect(70, 90, 350, 110)
            warnings = replace_in_rect_safe(page, rect, "")
            doc.saveIncr()

        self.assertEqual(len(warnings), 1)
        self.assertIn("empty", warnings[0].lower())
        # The original text should be redacted.
        self.assertNotIn("Original", _read_page_text(self.work))

    def test_oversized_text_warns_about_overflow(self):
        # Tiny rect, very long replacement text — even shrunk, won't fit.
        with fitz.open(self.work) as doc:
            page = doc[0]
            rect = fitz.Rect(70, 90, 90, 100)  # ~20pt × 10pt
            warnings = replace_in_rect_safe(
                page, rect, "This is a very long replacement text that cannot possibly fit." * 5,
            )

        self.assertTrue(warnings)
        self.assertTrue(any("not fit" in w.lower() for w in warnings))


class ReplaceWithFlowTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.lines = [
            "Lorem ipsum dolor sit amet consectetur adipiscing.",
            "Sed do eiusmod tempor incididunt ut labore et dolore.",
            "Magna aliqua ut enim ad minim veniam quis nostrud.",
            "Exercitation ullamco laboris nisi ut aliquip ex ea.",
        ]
        self.src = _make_pdf(self.lines, x=72, y=100, line_h=14, size=11)
        fd, self.work = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        shutil.copy(self.src, self.work)

    def tearDown(self):
        for p in (self.src, self.work):
            if os.path.exists(p):
                os.remove(p)

    def test_basic_flow_replacement(self):
        with fitz.open(self.work) as doc:
            page = doc[0]
            sel = fitz.Rect(80, 110, 200, 120)
            warnings = replace_with_flow(page, sel, "Brief new text.", original_text="Lorem")
            doc.saveIncr()

        page_text = _read_page_text(self.work)
        self.assertIn("Brief", page_text)
        # Whatever warnings come back should be a list of strings.
        for w in warnings:
            self.assertIsInstance(w, str)

    def test_flow_falls_back_to_safe_when_no_container(self):
        # Empty PDF (no text) — container detection should fail.
        doc = fitz.open()
        doc.new_page(width=400, height=400)
        fd, blank = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        doc.save(blank)
        doc.close()

        try:
            with fitz.open(blank) as d:
                page = d[0]
                warnings = replace_with_flow(page, fitz.Rect(50, 50, 150, 70), "fresh text")
            self.assertTrue(any("could not detect" in w.lower() or "fallback" in w.lower() or "safe" in w.lower() for w in warnings))
        finally:
            os.remove(blank)

    def test_flow_extends_page_when_replacement_overflows(self):
        # Place paragraph near bottom + add a "below" block that will overflow when shifted.
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        # Top paragraph (the one we'll replace) — short.
        page.insert_text((72, 200), "Short top.", fontsize=11)
        # Bottom block sitting near the bottom of the page.
        for i, ln in enumerate(["Bottom one.", "Bottom two.", "Bottom three."]):
            page.insert_text((72, 800 + i * 14 - 28), ln, fontsize=11)
        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        doc.save(path)
        doc.close()

        try:
            with fitz.open(path) as d:
                page = d[0]
                original_height = page.rect.height
                sel = fitz.Rect(70, 195, 130, 210)
                long_replacement = ("This is a much much longer replacement that should "
                                    "force a downward shift of all blocks below it. ") * 8
                warnings = replace_with_flow(page, sel, long_replacement)
                d.saveIncr()

            with fitz.open(path) as d:
                new_height = d[0].rect.height

            self.assertGreater(new_height, original_height)
            # And we should have a warning announcing the page extension.
            self.assertTrue(any("extended page" in w.lower() for w in warnings))
        finally:
            os.remove(path)


class RephraseWithCoordinatesTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.src = _make_pdf(["Some text to rephrase here.", "Second line of paragraph."])

    def tearDown(self):
        if os.path.exists(self.src):
            os.remove(self.src)

    def test_happy_path_writes_output_with_new_text(self):
        # Bottom-left coords: a rect inside the first line near the top of an A4 page.
        bbox_bl = {"x0": 70, "y0": 842 - 110, "x1": 350, "y1": 842 - 95}
        out_path, count, warnings = rephrase_with_coordinates(
            pdf_path=self.src,
            page_number=0,
            bounding_box_bl=bbox_bl,
            replace_text="Brand new content here.",
            mode="flow",
        )
        try:
            self.assertEqual(count, 1)
            self.assertTrue(os.path.exists(out_path))
            text = _read_page_text(out_path)
            self.assertIn("Brand", text)
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_safe_mode_also_works(self):
        bbox_bl = {"x0": 70, "y0": 842 - 110, "x1": 350, "y1": 842 - 95}
        out_path, count, warnings = rephrase_with_coordinates(
            pdf_path=self.src,
            page_number=0,
            bounding_box_bl=bbox_bl,
            replace_text="Safe replacement.",
            mode="safe",
        )
        try:
            self.assertTrue(os.path.exists(out_path))
            self.assertIn("safe", out_path.lower())  # filename includes mode
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_invalid_page_number_raises(self):
        bbox_bl = {"x0": 0, "y0": 0, "x1": 100, "y1": 100}
        with self.assertRaises(ValueError):
            rephrase_with_coordinates(
                pdf_path=self.src,
                page_number=99,
                bounding_box_bl=bbox_bl,
                replace_text="x",
            )
        with self.assertRaises(ValueError):
            rephrase_with_coordinates(
                pdf_path=self.src,
                page_number=-1,
                bounding_box_bl=bbox_bl,
                replace_text="x",
            )

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            rephrase_with_coordinates(
                pdf_path="/no/such/file.pdf",
                page_number=0,
                bounding_box_bl={"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                replace_text="x",
            )


class FindAndReplaceTextFlowTests(_MediaRootMixin, TestCase):
    def setUp(self):
        # Two-page PDF, "TARGET" appears on each page.
        doc = fitz.open()
        for n in range(2):
            page = doc.new_page(width=595, height=842)
            page.insert_text((72, 100), f"Page {n+1}: TARGET appears here.", fontsize=11)
        fd, self.path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        doc.save(self.path)
        doc.close()

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_replaces_on_all_pages_in_safe_mode(self):
        out_path, count, warnings = find_and_replace_text(
            pdf_path=self.path, search_text="TARGET", replace_text="REPL",
            case_sensitive=True, mode="safe",
        )
        try:
            self.assertGreaterEqual(count, 2)
            with fitz.open(out_path) as doc:
                for p in doc:
                    norm = _normalize(p.get_text()).upper()
                    self.assertIn("REPL", norm)
                    self.assertNotIn("TARGET", norm)
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_page_range_restricts_replacements(self):
        out_path, count, warnings = find_and_replace_text(
            pdf_path=self.path, search_text="TARGET", replace_text="REPL",
            case_sensitive=True, page_range="1", mode="safe",
        )
        try:
            self.assertEqual(count, 1)
            with fitz.open(out_path) as doc:
                self.assertNotIn("TARGET", _normalize(doc[0].get_text()))
                self.assertIn("TARGET", _normalize(doc[1].get_text()))
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_no_matches_returns_zero_count(self):
        out_path, count, _w = find_and_replace_text(
            pdf_path=self.path, search_text="ZZZZ", replace_text="X",
            case_sensitive=True,
        )
        try:
            self.assertEqual(count, 0)
            self.assertTrue(os.path.exists(out_path))
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_flow_mode_smokes_does_not_crash(self):
        out_path, count, warnings = find_and_replace_text(
            pdf_path=self.path, search_text="TARGET", replace_text="reflow",
            case_sensitive=True, mode="flow",
        )
        try:
            self.assertGreaterEqual(count, 2)
            self.assertTrue(os.path.exists(out_path))
            self.assertIn("flow", out_path.lower())
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            find_and_replace_text(
                pdf_path="/no/such/file.pdf", search_text="x", replace_text="y",
            )

    def test_filename_contains_mode_and_basename(self):
        out_path, _c, _w = find_and_replace_text(
            pdf_path=self.path, search_text="TARGET", replace_text="X", mode="safe",
        )
        try:
            base = os.path.basename(self.path).rsplit(".", 1)[0]
            self.assertIn(base, os.path.basename(out_path))
            self.assertIn("findreplace", os.path.basename(out_path))
            self.assertIn("safe", os.path.basename(out_path))
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)
