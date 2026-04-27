"""Tests for pdf_processor.extract — text-layer extraction + OCR fallback."""
import os
import shutil
import tempfile
from unittest import skipUnless

import fitz
from django.test import TestCase

from .pdf_processor.extract import extract_text_from_pdf, ocr_pdf_to_text


def _has_tesseract() -> bool:
    return shutil.which("tesseract") is not None


def _make_pdf(text_per_page):
    doc = fitz.open()
    for txt in text_per_page:
        page = doc.new_page(width=400, height=400)
        if txt:
            page.insert_text((50, 50), txt, fontsize=14)
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    doc.save(path)
    doc.close()
    return path


class ExtractTextFromPdfTests(TestCase):
    def setUp(self):
        self.path = _make_pdf(["First page text.", "Second page text."])

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_extracts_text_with_page_markers(self):
        text = extract_text_from_pdf(self.path)
        self.assertIn("=== Page 1 ===", text)
        self.assertIn("=== Page 2 ===", text)
        self.assertIn("First page text.", text)
        self.assertIn("Second page text.", text)

    def test_skips_pages_with_no_text(self):
        path = _make_pdf(["Only this one has text.", ""])
        try:
            text = extract_text_from_pdf(path)
            self.assertIn("=== Page 1 ===", text)
            self.assertNotIn("=== Page 2 ===", text)
        finally:
            os.remove(path)

    def test_empty_pdf_returns_helpful_message(self):
        path = _make_pdf(["", ""])
        try:
            text = extract_text_from_pdf(path)
            self.assertIn("OCR", text)
            self.assertIn("scanned", text.lower())
        finally:
            os.remove(path)

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            extract_text_from_pdf("/no/such/file.pdf")


class OcrPdfToTextTests(TestCase):
    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            ocr_pdf_to_text("/no/such/file.pdf")

    @skipUnless(_has_tesseract(), "tesseract binary not installed")
    def test_ocr_smoke_runs_without_crashing(self):
        # Render text large so OCR has a chance.
        doc = fitz.open()
        page = doc.new_page(width=600, height=400)
        page.insert_text((50, 200), "HELLO WORLD", fontsize=64)
        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        doc.save(path)
        doc.close()
        try:
            result = ocr_pdf_to_text(path)
            # Don't assert exact OCR output (varies by tesseract version);
            # just confirm it produced *some* page-marked content or the
            # informative fallback string.
            self.assertIsInstance(result, str)
            self.assertTrue(result)
        finally:
            os.remove(path)
