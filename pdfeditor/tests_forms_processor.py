"""Tests for pdf_processor.forms — AcroForm field detection + filling."""
import os
import shutil
import tempfile

import fitz
from django.test import TestCase, override_settings

from .pdf_processor.forms import (
    extract_form_fields,
    fill_form_fields,
    has_form_fields,
)


class _MediaRootMixin:
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._media_tmp = tempfile.mkdtemp(prefix="pdfedit_forms_")
        cls._media_override = override_settings(MEDIA_ROOT=cls._media_tmp)
        cls._media_override.enable()

    @classmethod
    def tearDownClass(cls):
        cls._media_override.disable()
        shutil.rmtree(cls._media_tmp, ignore_errors=True)
        super().tearDownClass()


def _make_form_pdf():
    """Build a 1-page PDF with a text field and a checkbox using PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Text widget for "name"
    text_widget = fitz.Widget()
    text_widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
    text_widget.field_name = "full_name"
    text_widget.rect = fitz.Rect(100, 100, 400, 130)
    text_widget.field_value = ""
    page.add_widget(text_widget)

    # Checkbox widget for "agree"
    cb_widget = fitz.Widget()
    cb_widget.field_type = fitz.PDF_WIDGET_TYPE_CHECKBOX
    cb_widget.field_name = "agree_terms"
    cb_widget.rect = fitz.Rect(100, 200, 120, 220)
    cb_widget.field_value = False
    page.add_widget(cb_widget)

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    doc.save(path)
    doc.close()
    return path


def _make_plain_pdf():
    doc = fitz.open()
    doc.new_page(width=400, height=400)
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    doc.save(path)
    doc.close()
    return path


class HasFormFieldsTests(TestCase):
    def test_detects_form_pdf(self):
        path = _make_form_pdf()
        try:
            self.assertTrue(has_form_fields(path))
        finally:
            os.remove(path)

    def test_returns_false_for_plain_pdf(self):
        path = _make_plain_pdf()
        try:
            self.assertFalse(has_form_fields(path))
        finally:
            os.remove(path)

    def test_returns_false_for_missing_file(self):
        self.assertFalse(has_form_fields("/no/such/file.pdf"))


class ExtractFormFieldsTests(TestCase):
    def setUp(self):
        self.path = _make_form_pdf()

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_extracts_text_and_checkbox(self):
        fields = extract_form_fields(self.path)
        names = sorted(f.name for f in fields)
        self.assertEqual(names, ["agree_terms", "full_name"])

        types = {f.name: f.field_type for f in fields}
        self.assertEqual(types["full_name"], "text")
        self.assertEqual(types["agree_terms"], "checkbox")

    def test_page_number_is_zero_indexed(self):
        fields = extract_form_fields(self.path)
        for f in fields:
            self.assertEqual(f.page_number, 0)

    def test_plain_pdf_has_no_fields(self):
        plain = _make_plain_pdf()
        try:
            self.assertEqual(extract_form_fields(plain), [])
        finally:
            os.remove(plain)

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            extract_form_fields("/no/such/file.pdf")


class FillFormFieldsTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.path = _make_form_pdf()

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_fills_text_field(self):
        out, count, warnings = fill_form_fields(self.path, {"full_name": "Alice Smith"})
        try:
            self.assertEqual(count, 1)
            with fitz.open(out) as d:
                page = d[0]
                widget = page.first_widget
                while widget is not None:
                    if widget.field_name == "full_name":
                        self.assertEqual(widget.field_value, "Alice Smith")
                        break
                    widget = widget.next
                else:
                    self.fail("full_name widget vanished after filling")
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_fills_checkbox_with_truthy_string(self):
        out, count, warnings = fill_form_fields(self.path, {"agree_terms": "yes"})
        try:
            self.assertEqual(count, 1)
            with fitz.open(out) as d:
                page = d[0]
                widget = page.first_widget
                while widget is not None:
                    if widget.field_name == "agree_terms":
                        # PyMuPDF returns "Yes"/"Off" or True/False depending on version.
                        self.assertIn(str(widget.field_value).lower(), ("yes", "true", "1"))
                        break
                    widget = widget.next
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_unknown_field_is_ignored(self):
        out, count, _w = fill_form_fields(self.path, {"nonexistent": "x"})
        try:
            self.assertEqual(count, 0)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_partial_fill_only_counts_matched(self):
        out, count, _w = fill_form_fields(
            self.path, {"full_name": "Bob", "nonexistent": "x"},
        )
        try:
            self.assertEqual(count, 1)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_filename_includes_filled_marker(self):
        out, _c, _w = fill_form_fields(self.path, {"full_name": "X"})
        try:
            self.assertIn("filled", os.path.basename(out))
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_flatten_bakes_text(self):
        out, _c, warnings = fill_form_fields(
            self.path, {"full_name": "FLAT TEXT"}, flatten=True,
        )
        try:
            with fitz.open(out) as d:
                page_text = d[0].get_text()
                self.assertIn("FLAT TEXT", page_text)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            fill_form_fields("/no/such/file.pdf", {"x": "y"})
