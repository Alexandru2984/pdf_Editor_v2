"""Tests for pdf_processor._common (pure helpers + introspection)."""

import os
import tempfile

import fitz
from django.test import TestCase

from .pdf_processor._common import (
    BASE14_FONTS,
    check_pdf_has_text,
    convert_color,
    map_font_name,
    parse_page_range,
    processed_dir,
    safe_basename,
    sanitize_pdf,
    timestamp,
)


class ConvertColorTests(TestCase):
    def test_black_is_zero(self):
        self.assertEqual(convert_color(0), (0.0, 0.0, 0.0))

    def test_pure_red(self):
        r, g, b = convert_color(0xFF0000)
        self.assertAlmostEqual(r, 1.0)
        self.assertEqual((g, b), (0.0, 0.0))

    def test_pure_green(self):
        r, g, b = convert_color(0x00FF00)
        self.assertAlmostEqual(g, 1.0)
        self.assertEqual((r, b), (0.0, 0.0))

    def test_pure_blue(self):
        r, g, b = convert_color(0x0000FF)
        self.assertAlmostEqual(b, 1.0)
        self.assertEqual((r, g), (0.0, 0.0))

    def test_white_is_one(self):
        self.assertEqual(convert_color(0xFFFFFF), (1.0, 1.0, 1.0))

    def test_mid_gray(self):
        r, g, b = convert_color(0x808080)
        for c in (r, g, b):
            self.assertAlmostEqual(c, 128 / 255.0)

    def test_non_int_returns_black(self):
        self.assertEqual(convert_color("not-an-int"), (0.0, 0.0, 0.0))
        self.assertEqual(convert_color(None), (0.0, 0.0, 0.0))
        self.assertEqual(convert_color(3.14), (0.0, 0.0, 0.0))


class MapFontNameTests(TestCase):
    def test_helvetica_plain(self):
        self.assertEqual(map_font_name("Helvetica"), "helv")
        self.assertEqual(map_font_name("Arial"), "helv")

    def test_helvetica_bold(self):
        self.assertEqual(map_font_name("Helvetica-Bold"), "hebo")
        self.assertEqual(map_font_name("Arial-Bold"), "hebo")

    def test_helvetica_italic(self):
        self.assertEqual(map_font_name("Helvetica-Oblique"), "heit")

    def test_helvetica_bold_italic(self):
        self.assertEqual(map_font_name("Helvetica-BoldOblique"), "hebi")

    def test_times_family(self):
        self.assertEqual(map_font_name("Times-Roman"), "tiro")
        self.assertEqual(map_font_name("Times-Bold"), "tibo")
        self.assertEqual(map_font_name("Times-Italic"), "tiri")
        self.assertEqual(map_font_name("Times-BoldItalic"), "tibi")

    def test_times_new_roman_variant(self):
        self.assertEqual(map_font_name("TimesNewRoman"), "tiro")
        self.assertEqual(map_font_name("TimesNewRomanPS-Bold"), "tibo")

    def test_courier_family(self):
        self.assertEqual(map_font_name("Courier"), "cour")
        self.assertEqual(map_font_name("CourierNew-Bold"), "cobo")
        self.assertEqual(map_font_name("Courier-Oblique"), "coit")
        self.assertEqual(map_font_name("Courier-BoldOblique"), "cobi")

    def test_symbol_and_dingbats(self):
        self.assertEqual(map_font_name("Symbol"), "symb")
        self.assertEqual(map_font_name("ZapfDingbats"), "zadb")

    def test_subset_prefix_stripped(self):
        # PyMuPDF prefixes embedded subsets like "AAAAAA+Helvetica-Bold".
        self.assertEqual(map_font_name("AAAAAA+Helvetica-Bold"), "hebo")
        self.assertEqual(map_font_name("BCDEFG+TimesNewRomanPS"), "tiro")

    def test_unknown_falls_back_to_helv(self):
        self.assertEqual(map_font_name("MyCustomFont"), "helv")
        self.assertEqual(map_font_name(""), "helv")
        self.assertEqual(map_font_name(None), "helv")

    def test_all_outputs_are_valid_base14(self):
        # Whatever we output must be a font PyMuPDF accepts directly.
        for sample in [
            "Helvetica",
            "Times-Roman",
            "Courier-Bold",
            "Symbol",
            "AAAAAA+Helvetica-BoldOblique",
            "WeirdFont",
            "",
        ]:
            self.assertIn(map_font_name(sample), BASE14_FONTS)


class SafeBasenameAndTimestampTests(TestCase):
    def test_safe_basename_strips_dir_and_extension(self):
        self.assertEqual(safe_basename("/tmp/foo/bar.pdf"), "bar")
        self.assertEqual(safe_basename("just-a-name.pdf"), "just-a-name")
        self.assertEqual(safe_basename("noext"), "noext")

    def test_timestamp_format(self):
        ts = timestamp()
        self.assertEqual(len(ts), 15)  # YYYYMMDD_HHMMSS
        self.assertEqual(ts[8], "_")
        self.assertTrue(ts[:8].isdigit())
        self.assertTrue(ts[9:].isdigit())


class ProcessedDirTests(TestCase):
    def test_processed_dir_is_created(self):
        path = processed_dir()
        self.assertTrue(os.path.isdir(path))
        self.assertTrue(path.endswith("processed"))


class ParsePageRangeEdgeCases(TestCase):
    """Existing tests.py covers the happy paths; this targets edge cases."""

    def test_empty_range_returns_all(self):
        self.assertEqual(parse_page_range("", 3), [0, 1, 2])
        self.assertEqual(parse_page_range("   ", 3), [0, 1, 2])
        self.assertEqual(parse_page_range(None or "", 0), [])

    def test_whitespace_inside_range_is_stripped(self):
        self.assertEqual(parse_page_range("1 - 3 , 5", 5), [0, 1, 2, 4])

    def test_duplicates_are_collapsed_and_sorted(self):
        self.assertEqual(parse_page_range("3,1,2,3,1", 5), [0, 1, 2])

    def test_single_page_zero_raises(self):
        with self.assertRaises(ValueError):
            parse_page_range("0", 5)

    def test_single_page_above_total_raises(self):
        with self.assertRaises(ValueError):
            parse_page_range("11", 10)

    def test_reverse_range_raises(self):
        with self.assertRaises(ValueError):
            parse_page_range("5-3", 10)

    def test_non_numeric_raises(self):
        with self.assertRaises(ValueError):
            parse_page_range("abc", 10)

    def test_trailing_comma_is_ignored(self):
        self.assertEqual(parse_page_range("1,2,", 5), [0, 1])


class SanitizePdfTests(TestCase):
    def setUp(self):
        self.tmp_files = []

    def tearDown(self):
        for p in self.tmp_files:
            if os.path.exists(p):
                os.remove(p)

    def _make_malicious_pdf(self):
        """A one-page PDF carrying document JavaScript that auto-runs on open
        via /OpenAction, plus a Names/JavaScript entry."""
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "legitimate content", fontsize=12)
        catalog = doc.pdf_catalog()

        js_xref = doc.get_new_xref()
        doc.update_object(js_xref, "<< /S /JavaScript /JS (app.alert\\(1\\);) >>")
        doc.xref_set_key(catalog, "OpenAction", f"{js_xref} 0 R")

        names_js_xref = doc.get_new_xref()
        doc.update_object(names_js_xref, "<< /S /JavaScript /JS (app.alert\\(2\\);) >>")
        doc.xref_set_key(catalog, "Names", f"<< /JavaScript << /Names [ (a) {names_js_xref} 0 R ] >> >>")

        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        doc.save(path)
        doc.close()
        self.tmp_files.append(path)
        return path

    def test_strips_javascript_and_open_action(self):
        path = self._make_malicious_pdf()
        # Sanity: the unsanitised file really does carry the executable payload.
        with open(path, "rb") as f:
            self.assertIn(b"app.alert", f.read())

        self.assertTrue(sanitize_pdf(path))

        with open(path, "rb") as f:
            sanitized = f.read()
        # The executable JavaScript code is gone entirely.
        self.assertNotIn(b"app.alert", sanitized)

        with fitz.open(path) as doc:
            # The catalog's auto-run hook no longer points at anything.
            catalog = doc.pdf_catalog()
            self.assertEqual(doc.xref_get_key(catalog, "OpenAction"), ("null", "null"))
            self.assertEqual(doc.xref_get_key(catalog, "Names/JavaScript"), ("null", "null"))
            # And the document itself survives: still a valid 1-page PDF with text.
            self.assertEqual(len(doc), 1)
            self.assertIn("legitimate content", doc.load_page(0).get_text())

    def test_preserves_a_clean_pdf(self):
        doc = fitz.open()
        doc.new_page().insert_text((50, 50), "nothing dangerous here", fontsize=12)
        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        doc.save(path)
        doc.close()
        self.tmp_files.append(path)

        self.assertTrue(sanitize_pdf(path))
        with fitz.open(path) as doc:
            self.assertEqual(len(doc), 1)
            self.assertIn("nothing dangerous here", doc.load_page(0).get_text())

    def test_skips_encrypted_pdf(self):
        doc = fitz.open()
        doc.new_page().insert_text((50, 50), "secret", fontsize=12)
        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        doc.save(path, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
        doc.close()
        self.tmp_files.append(path)

        # Encrypted → can't rewrite without the password, so it's skipped.
        self.assertFalse(sanitize_pdf(path))

    def test_missing_file_returns_false(self):
        self.assertFalse(sanitize_pdf("/no/such/file.pdf"))


class CheckPdfHasTextTests(TestCase):
    def setUp(self):
        self.tmp_files = []

    def tearDown(self):
        for p in self.tmp_files:
            if os.path.exists(p):
                os.remove(p)

    def _make_pdf(self, text=None):
        doc = fitz.open()
        page = doc.new_page()
        if text:
            page.insert_text((50, 50), text, fontsize=12)
        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        doc.save(path)
        doc.close()
        self.tmp_files.append(path)
        return path

    def test_with_text(self):
        has_text, _msg = check_pdf_has_text(self._make_pdf("hello"))
        self.assertTrue(has_text)

    def test_without_text(self):
        has_text, _msg = check_pdf_has_text(self._make_pdf())
        self.assertFalse(has_text)

    def test_missing_file_returns_false(self):
        has_text, msg = check_pdf_has_text("/no/such/file.pdf")
        self.assertFalse(has_text)
        self.assertIn("not found", msg.lower())

    def test_corrupted_file_returns_false(self):
        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(b"%PDF-1.4\nthis is not a real pdf body")
        self.tmp_files.append(path)

        has_text, msg = check_pdf_has_text(path)
        self.assertFalse(has_text)
        self.assertTrue(msg)
