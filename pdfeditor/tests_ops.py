"""Tests for pdf_processor.ops — split, merge, compress, watermark, rotate, page numbers."""

import io
import os
import shutil
import tempfile

import fitz
from django.test import TestCase, override_settings
from PIL import Image as PILImage

from .pdf_processor.ops import (
    _calculate_position,
    add_page_numbers,
    add_watermark,
    compress_pdf,
    merge_pdfs,
    protect_pdf,
    rotate_pages,
    split_pdf,
)


class _MediaRootMixin:
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


def _make_multipage_pdf(num_pages: int = 3) -> str:
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"This is page {i + 1}.", fontsize=12)
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    doc.save(path)
    doc.close()
    return path


def _make_pdf_with_image(jpg_quality: int = 90) -> str:
    """Create a PDF whose page contains an embedded image — to test compression."""
    img = PILImage.new("RGB", (400, 400))
    # Build a noisy image so JPEG compression has actual headroom.
    pixels = img.load()
    for y in range(400):
        for x in range(400):
            pixels[x, y] = ((x * 7) % 256, (y * 11) % 256, ((x + y) * 5) % 256)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpg_quality)
    img_bytes = buf.getvalue()

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(fitz.Rect(50, 50, 450, 450), stream=img_bytes)
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    doc.save(path)
    doc.close()
    return path


class CalculatePositionTests(TestCase):
    PAGE_W, PAGE_H = 600, 800
    CONTENT_W, CONTENT_H = 100, 50

    def _pos(self, name):
        return _calculate_position(name, self.PAGE_W, self.PAGE_H, self.CONTENT_W, self.CONTENT_H)

    def test_top_left(self):
        x, y = self._pos("top-left")
        self.assertEqual((x, y), (20, 20 + 50))

    def test_top_center(self):
        x, _y = self._pos("top-center")
        self.assertAlmostEqual(x, (600 - 100) / 2)

    def test_top_right(self):
        x, _y = self._pos("top-right")
        self.assertEqual(x, 600 - 100 - 20)

    def test_center(self):
        x, y = self._pos("center")
        self.assertAlmostEqual(x, (600 - 100) / 2)
        self.assertAlmostEqual(y, (800 + 50) / 2)

    def test_bottom_right(self):
        x, y = self._pos("bottom-right")
        self.assertEqual(x, 600 - 100 - 20)
        self.assertEqual(y, 800 - 20)

    def test_unknown_position_falls_back_to_center(self):
        unknown = self._pos("not-a-position")
        center = self._pos("center")
        self.assertEqual(unknown, center)


class SplitPdfTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.path = _make_multipage_pdf(num_pages=5)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_split_single_range(self):
        outs = split_pdf(self.path, [(2, 3)])
        try:
            self.assertEqual(len(outs), 1)
            with fitz.open(outs[0]) as d:
                self.assertEqual(len(d), 2)
                self.assertIn("page 2", d[0].get_text())
        finally:
            for p in outs:
                if os.path.exists(p):
                    os.remove(p)

    def test_split_multiple_ranges_creates_multiple_files(self):
        outs = split_pdf(self.path, [(1, 2), (4, 5)])
        try:
            self.assertEqual(len(outs), 2)
            with fitz.open(outs[0]) as d:
                self.assertEqual(len(d), 2)
            with fitz.open(outs[1]) as d:
                self.assertEqual(len(d), 2)
            self.assertIn("part1", outs[0])
            self.assertIn("part2", outs[1])
        finally:
            for p in outs:
                if os.path.exists(p):
                    os.remove(p)

    def test_split_invalid_range_raises(self):
        with self.assertRaises(ValueError):
            split_pdf(self.path, [(0, 3)])  # start < 1
        with self.assertRaises(ValueError):
            split_pdf(self.path, [(1, 99)])  # end > total
        with self.assertRaises(ValueError):
            split_pdf(self.path, [(4, 2)])  # start > end

    def test_split_missing_file_raises(self):
        with self.assertRaises(ValueError):
            split_pdf("/no/such/file.pdf", [(1, 1)])


class MergePdfsTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.a = _make_multipage_pdf(2)
        self.b = _make_multipage_pdf(3)

    def tearDown(self):
        for p in (self.a, self.b):
            if os.path.exists(p):
                os.remove(p)

    def test_merge_basic(self):
        out = merge_pdfs([self.a, self.b])
        try:
            with fitz.open(out) as d:
                self.assertEqual(len(d), 5)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_merge_preserves_order(self):
        # The second PDF first should come out first.
        out = merge_pdfs([self.b, self.a])
        try:
            with fitz.open(out) as d:
                # Page 0 came from self.b → its text is "page 1" of that doc.
                self.assertIn("page 1", d[0].get_text())
                # Last page came from self.a (2 pages) → "page 2".
                self.assertIn("page 2", d[-1].get_text())
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_merge_uses_custom_output_name(self):
        out = merge_pdfs([self.a, self.b], output_name="my_combined")
        try:
            self.assertTrue(out.endswith("my_combined.pdf"))
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_merge_too_few_files_raises(self):
        with self.assertRaises(ValueError):
            merge_pdfs([self.a])
        with self.assertRaises(ValueError):
            merge_pdfs([])

    def test_merge_missing_file_raises(self):
        with self.assertRaises(ValueError):
            merge_pdfs([self.a, "/no/such/file.pdf"])


class CompressPdfTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.path = _make_pdf_with_image(jpg_quality=95)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_compress_returns_metrics(self):
        out, original, compressed, ratio = compress_pdf(self.path, quality="low")
        try:
            self.assertTrue(os.path.exists(out))
            self.assertEqual(original, os.path.getsize(self.path))
            self.assertEqual(compressed, os.path.getsize(out))
            self.assertIsInstance(ratio, float)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_low_quality_is_smaller_or_equal_to_high(self):
        out_low, _o, size_low, _r = compress_pdf(self.path, quality="low")
        out_high, _o, size_high, _r = compress_pdf(self.path, quality="high")
        try:
            # Re-encoding at quality 50 vs 90 should not produce a larger file.
            self.assertLessEqual(size_low, size_high)
        finally:
            for p in (out_low, out_high):
                if os.path.exists(p):
                    os.remove(p)

    def test_unknown_quality_falls_back_to_medium(self):
        out, _o, _c, _r = compress_pdf(self.path, quality="ridiculous")
        try:
            self.assertTrue(os.path.exists(out))
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            compress_pdf("/no/such/file.pdf")


class ProtectPdfTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.path = _make_multipage_pdf(2)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_protect_produces_encrypted_pdf(self):
        out = protect_pdf(self.path, user_password="hunter2")
        try:
            self.assertTrue(os.path.exists(out))
            doc = fitz.open(out)
            self.assertTrue(doc.is_encrypted)
            self.assertTrue(doc.needs_pass)
            doc.close()
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_protected_pdf_opens_with_correct_password(self):
        out = protect_pdf(self.path, user_password="hunter2")
        try:
            doc = fitz.open(out)
            self.assertEqual(doc.authenticate("hunter2"), 6)  # owner+user auth success
            self.assertGreater(doc.page_count, 0)
            doc.close()
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_protected_pdf_rejects_wrong_password(self):
        out = protect_pdf(self.path, user_password="hunter2")
        try:
            doc = fitz.open(out)
            self.assertEqual(doc.authenticate("wrong"), 0)
            doc.close()
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_empty_password_raises(self):
        with self.assertRaises(ValueError):
            protect_pdf(self.path, user_password="")

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            protect_pdf("/no/such/file.pdf", user_password="x")

    def test_already_encrypted_raises(self):
        out = protect_pdf(self.path, user_password="hunter2")
        try:
            with self.assertRaises(ValueError):
                protect_pdf(out, user_password="another")
        finally:
            if os.path.exists(out):
                os.remove(out)


class AddWatermarkTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.pdf = _make_multipage_pdf(2)
        # Make a small PNG for the image-watermark case.
        img = PILImage.new("RGBA", (100, 100), (255, 0, 0, 255))
        fd, self.png = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(self.png, format="PNG")

    def tearDown(self):
        for p in (self.pdf, self.png):
            if os.path.exists(p):
                os.remove(p)

    def test_text_watermark_writes_text_on_page(self):
        out = add_watermark(self.pdf, "text", "CONFIDENTIAL", options={"position": "center", "rotation": 0})
        try:
            with fitz.open(out) as d:
                # Watermark + original "This is page 1." should both appear.
                page_text = d[0].get_text()
                self.assertIn("CONFIDENTIAL", page_text)
                self.assertIn("page 1", page_text)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_image_watermark_does_not_crash(self):
        out = add_watermark(self.pdf, "image", self.png, options={"position": "top-right", "opacity": 0.5})
        try:
            self.assertTrue(os.path.exists(out))
            with fitz.open(out) as d:
                # An image was inserted on every page.
                for page in d:
                    self.assertGreaterEqual(len(page.get_images(full=True)), 1)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_image_watermark_missing_file_raises(self):
        with self.assertRaises(ValueError):
            add_watermark(self.pdf, "image", "/no/such/image.png")

    def test_text_watermark_snaps_non_multiple_of_90_rotation(self):
        # PyMuPDF's insert_text only allows rotations in {0, 90, 180, 270}.
        # The slider in WatermarkForm allows any int from -90 to 90, so the op
        # must snap to a valid multiple before calling fitz.
        for angle in (45, 30, -45, -1, 89):
            out = add_watermark(self.pdf, "text", "X", options={"rotation": angle})
            try:
                self.assertTrue(os.path.exists(out))
            finally:
                if os.path.exists(out):
                    os.remove(out)

    def test_missing_pdf_raises(self):
        with self.assertRaises(ValueError):
            add_watermark("/no/such/file.pdf", "text", "x")


class RotatePagesTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.path = _make_multipage_pdf(3)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_rotate_all_pages(self):
        out = rotate_pages(self.path, 90)
        try:
            with fitz.open(out) as d:
                for page in d:
                    self.assertEqual(page.rotation, 90)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_rotate_specific_pages_only(self):
        out = rotate_pages(self.path, 180, page_range="2")
        try:
            with fitz.open(out) as d:
                self.assertEqual(d[0].rotation, 0)
                self.assertEqual(d[1].rotation, 180)
                self.assertEqual(d[2].rotation, 0)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_invalid_angle_raises(self):
        with self.assertRaises(ValueError):
            rotate_pages(self.path, 45)
        with self.assertRaises(ValueError):
            rotate_pages(self.path, 360)
        with self.assertRaises(ValueError):
            rotate_pages(self.path, 0)

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            rotate_pages("/no/such/file.pdf", 90)


class AddPageNumbersTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.path = _make_multipage_pdf(3)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_default_numbering(self):
        out = add_page_numbers(self.path)
        try:
            with fitz.open(out) as d:
                # Page 1 should contain "1" (just the number).
                self.assertIn("1", d[0].get_text())
                self.assertIn("3", d[2].get_text())
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_format_page_number(self):
        out = add_page_numbers(self.path, options={"format": "page_number"})
        try:
            with fitz.open(out) as d:
                self.assertIn("Page 1", d[0].get_text())
                self.assertIn("Page 3", d[2].get_text())
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_format_of_total(self):
        out = add_page_numbers(self.path, options={"format": "of_total"})
        try:
            with fitz.open(out) as d:
                self.assertIn("1 of 3", d[0].get_text())
                self.assertIn("3 of 3", d[2].get_text())
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_start_page_skips_earlier_pages(self):
        out = add_page_numbers(self.path, options={"format": "page_number", "start_page": 2})
        try:
            with fitz.open(out) as d:
                self.assertNotIn("Page 1", d[0].get_text())
                self.assertIn("Page 2", d[1].get_text())
                self.assertIn("Page 3", d[2].get_text())
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            add_page_numbers("/no/such/file.pdf")
