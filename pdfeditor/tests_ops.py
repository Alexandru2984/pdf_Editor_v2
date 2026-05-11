"""Tests for pdf_processor.ops — split, merge, compress, watermark, rotate, page numbers."""

import io
import os
import shutil
import tempfile
import unittest
import zipfile

import fitz
from django.test import TestCase, override_settings
from PIL import Image as PILImage

from .pdf_processor.extract import make_pdf_searchable
from .pdf_processor.ops import (
    _calculate_position,
    _parse_pdf_date,
    add_page_numbers,
    add_watermark,
    compare_pdfs,
    compress_pdf,
    convert_images_to_pdf,
    convert_pdf_to_docx,
    convert_pdf_to_images,
    convert_to_pdfa,
    crop_pages,
    edit_pdf_metadata,
    flatten_pdf,
    merge_pdfs,
    protect_pdf,
    read_pdf_metadata,
    redact_text,
    remove_pdf_password,
    render_page_thumbnail,
    reorder_pages,
    rotate_pages,
    sign_pdf,
    split_pdf,
    verify_pdf_signatures,
)


def _make_self_signed_p12(passphrase: bytes = b"test123") -> tuple[bytes, bytes]:
    """Generate self-signed PKCS#12 archive + cert PEM. Returns (p12_bytes, cert_pem)."""
    from datetime import datetime, timedelta, timezone

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.serialization import pkcs12
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test Signer")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=30))
        .sign(key, hashes.SHA256())
    )
    pfx = pkcs12.serialize_key_and_certificates(
        b"test", key, cert, [], serialization.BestAvailableEncryption(passphrase)
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    return pfx, cert_pem


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


class ConvertPdfToDocxTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.path = _make_multipage_pdf(2)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_convert_returns_docx_with_text(self):
        out, has_text = convert_pdf_to_docx(self.path)
        try:
            self.assertTrue(out.endswith(".docx"))
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 0)
            self.assertTrue(has_text)
            from docx import Document

            doc = Document(out)
            full = "\n".join(p.text for p in doc.paragraphs)
            self.assertIn("This is page 1", full)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_image_only_pdf_reports_no_text(self):
        # PDF with image but no text → has_text False
        img_path = _make_pdf_with_image()
        try:
            out, has_text = convert_pdf_to_docx(img_path)
            try:
                self.assertTrue(os.path.exists(out))
                self.assertFalse(has_text)
            finally:
                if os.path.exists(out):
                    os.remove(out)
        finally:
            if os.path.exists(img_path):
                os.remove(img_path)

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            convert_pdf_to_docx("/no/such/file.pdf")

    def test_encrypted_pdf_raises(self):
        out = protect_pdf(self.path, user_password="hunter2")
        try:
            with self.assertRaises(ValueError):
                convert_pdf_to_docx(out)
        finally:
            if os.path.exists(out):
                os.remove(out)


class SignPdfTests(_MediaRootMixin, TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Cert generation is slow (RSA key) — share one across tests.
        cls.p12, cls.cert_pem = _make_self_signed_p12(b"test123")

    def setUp(self):
        self.pdf = _make_multipage_pdf(2)

    def tearDown(self):
        if os.path.exists(self.pdf):
            os.remove(self.pdf)

    def _cleanup(self, *paths):
        for p in paths:
            if p and os.path.exists(p):
                os.remove(p)

    def test_sign_produces_signed_pdf(self):
        out = sign_pdf(self.pdf, p12_bytes=self.p12, p12_password="test123")
        try:
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), os.path.getsize(self.pdf))
        finally:
            self._cleanup(out)

    def test_signed_pdf_has_signature_object(self):
        from pyhanko.pdf_utils.reader import PdfFileReader
        from pyhanko.sign.validation import async_validate_pdf_signature  # noqa: F401

        out = sign_pdf(self.pdf, p12_bytes=self.p12, p12_password="test123")
        try:
            with open(out, "rb") as f:
                reader = PdfFileReader(f)
                self.assertEqual(len(reader.embedded_signatures), 1)
                emb = reader.embedded_signatures[0]
                self.assertEqual(emb.field_name, "Signature1")
        finally:
            self._cleanup(out)

    def test_wrong_password_raises(self):
        with self.assertRaises(ValueError):
            sign_pdf(self.pdf, p12_bytes=self.p12, p12_password="wrong-pass")

    def test_empty_p12_raises(self):
        with self.assertRaises(ValueError):
            sign_pdf(self.pdf, p12_bytes=b"", p12_password="x")

    def test_missing_pdf_raises(self):
        with self.assertRaises(ValueError):
            sign_pdf("/no/such/file.pdf", p12_bytes=self.p12, p12_password="test123")

    def test_invalid_page_raises(self):
        with self.assertRaises(ValueError):
            sign_pdf(self.pdf, p12_bytes=self.p12, p12_password="test123", page=99)

    def test_signing_encrypted_pdf_raises(self):
        # Protect first, then attempt to sign — should refuse.
        protected = protect_pdf(self.pdf, user_password="secret")
        try:
            with self.assertRaises(ValueError):
                sign_pdf(protected, p12_bytes=self.p12, p12_password="test123")
        finally:
            self._cleanup(protected)

    def test_sign_at_different_positions(self):
        out = sign_pdf(self.pdf, p12_bytes=self.p12, p12_password="test123", position="top-left")
        try:
            self.assertTrue(os.path.exists(out))
        finally:
            self._cleanup(out)

    def test_unreachable_tsa_raises(self):
        # Port 1 on loopback has nothing listening — pyHanko's HTTP timestamp
        # call fails fast. Either our wrapped ValueError or an OSError-family
        # connection error is acceptable; signing must not silently succeed.
        with self.assertRaises((ValueError, OSError)):
            sign_pdf(
                self.pdf,
                p12_bytes=self.p12,
                p12_password="test123",
                tsa_url="http://127.0.0.1:1/",
            )

    def test_ltv_with_self_signed_cert_raises(self):
        # PAdES B-LT requires a verifiable chain — self-signed must error out.
        with self.assertRaises(ValueError):
            sign_pdf(
                self.pdf,
                p12_bytes=self.p12,
                p12_password="test123",
                embed_validation_info=True,
            )

    def test_doc_timestamp_requires_tsa_url(self):
        # Asking for an archival timestamp without a TSA must error explicitly.
        with self.assertRaises(ValueError):
            sign_pdf(
                self.pdf,
                p12_bytes=self.p12,
                p12_password="test123",
                add_doc_timestamp=True,
            )

    def test_doc_timestamp_unreachable_tsa_raises(self):
        with self.assertRaises((ValueError, OSError)):
            sign_pdf(
                self.pdf,
                p12_bytes=self.p12,
                p12_password="test123",
                tsa_url="http://127.0.0.1:1/",
                add_doc_timestamp=True,
            )


class VerifyPdfSignaturesTests(_MediaRootMixin, TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.p12, cls.cert_pem = _make_self_signed_p12(b"test123")

    def setUp(self):
        self.pdf = _make_multipage_pdf(2)
        self.signed = sign_pdf(self.pdf, p12_bytes=self.p12, p12_password="test123")

    def tearDown(self):
        for p in (self.pdf, self.signed):
            if p and os.path.exists(p):
                os.remove(p)

    def test_unsigned_pdf_returns_empty_report(self):
        reports = verify_pdf_signatures(self.pdf)
        self.assertEqual(reports, [])

    def test_signed_pdf_intact_but_untrusted_without_anchor(self):
        reports = verify_pdf_signatures(self.signed)
        self.assertEqual(len(reports), 1)
        r = reports[0]
        self.assertTrue(r["intact"])
        self.assertFalse(r["trusted"])  # self-signed → no chain
        self.assertEqual(r["field_name"], "Signature1")
        self.assertIn("Test Signer", r["signer_name"])
        self.assertIsNotNone(r["signing_time"])

    def test_signed_pdf_trusted_when_anchor_provided(self):
        reports = verify_pdf_signatures(self.signed, extra_trust_certs=[self.cert_pem])
        self.assertEqual(len(reports), 1)
        r = reports[0]
        self.assertTrue(r["intact"])
        self.assertTrue(r["trusted"])

    def test_tampered_pdf_loses_integrity(self):
        # Flip a byte in the middle of the signed file. The original byte range
        # covered by the signature includes most of the body (excluding only the
        # signature blob itself), so any mid-file mutation breaks `intact`.
        with open(self.signed, "rb") as f:
            data = bytearray(f.read())
        mid = len(data) // 2
        # Pick a byte that we can safely mutate (anything other than '\n' near
        # the middle works; binary content here makes any flip a real corruption).
        original = data[mid]
        data[mid] = (original + 1) % 256
        with open(self.signed, "wb") as f:
            f.write(bytes(data))
        reports = verify_pdf_signatures(self.signed, extra_trust_certs=[self.cert_pem])
        self.assertEqual(len(reports), 1)
        # Tampering breaks integrity OR coverage OR raises errors during
        # validation; no path should yield "intact AND trusted AND no errors".
        r = reports[0]
        self.assertFalse(r["intact"] and r["trusted"] and not r["errors"])

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            verify_pdf_signatures("/no/such/file.pdf")


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


class ReorderPagesTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.path = _make_multipage_pdf(4)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def _read_pages_text(self, path):
        with fitz.open(path) as d:
            return [p.get_text("text").strip() for p in d]

    def test_reverse_order(self):
        out = reorder_pages(self.path, [4, 3, 2, 1])
        try:
            texts = self._read_pages_text(out)
            self.assertIn("page 4", texts[0])
            self.assertIn("page 3", texts[1])
            self.assertIn("page 2", texts[2])
            self.assertIn("page 1", texts[3])
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_delete_some_pages(self):
        out = reorder_pages(self.path, [1, 3])
        try:
            with fitz.open(out) as d:
                self.assertEqual(len(d), 2)
            texts = self._read_pages_text(out)
            self.assertIn("page 1", texts[0])
            self.assertIn("page 3", texts[1])
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_keep_only_one_page(self):
        out = reorder_pages(self.path, [2])
        try:
            with fitz.open(out) as d:
                self.assertEqual(len(d), 1)
            texts = self._read_pages_text(out)
            self.assertIn("page 2", texts[0])
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_empty_order_raises(self):
        with self.assertRaises(ValueError):
            reorder_pages(self.path, [])

    def test_duplicate_page_raises(self):
        with self.assertRaises(ValueError):
            reorder_pages(self.path, [1, 2, 2])

    def test_out_of_range_page_raises(self):
        with self.assertRaises(ValueError):
            reorder_pages(self.path, [1, 5])
        with self.assertRaises(ValueError):
            reorder_pages(self.path, [0, 1])

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            reorder_pages("/no/such/file.pdf", [1])


class RenderPageThumbnailTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.path = _make_multipage_pdf(3)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_returns_png_bytes(self):
        png = render_page_thumbnail(self.path, 1, max_width=120)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(len(png), 100)

    def test_invalid_page_raises(self):
        with self.assertRaises(ValueError):
            render_page_thumbnail(self.path, 0)
        with self.assertRaises(ValueError):
            render_page_thumbnail(self.path, 99)

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            render_page_thumbnail("/no/such/file.pdf", 1)


class ConvertPdfToImagesTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.path = _make_multipage_pdf(3)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_png_export_creates_zip_with_one_image_per_page(self):
        zip_path, count = convert_pdf_to_images(self.path, fmt="png", dpi=72)
        try:
            self.assertEqual(count, 3)
            self.assertTrue(zip_path.endswith(".zip"))
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                self.assertEqual(len(names), 3)
                self.assertTrue(all(n.endswith(".png") for n in names))
                # Page numbering is 1-indexed and zero-padded.
                self.assertTrue(any("_page_001.png" in n for n in names))
                # PNG magic.
                self.assertTrue(zf.read(names[0]).startswith(b"\x89PNG\r\n\x1a\n"))
        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path)

    def test_jpg_export_writes_jpeg_files(self):
        zip_path, count = convert_pdf_to_images(self.path, fmt="jpg", dpi=72)
        try:
            self.assertEqual(count, 3)
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                self.assertTrue(all(n.endswith(".jpg") for n in names))
                # JPEG starts with FF D8 FF.
                self.assertTrue(zf.read(names[0]).startswith(b"\xff\xd8\xff"))
        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path)

    def test_jpeg_alias_normalizes_to_jpg(self):
        zip_path, _ = convert_pdf_to_images(self.path, fmt="jpeg", dpi=72)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                self.assertTrue(all(n.endswith(".jpg") for n in zf.namelist()))
        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path)

    def test_unsupported_format_raises(self):
        with self.assertRaises(ValueError):
            convert_pdf_to_images(self.path, fmt="bmp")

    def test_dpi_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            convert_pdf_to_images(self.path, dpi=10)
        with self.assertRaises(ValueError):
            convert_pdf_to_images(self.path, dpi=900)

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            convert_pdf_to_images("/no/such/file.pdf")

    def test_higher_dpi_produces_larger_image_dimensions(self):
        low_zip, _ = convert_pdf_to_images(self.path, fmt="png", dpi=72)
        high_zip, _ = convert_pdf_to_images(self.path, fmt="png", dpi=300)
        try:
            with zipfile.ZipFile(low_zip) as zf:
                low_img = PILImage.open(io.BytesIO(zf.read(zf.namelist()[0])))
                low_w, low_h = low_img.size
            with zipfile.ZipFile(high_zip) as zf:
                high_img = PILImage.open(io.BytesIO(zf.read(zf.namelist()[0])))
                high_w, high_h = high_img.size
            self.assertGreater(high_w, low_w)
            self.assertGreater(high_h, low_h)
        finally:
            for p in (low_zip, high_zip):
                if os.path.exists(p):
                    os.remove(p)


def _make_image_file(
    suffix: str, size: tuple[int, int] = (200, 150), color=(50, 100, 200), mode="RGB"
) -> str:
    """Write a simple image to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    im = PILImage.new("RGBA", size, color + (180,)) if mode == "RGBA" else PILImage.new(mode, size, color)
    fmt = "PNG"
    if suffix.lower() in (".jpg", ".jpeg"):
        fmt = "JPEG"
        if im.mode != "RGB":
            im = im.convert("RGB")
    elif suffix.lower() == ".bmp":
        fmt = "BMP"
        if im.mode == "RGBA":
            im = im.convert("RGB")
    im.save(path, format=fmt)
    return path


class ConvertImagesToPdfTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.tmp_files: list[str] = []

    def tearDown(self):
        for p in self.tmp_files:
            if os.path.exists(p):
                os.remove(p)

    def _make(self, suffix=".png", size=(200, 150), **kw) -> str:
        path = _make_image_file(suffix, size=size, **kw)
        self.tmp_files.append(path)
        return path

    def test_single_png_to_pdf_auto_size(self):
        img = self._make(".png", size=(300, 200))
        out, count = convert_images_to_pdf([img], page_size="auto", fit_mode="fit")
        try:
            self.assertEqual(count, 1)
            self.assertTrue(out.endswith(".pdf"))
            with fitz.open(out) as doc:
                self.assertEqual(len(doc), 1)
                # Auto: page dimensions match image pixels (1px = 1pt).
                rect = doc[0].rect
                self.assertAlmostEqual(rect.width, 300, places=0)
                self.assertAlmostEqual(rect.height, 200, places=0)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_multiple_images_one_page_each(self):
        imgs = [self._make(".png"), self._make(".jpg"), self._make(".bmp")]
        out, count = convert_images_to_pdf(imgs)
        try:
            self.assertEqual(count, 3)
            with fitz.open(out) as doc:
                self.assertEqual(len(doc), 3)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_a4_page_size_orients_landscape_for_landscape_image(self):
        img = self._make(".png", size=(800, 400))  # landscape
        out, _ = convert_images_to_pdf([img], page_size="a4", fit_mode="fit")
        try:
            with fitz.open(out) as doc:
                rect = doc[0].rect
                # Landscape A4: 842 x 595.
                self.assertAlmostEqual(rect.width, 842, places=0)
                self.assertAlmostEqual(rect.height, 595, places=0)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_a4_page_size_orients_portrait_for_portrait_image(self):
        img = self._make(".png", size=(400, 800))  # portrait
        out, _ = convert_images_to_pdf([img], page_size="a4")
        try:
            with fitz.open(out) as doc:
                rect = doc[0].rect
                self.assertAlmostEqual(rect.width, 595, places=0)
                self.assertAlmostEqual(rect.height, 842, places=0)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_letter_page_size(self):
        img = self._make(".png", size=(400, 800))
        out, _ = convert_images_to_pdf([img], page_size="letter")
        try:
            with fitz.open(out) as doc:
                rect = doc[0].rect
                self.assertAlmostEqual(rect.width, 612, places=0)
                self.assertAlmostEqual(rect.height, 792, places=0)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_rgba_with_transparency_flattens_onto_white(self):
        img = self._make(".png", size=(200, 200), mode="RGBA")
        out, _ = convert_images_to_pdf([img])
        try:
            self.assertTrue(os.path.exists(out))
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_empty_list_raises(self):
        with self.assertRaises(ValueError):
            convert_images_to_pdf([])

    def test_missing_image_raises(self):
        with self.assertRaises(ValueError):
            convert_images_to_pdf(["/no/such/image.png"])

    def test_unsupported_page_size_raises(self):
        img = self._make(".png")
        with self.assertRaises(ValueError):
            convert_images_to_pdf([img], page_size="legal")

    def test_unsupported_fit_mode_raises(self):
        img = self._make(".png")
        with self.assertRaises(ValueError):
            convert_images_to_pdf([img], fit_mode="crop")

    def test_corrupt_image_raises_value_error(self):
        fd, bad = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        with open(bad, "wb") as f:
            f.write(b"not a real PNG")
        self.tmp_files.append(bad)
        with self.assertRaises(ValueError):
            convert_images_to_pdf([bad])


class ParsePdfDateTests(TestCase):
    def test_full_date_with_offset(self):
        self.assertEqual(_parse_pdf_date("D:20250304151230+02'00'"), "2025-03-04 15:12")

    def test_date_without_d_prefix(self):
        self.assertEqual(_parse_pdf_date("20240115093000Z"), "2024-01-15 09:30")

    def test_date_only(self):
        self.assertEqual(_parse_pdf_date("D:20231201"), "2023-12-01 00:00")

    def test_empty_returns_none(self):
        self.assertIsNone(_parse_pdf_date(""))
        self.assertIsNone(_parse_pdf_date(None))

    def test_garbage_returns_none(self):
        self.assertIsNone(_parse_pdf_date("not-a-date"))


def _make_pdf_with_metadata(meta: dict) -> str:
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    full = {
        "title": "",
        "author": "",
        "subject": "",
        "keywords": "",
        "creator": "",
        "producer": "",
        "creationDate": "",
        "modDate": "",
        **meta,
    }
    doc.set_metadata(full)
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    doc.save(path)
    doc.close()
    return path


class ReadPdfMetadataTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.tmp_files: list[str] = []

    def tearDown(self):
        for p in self.tmp_files:
            if os.path.exists(p):
                os.remove(p)

    def test_reads_all_fields(self):
        path = _make_pdf_with_metadata(
            {
                "title": "Doc Title",
                "author": "Jane",
                "subject": "Sales",
                "keywords": "k1,k2",
                "creator": "Word",
                "producer": "PyMuPDF",
            }
        )
        self.tmp_files.append(path)
        meta = read_pdf_metadata(path)
        self.assertEqual(meta["title"], "Doc Title")
        self.assertEqual(meta["author"], "Jane")
        self.assertEqual(meta["subject"], "Sales")
        self.assertEqual(meta["keywords"], "k1,k2")
        self.assertEqual(meta["creator"], "Word")
        self.assertIn("PyMuPDF", meta["producer"])

    def test_missing_fields_return_empty_strings(self):
        path = _make_pdf_with_metadata({})
        self.tmp_files.append(path)
        meta = read_pdf_metadata(path)
        for key in ("title", "author", "subject", "keywords", "creator"):
            self.assertEqual(meta[key], "")

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            read_pdf_metadata("/no/such/file.pdf")


class EditPdfMetadataTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.tmp_files: list[str] = []

    def tearDown(self):
        for p in self.tmp_files:
            if os.path.exists(p):
                os.remove(p)

    def _src(self, **meta) -> str:
        path = _make_pdf_with_metadata(meta)
        self.tmp_files.append(path)
        return path

    def test_writes_new_fields(self):
        src = self._src(title="Old", author="Old Author")
        out = edit_pdf_metadata(
            src,
            metadata={
                "title": "New Title",
                "author": "New Author",
                "subject": "S",
                "keywords": "k",
                "creator": "",
                "producer": "",
            },
        )
        self.tmp_files.append(out)
        meta = read_pdf_metadata(out)
        self.assertEqual(meta["title"], "New Title")
        self.assertEqual(meta["author"], "New Author")
        self.assertEqual(meta["subject"], "S")
        self.assertEqual(meta["keywords"], "k")

    def test_empty_string_clears_field(self):
        src = self._src(title="Old", author="Old")
        out = edit_pdf_metadata(src, metadata={"title": "", "author": "Keep"})
        self.tmp_files.append(out)
        meta = read_pdf_metadata(out)
        self.assertEqual(meta["title"], "")
        self.assertEqual(meta["author"], "Keep")

    def test_unspecified_fields_preserved(self):
        src = self._src(title="Keep Title", author="Keep Author")
        out = edit_pdf_metadata(src, metadata={"subject": "Only Subject"})
        self.tmp_files.append(out)
        meta = read_pdf_metadata(out)
        self.assertEqual(meta["title"], "Keep Title")
        self.assertEqual(meta["author"], "Keep Author")
        self.assertEqual(meta["subject"], "Only Subject")

    def test_clear_dates(self):
        src = self._src(title="x", creationDate="D:20240115093000Z", modDate="D:20240601120000Z")
        out = edit_pdf_metadata(src, metadata={}, clear_dates=True)
        self.tmp_files.append(out)
        meta = read_pdf_metadata(out)
        self.assertIsNone(meta["creation_date"])
        self.assertIsNone(meta["mod_date"])

    def test_output_in_processed_dir(self):
        src = self._src(title="x")
        out = edit_pdf_metadata(src, metadata={"title": "y"})
        self.tmp_files.append(out)
        self.assertTrue(out.endswith(".pdf"))
        self.assertIn("metadata", os.path.basename(out))

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            edit_pdf_metadata("/no/such/file.pdf", metadata={"title": "x"})

    def test_encrypted_pdf_raises(self):
        src = self._src(title="x")
        encrypted = protect_pdf(src, owner_password="o", user_password="u")
        self.tmp_files.append(encrypted)
        with self.assertRaises(ValueError):
            edit_pdf_metadata(encrypted, metadata={"title": "y"})


class RemovePdfPasswordTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.tmp_files: list[str] = []
        self.plain = _make_multipage_pdf(num_pages=2)
        self.tmp_files.append(self.plain)

    def tearDown(self):
        for p in self.tmp_files:
            if os.path.exists(p):
                os.remove(p)

    def test_removes_password_with_correct_input(self):
        encrypted = protect_pdf(self.plain, user_password="secret123")
        self.tmp_files.append(encrypted)
        out = remove_pdf_password(encrypted, password="secret123")
        self.tmp_files.append(out)
        with fitz.open(out) as doc:
            self.assertFalse(doc.is_encrypted)
            self.assertEqual(len(doc), 2)

    def test_owner_password_also_works(self):
        encrypted = protect_pdf(self.plain, user_password="user-pw", owner_password="owner-pw")
        self.tmp_files.append(encrypted)
        out = remove_pdf_password(encrypted, password="owner-pw")
        self.tmp_files.append(out)
        with fitz.open(out) as doc:
            self.assertFalse(doc.is_encrypted)

    def test_wrong_password_raises(self):
        encrypted = protect_pdf(self.plain, user_password="correct")
        self.tmp_files.append(encrypted)
        with self.assertRaises(ValueError):
            remove_pdf_password(encrypted, password="wrong")

    def test_unencrypted_pdf_raises(self):
        with self.assertRaises(ValueError):
            remove_pdf_password(self.plain, password="anything")

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            remove_pdf_password("/no/such/file.pdf", password="x")

    def test_none_password_raises(self):
        encrypted = protect_pdf(self.plain, user_password="x")
        self.tmp_files.append(encrypted)
        with self.assertRaises(ValueError):
            remove_pdf_password(encrypted, password=None)  # type: ignore[arg-type]

    def test_output_basename_marks_unprotected(self):
        encrypted = protect_pdf(self.plain, user_password="pw")
        self.tmp_files.append(encrypted)
        out = remove_pdf_password(encrypted, password="pw")
        self.tmp_files.append(out)
        self.assertIn("unprotected", os.path.basename(out))


class CropPagesTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.tmp_files: list[str] = []
        self.path = _make_multipage_pdf(num_pages=3)
        self.tmp_files.append(self.path)

    def tearDown(self):
        for p in self.tmp_files:
            if os.path.exists(p):
                os.remove(p)

    def test_simple_crop_all_pages(self):
        out = crop_pages(self.path, top=10, right=10, bottom=10, left=10)
        self.tmp_files.append(out)
        with fitz.open(self.path) as src, fitz.open(out) as dst:
            self.assertEqual(len(src), len(dst))
            src_rect = src[0].rect
            dst_box = dst[0].cropbox
            # Width shrinks by 20% (10% each side), height by 20% as well.
            self.assertAlmostEqual(dst_box.width, src_rect.width * 0.8, delta=0.5)
            self.assertAlmostEqual(dst_box.height, src_rect.height * 0.8, delta=0.5)

    def test_only_top_margin(self):
        out = crop_pages(self.path, top=20)
        self.tmp_files.append(out)
        with fitz.open(self.path) as src, fitz.open(out) as dst:
            src_rect = src[0].rect
            dst_box = dst[0].cropbox
            self.assertAlmostEqual(dst_box.width, src_rect.width, delta=0.5)
            self.assertAlmostEqual(dst_box.height, src_rect.height * 0.8, delta=0.5)

    def test_page_range_only_crops_specified_pages(self):
        out = crop_pages(self.path, top=15, page_range="1,3")
        self.tmp_files.append(out)
        with fitz.open(self.path) as src, fitz.open(out) as dst:
            src_h = src[0].rect.height
            self.assertAlmostEqual(dst[0].cropbox.height, src_h * 0.85, delta=0.5)
            # Page 2 untouched.
            self.assertAlmostEqual(dst[1].cropbox.height, src_h, delta=0.5)
            self.assertAlmostEqual(dst[2].cropbox.height, src_h * 0.85, delta=0.5)

    def test_negative_margin_raises(self):
        with self.assertRaises(ValueError):
            crop_pages(self.path, top=-5)

    def test_margin_at_or_above_50_raises(self):
        with self.assertRaises(ValueError):
            crop_pages(self.path, top=50)

    def test_zero_margins_raises(self):
        with self.assertRaises(ValueError):
            crop_pages(self.path)

    def test_invalid_page_range_raises(self):
        with self.assertRaises(ValueError):
            crop_pages(self.path, top=10, page_range="99")

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            crop_pages("/no/such/file.pdf", top=10)

    def test_encrypted_pdf_raises(self):
        encrypted = protect_pdf(self.path, user_password="pw")
        self.tmp_files.append(encrypted)
        with self.assertRaises(ValueError):
            crop_pages(encrypted, top=10)

    def test_output_basename_marks_cropped(self):
        out = crop_pages(self.path, top=10)
        self.tmp_files.append(out)
        self.assertIn("cropped", os.path.basename(out))


def _make_pdf_with_form_and_annotation() -> str:
    """Create a 1-page PDF with one text widget and one text annotation."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), "Document body.", fontsize=12)
    widget = fitz.Widget()
    widget.field_name = "name"
    widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
    widget.field_value = "Alice"
    widget.rect = fitz.Rect(72, 120, 280, 145)
    page.add_widget(widget)
    page.add_text_annot((72, 200), "Reviewer note")
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    doc.save(path)
    doc.close()
    return path


class FlattenPdfTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.tmp_files: list[str] = []
        self.path = _make_pdf_with_form_and_annotation()
        self.tmp_files.append(self.path)

    def tearDown(self):
        for p in self.tmp_files:
            if os.path.exists(p):
                os.remove(p)

    def _count(self, pdf_path):
        with fitz.open(pdf_path) as doc:
            page = doc[0]
            widgets = list(page.widgets() or [])
            annots = list(page.annots() or [])
            return len(widgets), len(annots)

    def test_source_has_form_and_annotation(self):
        widgets, annots = self._count(self.path)
        self.assertEqual(widgets, 1)
        self.assertGreaterEqual(annots, 1)

    def test_flatten_both_removes_widgets_and_annotations(self):
        out = flatten_pdf(self.path)
        self.tmp_files.append(out)
        widgets, annots = self._count(out)
        self.assertEqual(widgets, 0)
        self.assertEqual(annots, 0)

    def test_flatten_only_forms_keeps_annotations(self):
        out = flatten_pdf(self.path, flatten_annotations=False, flatten_forms=True)
        self.tmp_files.append(out)
        widgets, annots = self._count(out)
        self.assertEqual(widgets, 0)
        self.assertGreaterEqual(annots, 1)

    def test_flatten_only_annotations_keeps_widgets(self):
        out = flatten_pdf(self.path, flatten_annotations=True, flatten_forms=False)
        self.tmp_files.append(out)
        widgets, annots = self._count(out)
        self.assertEqual(widgets, 1)
        self.assertEqual(annots, 0)

    def test_neither_option_raises(self):
        with self.assertRaises(ValueError):
            flatten_pdf(self.path, flatten_annotations=False, flatten_forms=False)

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            flatten_pdf("/no/such/file.pdf")

    def test_encrypted_pdf_raises(self):
        encrypted = protect_pdf(self.path, user_password="pw")
        self.tmp_files.append(encrypted)
        with self.assertRaises(ValueError):
            flatten_pdf(encrypted)

    def test_output_basename_marks_flattened(self):
        out = flatten_pdf(self.path)
        self.tmp_files.append(out)
        self.assertIn("flattened", os.path.basename(out))

    def test_text_remains_searchable(self):
        out = flatten_pdf(self.path)
        self.tmp_files.append(out)
        with fitz.open(out) as doc:
            text = doc[0].get_text()
        self.assertIn("Document body", text)


def _make_pdf_with_text(pages_text: list[str]) -> str:
    """Build a multi-page PDF where each page contains the given string."""
    doc = fitz.open()
    for txt in pages_text:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), txt, fontsize=12)
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    doc.save(path)
    doc.close()
    return path


class RedactTextTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.tmp_files: list[str] = []

    def tearDown(self):
        for p in self.tmp_files:
            if os.path.exists(p):
                os.remove(p)

    def _src(self, pages_text):
        path = _make_pdf_with_text(pages_text)
        self.tmp_files.append(path)
        return path

    def test_removes_term_from_text(self):
        src = self._src(["Hello SECRET world"])
        out, count = redact_text(src, ["SECRET"])
        self.tmp_files.append(out)
        self.assertEqual(count, 1)
        with fitz.open(out) as doc:
            text = doc[0].get_text()
        self.assertNotIn("SECRET", text)
        self.assertIn("Hello", text)
        self.assertIn("world", text)

    def test_case_insensitive_match(self):
        src = self._src(["The Secret of monkey island"])
        out, count = redact_text(src, ["secret"])
        self.tmp_files.append(out)
        self.assertEqual(count, 1)
        with fitz.open(out) as doc:
            self.assertNotIn("Secret", doc[0].get_text())

    def test_multiple_terms(self):
        src = self._src(["Alice met Bob in Paris"])
        out, count = redact_text(src, ["Alice", "Bob"])
        self.tmp_files.append(out)
        self.assertEqual(count, 2)
        with fitz.open(out) as doc:
            text = doc[0].get_text()
        self.assertNotIn("Alice", text)
        self.assertNotIn("Bob", text)
        self.assertIn("Paris", text)

    def test_multiple_occurrences_same_term(self):
        src = self._src(["foo foo foo bar"])
        out, count = redact_text(src, ["foo"])
        self.tmp_files.append(out)
        self.assertEqual(count, 3)

    def test_term_not_present_returns_zero(self):
        src = self._src(["just some text"])
        out, count = redact_text(src, ["NOT_HERE"])
        self.tmp_files.append(out)
        self.assertEqual(count, 0)
        with fitz.open(out) as doc:
            self.assertIn("just some text", doc[0].get_text())

    def test_page_range_limits_scope(self):
        src = self._src(["badword on page 1", "badword on page 2", "badword on page 3"])
        out, count = redact_text(src, ["badword"], page_range="2")
        self.tmp_files.append(out)
        self.assertEqual(count, 1)
        with fitz.open(out) as doc:
            self.assertIn("badword", doc[0].get_text())  # page 1 untouched
            self.assertNotIn("badword", doc[1].get_text())  # page 2 redacted
            self.assertIn("badword", doc[2].get_text())  # page 3 untouched

    def test_empty_terms_raises(self):
        src = self._src(["any text"])
        with self.assertRaises(ValueError):
            redact_text(src, [])
        with self.assertRaises(ValueError):
            redact_text(src, ["", "  "])

    def test_terms_with_whitespace_are_stripped(self):
        src = self._src(["Hello world"])
        out, count = redact_text(src, ["  world  "])
        self.tmp_files.append(out)
        self.assertEqual(count, 1)

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            redact_text("/no/such/file.pdf", ["foo"])

    def test_invalid_page_range_raises(self):
        src = self._src(["text"])
        with self.assertRaises(ValueError):
            redact_text(src, ["text"], page_range="99")

    def test_encrypted_pdf_raises(self):
        src = self._src(["text"])
        encrypted = protect_pdf(src, user_password="pw")
        self.tmp_files.append(encrypted)
        with self.assertRaises(ValueError):
            redact_text(encrypted, ["text"])

    def test_output_basename_marks_redacted(self):
        src = self._src(["text"])
        out, _ = redact_text(src, ["text"])
        self.tmp_files.append(out)
        self.assertIn("redacted", os.path.basename(out))


def _make_image_only_pdf(text: str = "HELLO OCR", pages: int = 1) -> str:
    """Render `text` as a raster image, then embed each image as a page —
    the resulting PDF has no text layer, so OCR must produce one to make it
    searchable."""
    from PIL import ImageDraw, ImageFont

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    img_w, img_h = 1240, 1754  # ~A4 @ 150dpi
    doc = fitz.open()
    try:
        for _i in range(pages):
            img = PILImage.new("RGB", (img_w, img_h), "white")
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 80)
            except OSError:
                font = ImageFont.load_default()
            draw.text((100, 200), text, fill="black", font=font)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            page = doc.new_page(width=595, height=842)
            page.insert_image(page.rect, stream=buf.getvalue())
        doc.save(path)
    finally:
        doc.close()
    return path


_HAS_TESSERACT = shutil.which("tesseract") is not None


@unittest.skipUnless(_HAS_TESSERACT, "tesseract not installed")
class MakePdfSearchableTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.tmp_files: list[str] = []

    def tearDown(self):
        for p in self.tmp_files:
            if os.path.exists(p):
                os.remove(p)

    def test_image_only_page_gets_text_layer(self):
        src = _make_image_only_pdf("HELLO OCR")
        self.tmp_files.append(src)
        out, pages_ocrd = make_pdf_searchable(src, language="eng", dpi=150)
        self.tmp_files.append(out)
        self.assertEqual(pages_ocrd, 1)
        with fitz.open(out) as doc:
            extracted = doc[0].get_text().upper()
        self.assertIn("HELLO", extracted)

    def test_text_page_is_copied_unchanged(self):
        src = _make_pdf_with_text(["Already selectable"])
        self.tmp_files.append(src)
        out, pages_ocrd = make_pdf_searchable(src, language="eng", dpi=150)
        self.tmp_files.append(out)
        self.assertEqual(pages_ocrd, 0)
        with fitz.open(out) as doc:
            self.assertIn("Already selectable", doc[0].get_text())

    def test_mixed_document_only_ocrs_image_pages(self):
        text_pdf = _make_pdf_with_text(["plain text page"])
        image_pdf = _make_image_only_pdf("SCANNED")
        merged = merge_pdfs([text_pdf, image_pdf])
        self.tmp_files.extend([text_pdf, image_pdf, merged])
        out, pages_ocrd = make_pdf_searchable(merged, language="eng", dpi=150)
        self.tmp_files.append(out)
        self.assertEqual(pages_ocrd, 1)

    def test_invalid_dpi_raises(self):
        src = _make_pdf_with_text(["x"])
        self.tmp_files.append(src)
        with self.assertRaises(ValueError):
            make_pdf_searchable(src, dpi=50)
        with self.assertRaises(ValueError):
            make_pdf_searchable(src, dpi=1000)

    def test_blank_language_raises(self):
        src = _make_pdf_with_text(["x"])
        self.tmp_files.append(src)
        with self.assertRaises(ValueError):
            make_pdf_searchable(src, language="")
        with self.assertRaises(ValueError):
            make_pdf_searchable(src, language="   ")

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            make_pdf_searchable("/no/such/file.pdf")

    def test_encrypted_pdf_raises(self):
        src = _make_pdf_with_text(["secret"])
        encrypted = protect_pdf(src, user_password="pw")
        self.tmp_files.extend([src, encrypted])
        with self.assertRaises(ValueError):
            make_pdf_searchable(encrypted)

    def test_output_basename_marks_searchable(self):
        src = _make_pdf_with_text(["x"])
        self.tmp_files.append(src)
        out, _ = make_pdf_searchable(src, language="eng", dpi=150)
        self.tmp_files.append(out)
        self.assertIn("searchable", os.path.basename(out))


_HAS_GHOSTSCRIPT = shutil.which("gs") is not None


@unittest.skipUnless(_HAS_GHOSTSCRIPT, "ghostscript not installed")
class ConvertToPdfaTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.tmp_files: list[str] = []

    def tearDown(self):
        for p in self.tmp_files:
            if os.path.exists(p):
                os.remove(p)

    def _src(self, pages_text=None):
        path = _make_pdf_with_text(pages_text or ["Archival document body"])
        self.tmp_files.append(path)
        return path

    def test_convert_to_pdfa_2b_returns_pdfa_file(self):
        src = self._src()
        out, version = convert_to_pdfa(src, "2b")
        self.tmp_files.append(out)
        self.assertEqual(version, "2b")
        self.assertTrue(os.path.exists(out))
        with open(out, "rb") as f:
            body = f.read()
        # ghostscript embeds a PDF/A identification block in XMP metadata.
        self.assertIn(b"pdfaid", body)

    def test_convert_to_pdfa_1b_returns_pdfa_file(self):
        src = self._src()
        out, version = convert_to_pdfa(src, "1b")
        self.tmp_files.append(out)
        self.assertEqual(version, "1b")
        with open(out, "rb") as f:
            body = f.read()
        self.assertIn(b"pdfaid", body)

    def test_invalid_version_raises(self):
        src = self._src()
        with self.assertRaises(ValueError):
            convert_to_pdfa(src, "3b")
        with self.assertRaises(ValueError):
            convert_to_pdfa(src, "")

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            convert_to_pdfa("/no/such/file.pdf", "2b")

    def test_encrypted_pdf_raises(self):
        src = self._src()
        encrypted = protect_pdf(src, user_password="pw")
        self.tmp_files.append(encrypted)
        with self.assertRaises(ValueError):
            convert_to_pdfa(encrypted, "2b")

    def test_output_basename_marks_pdfa(self):
        src = self._src()
        out, _ = convert_to_pdfa(src, "2b")
        self.tmp_files.append(out)
        self.assertIn("pdfa2b", os.path.basename(out))


class ComparePdfsTests(_MediaRootMixin, TestCase):
    def setUp(self):
        self.tmp_files: list[str] = []

    def tearDown(self):
        for p in self.tmp_files:
            if os.path.exists(p):
                os.remove(p)

    def _src(self, pages_text):
        path = _make_pdf_with_text(pages_text)
        self.tmp_files.append(path)
        return path

    def test_identical_pdfs_report_no_changes(self):
        a = self._src(["Hello world", "Page two body"])
        b = self._src(["Hello world", "Page two body"])
        out, stats = compare_pdfs(a, b)
        self.tmp_files.append(out)
        self.assertEqual(stats["identical"], 2)
        self.assertEqual(stats["changed"], 0)
        self.assertEqual(stats["added"], 0)
        self.assertEqual(stats["removed"], 0)
        self.assertEqual(stats["pages_a"], 2)
        self.assertEqual(stats["pages_b"], 2)
        self.assertTrue(os.path.exists(out))

    def test_changed_page_is_detected(self):
        a = self._src(["original line"])
        b = self._src(["revised line"])
        out, stats = compare_pdfs(a, b)
        self.tmp_files.append(out)
        self.assertEqual(stats["changed"], 1)
        self.assertEqual(stats["identical"], 0)

    def test_added_pages_are_counted(self):
        a = self._src(["page 1"])
        b = self._src(["page 1", "page 2 new"])
        out, stats = compare_pdfs(a, b)
        self.tmp_files.append(out)
        self.assertEqual(stats["identical"], 1)
        self.assertEqual(stats["added"], 1)
        self.assertEqual(stats["removed"], 0)

    def test_removed_pages_are_counted(self):
        a = self._src(["page 1", "page 2 to be cut"])
        b = self._src(["page 1"])
        out, stats = compare_pdfs(a, b)
        self.tmp_files.append(out)
        self.assertEqual(stats["identical"], 1)
        self.assertEqual(stats["removed"], 1)
        self.assertEqual(stats["added"], 0)

    def test_same_file_path_raises(self):
        a = self._src(["x"])
        with self.assertRaises(ValueError):
            compare_pdfs(a, a)

    def test_missing_file_raises(self):
        a = self._src(["x"])
        with self.assertRaises(ValueError):
            compare_pdfs(a, "/no/such/file.pdf")
        with self.assertRaises(ValueError):
            compare_pdfs("/no/such/file.pdf", a)

    def test_encrypted_pdf_raises(self):
        a = self._src(["secret"])
        b = self._src(["secret"])
        encrypted = protect_pdf(a, user_password="pw")
        self.tmp_files.append(encrypted)
        with self.assertRaises(ValueError):
            compare_pdfs(encrypted, b)

    def test_output_basename_marks_diff(self):
        a = self._src(["a"])
        b = self._src(["b"])
        out, _ = compare_pdfs(a, b)
        self.tmp_files.append(out)
        self.assertIn("diff", os.path.basename(out))
