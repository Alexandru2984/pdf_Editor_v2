"""End-to-end view tests covering upload, basic_ops, layout_ops, edit, extract, rephrase."""

import io
import os
import shutil
import tempfile
from unittest.mock import AsyncMock, patch

import fitz
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from PIL import Image as PILImage

# ---- Fixture helpers --------------------------------------------------------


def _multipage_pdf_bytes(num_pages=2, text_prefix="Page"):
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"{text_prefix} {i + 1} content here.", fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _png_bytes(size=(50, 50), color=(0, 200, 0, 255)):
    img = PILImage.new("RGBA", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_uploaded(name, num_pages=2, text_prefix="Page"):
    return SimpleUploadedFile(
        name,
        _multipage_pdf_bytes(num_pages, text_prefix),
        content_type="application/pdf",
    )


@override_settings(MEDIA_ROOT=tempfile.mkdtemp(prefix="pdfedit_views_"))
class _ViewTestBase(TestCase):
    """Each subclass gets a fresh client + an isolated MEDIA_ROOT (for the class)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Ensure MEDIA_ROOT exists.
        from django.conf import settings as dj_settings

        os.makedirs(dj_settings.MEDIA_ROOT, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        from django.conf import settings as dj_settings

        shutil.rmtree(dj_settings.MEDIA_ROOT, ignore_errors=True)
        super().tearDownClass()

    def setUp(self):
        self.client = Client()

    def upload(self, name="x.pdf", num_pages=2, text_prefix="Page"):
        """POST a PDF and return the redirect response."""
        return self.client.post(
            reverse("upload"),
            {"pdf_file": _make_uploaded(name, num_pages, text_prefix)},
        )

    def upload_and_get_pdf(self, **kwargs):
        from .models import UploadedPDF

        self.upload(**kwargs)
        return UploadedPDF.objects.first()


# ---- Upload + dashboard -----------------------------------------------------


class UploadViewExtraTests(_ViewTestBase):
    """Beyond what tests.py covers — page-count limit, magic-byte rejection, multi-file."""

    def test_dashboard_renders_empty(self):
        resp = self.client.get(reverse("dashboard"))
        self.assertEqual(resp.status_code, 200)

    def test_upload_no_files_shows_error(self):
        resp = self.client.post(reverse("upload"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "select at least one")

    def test_upload_rejects_oversize(self):
        big = SimpleUploadedFile(
            "big.pdf",
            b"%PDF-" + b"\x00" * (11 * 1024 * 1024),
            content_type="application/pdf",
        )
        resp = self.client.post(reverse("upload"), {"pdf_file": big})
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "exceeds")

    def test_upload_rejects_non_pdf_magic_bytes(self):
        bad = SimpleUploadedFile("fake.pdf", b"NOTAPDF" * 10, content_type="application/pdf")
        resp = self.client.post(reverse("upload"), {"pdf_file": bad})
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "not a valid PDF")

    @override_settings(PDF_MAX_PAGES=2)
    def test_upload_rejects_too_many_pages(self):
        resp = self.client.post(reverse("upload"), {"pdf_file": _make_uploaded("big.pdf", num_pages=5)})
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "exceeds the 2-page limit")

    def test_upload_rejects_unparseable_pdf_bytes(self):
        # Magic bytes pass but body is garbage — fitz should fail to parse.
        bad = SimpleUploadedFile(
            "evil.pdf",
            b"%PDF-1.4\nthis is not a real pdf at all just text\n%%EOF\n",
            content_type="application/pdf",
        )
        resp = self.client.post(reverse("upload"), {"pdf_file": bad})
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "could not be parsed")

    def test_upload_multiple_files_redirects_to_dashboard(self):
        resp = self.client.post(
            reverse("upload"),
            {
                "pdf_file": [_make_uploaded("a.pdf"), _make_uploaded("b.pdf")],
            },
        )
        self.assertEqual(resp.status_code, 302)
        self.assertTrue(resp.url.endswith(reverse("dashboard")))

    def test_delete_pdf_removes_row(self):
        from .models import UploadedPDF

        self.upload(name="del.pdf")
        pdf = UploadedPDF.objects.first()
        resp = self.client.get(reverse("delete_pdf", args=[pdf.id]))
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(UploadedPDF.objects.count(), 0)

    def test_delete_pdf_unknown_id_redirects(self):
        # UUID that doesn't exist — should redirect with error message, not 500.
        resp = self.client.get(reverse("delete_pdf", args=["00000000-0000-0000-0000-000000000000"]))
        self.assertEqual(resp.status_code, 302)


# ---- Edit (find/replace) ----------------------------------------------------


class EditViewTests(_ViewTestBase):
    def test_get_without_upload_redirects(self):
        resp = self.client.get(reverse("edit"))
        self.assertEqual(resp.status_code, 302)
        self.assertTrue(resp.url.endswith(reverse("dashboard")))

    def test_get_with_uploaded_pdf_renders(self):
        self.upload()
        resp = self.client.get(reverse("edit"))
        self.assertEqual(resp.status_code, 200)

    def test_post_invalid_form_renders_with_errors(self):
        self.upload()
        resp = self.client.post(reverse("edit"), {"search_text": "", "replace_text": ""})
        self.assertEqual(resp.status_code, 200)

    def test_full_workflow_edit_result_download(self):
        self.upload(text_prefix="Hello")  # produces "Hello 1 content here."
        resp = self.client.post(
            reverse("edit"),
            {
                "search_text": "Hello",
                "replace_text": "Goodbye",
                "case_sensitive": True,
                "page_range": "",
            },
        )
        self.assertEqual(resp.status_code, 302)

        result = self.client.get(reverse("result"))
        self.assertEqual(result.status_code, 200)

        download = self.client.get(reverse("download"))
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download["Content-Type"], "application/pdf")

    def test_result_without_processing_redirects(self):
        resp = self.client.get(reverse("result"))
        self.assertEqual(resp.status_code, 302)

    def test_preview_uploaded_renders(self):
        self.upload()
        resp = self.client.get(reverse("preview"))
        self.assertEqual(resp.status_code, 200)

    def test_preview_processed_renders_after_edit(self):
        self.upload(text_prefix="Hello")
        self.client.post(
            reverse("edit"),
            {
                "search_text": "Hello",
                "replace_text": "Hi",
                "case_sensitive": True,
                "page_range": "",
            },
        )
        resp = self.client.get(reverse("preview") + "?type=processed")
        self.assertEqual(resp.status_code, 200)

    def test_preview_with_no_pdf_redirects(self):
        resp = self.client.get(reverse("preview"))
        self.assertEqual(resp.status_code, 302)


# ---- Split / Merge / Compress ----------------------------------------------


class SplitViewTests(_ViewTestBase):
    def test_get_renders_form(self):
        self.upload(num_pages=5)
        resp = self.client.get(reverse("split"))
        self.assertEqual(resp.status_code, 200)

    def test_split_workflow_produces_files(self):
        self.upload(num_pages=5, name="s.pdf")
        resp = self.client.post(reverse("split"), {"ranges": "1-2,4"})
        self.assertEqual(resp.status_code, 302)

        result = self.client.get(reverse("split_result"))
        self.assertEqual(result.status_code, 200)

        # Two output files (1-2 and 4).
        download0 = self.client.get(reverse("download_split") + "?file=0")
        self.assertEqual(download0.status_code, 200)
        download1 = self.client.get(reverse("download_split") + "?file=1")
        self.assertEqual(download1.status_code, 200)

    def test_split_invalid_range_keeps_form(self):
        self.upload(num_pages=3)
        resp = self.client.post(reverse("split"), {"ranges": "invalid"})
        self.assertEqual(resp.status_code, 200)

    def test_split_out_of_bounds_shows_error(self):
        self.upload(num_pages=3)
        resp = self.client.post(reverse("split"), {"ranges": "1-99"})
        # Form-level validator passes (start<=end, both >=1) but split_pdf raises ValueError.
        self.assertEqual(resp.status_code, 200)

    def test_split_result_without_processing_redirects(self):
        resp = self.client.get(reverse("split_result"))
        self.assertEqual(resp.status_code, 302)

    def test_download_split_invalid_index_404(self):
        self.upload(num_pages=3)
        self.client.post(reverse("split"), {"ranges": "1"})
        resp = self.client.get(reverse("download_split") + "?file=99")
        self.assertEqual(resp.status_code, 404)

    def test_download_split_no_param_404(self):
        resp = self.client.get(reverse("download_split"))
        self.assertEqual(resp.status_code, 404)


class MergeViewTests(_ViewTestBase):
    def test_with_one_pdf_redirects(self):
        self.upload()
        resp = self.client.get(reverse("merge"))
        self.assertEqual(resp.status_code, 302)

    def test_with_two_pdfs_renders_form(self):
        self.upload(name="a.pdf")
        self.upload(name="b.pdf")
        resp = self.client.get(reverse("merge"))
        self.assertEqual(resp.status_code, 200)

    def test_full_merge_workflow(self):
        from .models import UploadedPDF

        self.upload(name="a.pdf")
        self.upload(name="b.pdf")
        ids = list(UploadedPDF.objects.values_list("id", flat=True))
        resp = self.client.post(
            reverse("merge"),
            {
                "selected_pdfs": ",".join(str(i) for i in ids),
                "output_name": "combined",
            },
        )
        self.assertEqual(resp.status_code, 302)

        result = self.client.get(reverse("merge_result"))
        self.assertEqual(result.status_code, 200)

        download = self.client.get(reverse("download_merged"))
        self.assertEqual(download.status_code, 200)


class CompressViewTests(_ViewTestBase):
    def test_get_renders(self):
        self.upload()
        resp = self.client.get(reverse("compress"))
        self.assertEqual(resp.status_code, 200)

    def test_full_compress_workflow(self):
        self.upload()
        resp = self.client.post(reverse("compress"), {"quality": "medium"})
        self.assertEqual(resp.status_code, 302)

        result = self.client.get(reverse("compress_result"))
        self.assertEqual(result.status_code, 200)

        download = self.client.get(reverse("download_compressed"))
        self.assertEqual(download.status_code, 200)


# ---- Convert to DOCX -------------------------------------------------------


class ConvertViewTests(_ViewTestBase):
    def test_get_without_upload_redirects(self):
        resp = self.client.get(reverse("convert"))
        self.assertEqual(resp.status_code, 302)

    def test_get_renders(self):
        self.upload()
        resp = self.client.get(reverse("convert"))
        self.assertEqual(resp.status_code, 200)

    def test_full_convert_workflow(self):
        self.upload()
        resp = self.client.post(reverse("convert"), {})
        self.assertEqual(resp.status_code, 302)

        result = self.client.get(reverse("convert_result"))
        self.assertEqual(result.status_code, 200)
        # Filename ends with .docx and is offered as a download.
        download = self.client.get(reverse("download_converted"))
        self.assertEqual(download.status_code, 200)
        # Recorded ProcessedPDF has the right kind and a .docx path.
        from .models import ProcessedPDF

        latest = ProcessedPDF.objects.first()
        self.assertEqual(latest.kind, ProcessedPDF.KIND_CONVERT)
        self.assertTrue(latest.path.endswith(".docx"))

    def test_convert_result_without_session_redirects(self):
        resp = self.client.get(reverse("convert_result"))
        self.assertEqual(resp.status_code, 302)

    def test_download_without_session_redirects(self):
        resp = self.client.get(reverse("download_converted"))
        self.assertEqual(resp.status_code, 302)


# ---- Watermark / Rotate / Page numbers --------------------------------------


class WatermarkViewTests(_ViewTestBase):
    def test_get_renders(self):
        self.upload()
        resp = self.client.get(reverse("watermark"))
        self.assertEqual(resp.status_code, 200)

    def test_text_watermark_workflow(self):
        self.upload()
        # rotation=45 forces the snap-to-90 path in ops.py.
        resp = self.client.post(
            reverse("watermark"),
            {
                "watermark_type": "text",
                "text_content": "DRAFT",
                "font_size": 48,
                "position": "center",
                "opacity": 0.3,
                "rotation": 45,
            },
        )
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(self.client.get(reverse("watermark_result")).status_code, 200)
        self.assertEqual(self.client.get(reverse("download_watermarked")).status_code, 200)

    def test_image_watermark_workflow(self):
        self.upload()
        png = SimpleUploadedFile("logo.png", _png_bytes(), content_type="image/png")
        resp = self.client.post(
            reverse("watermark"),
            {
                "watermark_type": "image",
                "watermark_image": png,
                "position": "top-right",
                "opacity": 0.5,
                "rotation": 0,
            },
        )
        self.assertEqual(resp.status_code, 302)

    def test_text_watermark_missing_content_invalid(self):
        self.upload()
        resp = self.client.post(
            reverse("watermark"),
            {
                "watermark_type": "text",
                "text_content": "",
                "position": "center",
                "opacity": 0.3,
                "rotation": 0,
            },
        )
        self.assertEqual(resp.status_code, 200)


class RotateViewTests(_ViewTestBase):
    def test_get_renders(self):
        self.upload()
        resp = self.client.get(reverse("rotate"))
        self.assertEqual(resp.status_code, 200)

    def test_rotate_workflow(self):
        self.upload(num_pages=3)
        resp = self.client.post(reverse("rotate"), {"rotation_angle": 90, "page_range": ""})
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(self.client.get(reverse("rotate_result")).status_code, 200)
        self.assertEqual(self.client.get(reverse("download_rotated")).status_code, 200)

    def test_rotate_out_of_range_shows_error(self):
        self.upload(num_pages=2)
        resp = self.client.post(reverse("rotate"), {"rotation_angle": 90, "page_range": "5-10"})
        self.assertEqual(resp.status_code, 200)


class PageNumbersViewTests(_ViewTestBase):
    def test_get_renders(self):
        self.upload()
        resp = self.client.get(reverse("page_numbers"))
        self.assertEqual(resp.status_code, 200)

    def test_page_numbers_workflow(self):
        self.upload(num_pages=3)
        resp = self.client.post(
            reverse("page_numbers"),
            {
                "position": "bottom-center",
                "format": "page_number",
                "font_size": 12,
                "start_page": 1,
            },
        )
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(self.client.get(reverse("page_numbers_result")).status_code, 200)
        self.assertEqual(self.client.get(reverse("download_numbered")).status_code, 200)


# ---- Extract + OCR ----------------------------------------------------------


class ExtractViewTests(_ViewTestBase):
    def test_more_tools_no_pdf_redirects(self):
        resp = self.client.get(reverse("more_tools"))
        self.assertEqual(resp.status_code, 302)

    def test_more_tools_with_pdf_renders(self):
        self.upload()
        resp = self.client.get(reverse("more_tools"))
        self.assertEqual(resp.status_code, 200)

    def test_extract_text_ajax_returns_json(self):
        pdf = self.upload_and_get_pdf(text_prefix="Extract")
        resp = self.client.post(reverse("extract_text", args=[pdf.id]))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["Content-Type"], "application/json")
        body = resp.json()
        self.assertTrue(body["success"])
        self.assertIn("Extract", body["text"])

    def test_extract_text_get_returns_invalid_method(self):
        pdf = self.upload_and_get_pdf()
        resp = self.client.get(reverse("extract_text", args=[pdf.id]))
        body = resp.json()
        self.assertFalse(body["success"])

    def test_extract_text_unknown_pdf_returns_error(self):
        resp = self.client.post(reverse("extract_text", args=["00000000-0000-0000-0000-000000000000"]))
        body = resp.json()
        self.assertFalse(body["success"])

    @patch("pdfeditor.views.extract.ocr_pdf_to_text", return_value="MOCKED OCR RESULT")
    def test_ocr_text_ajax_uses_extractor(self, mock_ocr):
        pdf = self.upload_and_get_pdf()
        resp = self.client.post(reverse("ocr_text", args=[pdf.id]))
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["text"], "MOCKED OCR RESULT")
        mock_ocr.assert_called_once_with(pdf.path)

    def test_download_text_without_session_redirects(self):
        resp = self.client.get(reverse("download_text"))
        self.assertEqual(resp.status_code, 302)

    def test_download_text_after_extract_works(self):
        pdf = self.upload_and_get_pdf(text_prefix="Hello")
        self.client.post(reverse("extract_text", args=[pdf.id]))
        resp = self.client.get(reverse("download_text"))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["Content-Type"], "text/plain")


# ---- Rephrase ---------------------------------------------------------------


class RephraseViewTests(_ViewTestBase):
    def test_get_without_upload_redirects(self):
        resp = self.client.get(reverse("rephrase"))
        self.assertEqual(resp.status_code, 302)

    @patch("pdfeditor.views.rephrase.get_all_models", return_value={"ollama": ["llama3"], "groq": []})
    def test_get_with_upload_renders(self, _models):
        self.upload()
        resp = self.client.get(reverse("rephrase"))
        self.assertEqual(resp.status_code, 200)

    @patch("pdfeditor.views.rephrase.get_all_models", return_value={"ollama": ["llama3"], "groq": []})
    @patch("pdfeditor.views.rephrase.get_provider")
    def test_post_happy_path(self, mock_provider, _models):
        mock_provider.return_value.rephrase.return_value = ("Rephrased.", True, "")
        self.upload()

        resp = self.client.post(
            reverse("rephrase"),
            {
                "selected_text": "Page 1 content here.",
                "rephrase_style": "formal",
                "model": "llama3",
                "page_number": "0",
                "bbox_x0": "70",
                "bbox_y0": "732",
                "bbox_x1": "350",
                "bbox_y1": "742",
            },
        )
        self.assertEqual(resp.status_code, 302)
        self.assertTrue(resp.url.endswith(reverse("rephrase_result")))

        # Result + download flow.
        result = self.client.get(reverse("rephrase_result"))
        self.assertEqual(result.status_code, 200)
        download = self.client.get(reverse("download_rephrased"))
        self.assertEqual(download.status_code, 200)

    @patch("pdfeditor.views.rephrase.get_all_models", return_value={"ollama": ["llama3"], "groq": []})
    def test_post_missing_selected_text(self, _models):
        self.upload()
        resp = self.client.post(
            reverse("rephrase"),
            {
                "selected_text": "",
                "rephrase_style": "formal",
                "model": "llama3",
                "page_number": "0",
                "bbox_x0": "0",
                "bbox_y0": "0",
                "bbox_x1": "0",
                "bbox_y1": "0",
            },
        )
        self.assertEqual(resp.status_code, 200)  # rendered with error

    @patch("pdfeditor.views.rephrase.get_all_models", return_value={"ollama": ["llama3"], "groq": []})
    def test_post_zero_bbox_is_rejected(self, _models):
        self.upload()
        resp = self.client.post(
            reverse("rephrase"),
            {
                "selected_text": "some text",
                "rephrase_style": "formal",
                "model": "llama3",
                "page_number": "0",
                "bbox_x0": "0",
                "bbox_y0": "0",
                "bbox_x1": "0",
                "bbox_y1": "0",
            },
        )
        self.assertEqual(resp.status_code, 200)

    @patch("pdfeditor.views.rephrase.get_all_models", return_value={"ollama": ["llama3"], "groq": []})
    @patch("pdfeditor.views.rephrase.get_provider")
    def test_post_provider_failure_shows_error(self, mock_provider, _models):
        mock_provider.return_value.rephrase.return_value = ("", False, "API down")
        self.upload()
        resp = self.client.post(
            reverse("rephrase"),
            {
                "selected_text": "Page 1",
                "rephrase_style": "formal",
                "model": "llama3",
                "page_number": "0",
                "bbox_x0": "70",
                "bbox_y0": "732",
                "bbox_x1": "350",
                "bbox_y1": "742",
            },
        )
        self.assertEqual(resp.status_code, 200)

    def test_rephrase_preview_get_returns_405(self):
        resp = self.client.get(reverse("rephrase_preview"))
        self.assertEqual(resp.status_code, 405)

    def test_rephrase_preview_no_text_returns_400(self):
        resp = self.client.post(reverse("rephrase_preview"), {"text": ""})
        self.assertEqual(resp.status_code, 400)

    @patch("pdfeditor.views.rephrase.get_all_models", return_value={"ollama": ["llama3"], "groq": []})
    @patch("pdfeditor.views.rephrase.get_provider")
    def test_rephrase_preview_happy_path(self, mock_provider, _models):
        # Preview endpoint runs sync (WSGI-compatible) and calls `rephrase`.
        mock_provider.return_value.rephrase.return_value = ("Better text.", True, "")
        resp = self.client.post(
            reverse("rephrase_preview"),
            {
                "text": "hello world",
                "style": "formal",
                "model": "llama3",
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["rephrased_text"], "Better text.")

    def test_rephrase_result_without_session_redirects(self):
        resp = self.client.get(reverse("rephrase_result"))
        self.assertEqual(resp.status_code, 302)

    def test_download_rephrased_without_session_redirects(self):
        resp = self.client.get(reverse("download_rephrased"))
        self.assertEqual(resp.status_code, 302)


# ---- Media serving (path-traversal guard) ----------------------------------


class ServeMediaViewTests(_ViewTestBase):
    def test_serves_session_owned_file(self):
        from django.conf import settings as dj_settings

        pdf = self.upload_and_get_pdf()
        rel = os.path.relpath(pdf.path, dj_settings.MEDIA_ROOT)
        resp = self.client.get(f"/media/{rel}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["Content-Type"], "application/pdf")

    def test_path_traversal_blocked(self):
        # Try to escape MEDIA_ROOT via "..".
        resp = self.client.get("/media/../etc/passwd")
        self.assertIn(resp.status_code, (404, 301))  # 301 if django normalizes the URL

    def test_unowned_file_returns_404(self):
        # Upload as session A, then make a fresh client (session B) and ask for the same path.
        from django.conf import settings as dj_settings

        pdf = self.upload_and_get_pdf()
        rel = os.path.relpath(pdf.path, dj_settings.MEDIA_ROOT)

        other = Client()
        resp = other.get(f"/media/{rel}")
        self.assertEqual(resp.status_code, 404)

    def test_nonexistent_path_returns_404(self):
        resp = self.client.get("/media/uploads/not_a_real_file.pdf")
        self.assertEqual(resp.status_code, 404)


# ---- Form fill --------------------------------------------------------------


def _form_pdf_bytes():
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    text_widget = fitz.Widget()
    text_widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
    text_widget.field_name = "full_name"
    text_widget.rect = fitz.Rect(100, 100, 400, 130)
    page.add_widget(text_widget)

    cb_widget = fitz.Widget()
    cb_widget.field_type = fitz.PDF_WIDGET_TYPE_CHECKBOX
    cb_widget.field_name = "agree"
    cb_widget.rect = fitz.Rect(100, 200, 120, 220)
    page.add_widget(cb_widget)

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


class FormFillViewTests(_ViewTestBase):
    def _upload_form_pdf(self, name="form.pdf"):
        return self.client.post(
            reverse("upload"),
            {"pdf_file": SimpleUploadedFile(name, _form_pdf_bytes(), content_type="application/pdf")},
        )

    def test_get_without_pdf_redirects(self):
        resp = self.client.get(reverse("form_fill"))
        self.assertEqual(resp.status_code, 302)

    def test_get_with_form_pdf_renders_fields(self):
        self._upload_form_pdf()
        resp = self.client.get(reverse("form_fill"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "full_name")
        self.assertContains(resp, "agree")

    def test_get_with_plain_pdf_redirects_with_warning(self):
        # Plain PDF without widgets — should redirect to dashboard.
        self.upload(name="plain.pdf")
        resp = self.client.get(reverse("form_fill"))
        self.assertEqual(resp.status_code, 302)
        self.assertTrue(resp.url.endswith(reverse("dashboard")))

    def test_post_fills_text_field(self):
        self._upload_form_pdf()
        resp = self.client.post(
            reverse("form_fill"),
            {
                "field_full_name": "Alice Example",
            },
        )
        self.assertEqual(resp.status_code, 302)
        self.assertTrue(resp.url.endswith(reverse("form_fill_result")))

        result = self.client.get(reverse("form_fill_result"))
        self.assertEqual(result.status_code, 200)
        self.assertContains(result, "Form Filled")

        download = self.client.get(reverse("download_filled"))
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download["Content-Type"], "application/pdf")

    def test_post_with_checkbox_on(self):
        self._upload_form_pdf()
        resp = self.client.post(
            reverse("form_fill"),
            {
                "field_full_name": "Bob",
                "field_agree": "on",
            },
        )
        self.assertEqual(resp.status_code, 302)

    def test_post_with_flatten_writes_text_into_page(self):
        self._upload_form_pdf()
        resp = self.client.post(
            reverse("form_fill"),
            {
                "field_full_name": "FLATTENED VALUE",
                "flatten": "on",
            },
        )
        self.assertEqual(resp.status_code, 302)

        download = self.client.get(reverse("download_filled"))
        self.assertEqual(download.status_code, 200)
        # Download the streamed FileResponse to disk and re-open with fitz.
        fd, out_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        try:
            with open(out_path, "wb") as f:
                f.write(b"".join(download.streaming_content))
            with fitz.open(out_path) as d:
                self.assertIn("FLATTENED VALUE", d[0].get_text())
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_result_without_session_redirects(self):
        resp = self.client.get(reverse("form_fill_result"))
        self.assertEqual(resp.status_code, 302)

    def test_download_without_session_redirects(self):
        resp = self.client.get(reverse("download_filled"))
        self.assertEqual(resp.status_code, 302)

    def test_unknown_pdf_id_redirects(self):
        self._upload_form_pdf()
        resp = self.client.get(
            reverse("form_fill") + "?pdf=00000000-0000-0000-0000-000000000000",
        )
        self.assertEqual(resp.status_code, 302)


# ---- History ---------------------------------------------------------------


class HistoryViewTests(_ViewTestBase):
    def test_empty_history_renders_empty_state(self):
        resp = self.client.get(reverse("history"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "No operations yet")

    def test_history_lists_processed_pdfs(self):
        # Run a compress operation to generate a history entry.
        self.upload()
        self.client.post(reverse("compress"), {"quality": "medium"})

        resp = self.client.get(reverse("history"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Compress")

    def test_history_skips_other_sessions(self):
        from .models import ProcessedPDF

        # Other session's outputs shouldn't appear here.
        ProcessedPDF.objects.create(
            session_key="some-other-session",
            kind=ProcessedPDF.KIND_SPLIT,
            name="other.pdf",
            path="/tmp/nope.pdf",
            size=0,
        )
        resp = self.client.get(reverse("history"))
        self.assertEqual(resp.status_code, 200)
        self.assertNotContains(resp, "other.pdf")

    def test_history_prunes_rows_with_missing_files(self):
        # Create a row pointing to a non-existent file under our session.
        from django.conf import settings as dj_settings

        from .models import ProcessedPDF

        # Make sure session_key is initialized.
        self.client.get(reverse("dashboard"))
        session_key = self.client.session.session_key

        ProcessedPDF.objects.create(
            session_key=session_key,
            kind=ProcessedPDF.KIND_SPLIT,
            name="ghost.pdf",
            path=os.path.join(dj_settings.MEDIA_ROOT, "processed", "ghost.pdf"),
            size=0,
        )
        self.assertEqual(ProcessedPDF.objects.count(), 1)

        resp = self.client.get(reverse("history"))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(ProcessedPDF.objects.count(), 0)

    def test_history_download_returns_file(self):
        self.upload()
        self.client.post(reverse("compress"), {"quality": "medium"})
        from .models import ProcessedPDF

        out = ProcessedPDF.objects.first()

        resp = self.client.get(reverse("history_download", args=[out.id]))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["Content-Type"], "application/pdf")

    def test_history_download_unowned_redirects(self):
        from .models import ProcessedPDF

        # Create an output for a different session.
        out = ProcessedPDF.objects.create(
            session_key="not-mine",
            kind=ProcessedPDF.KIND_SPLIT,
            name="x.pdf",
            path="/tmp/nope.pdf",
            size=0,
        )
        resp = self.client.get(reverse("history_download", args=[out.id]))
        self.assertEqual(resp.status_code, 302)

    def test_history_delete_removes_row(self):
        self.upload()
        self.client.post(reverse("compress"), {"quality": "medium"})
        from .models import ProcessedPDF

        out = ProcessedPDF.objects.first()
        self.assertIsNotNone(out)

        resp = self.client.get(reverse("history_delete", args=[out.id]))
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(ProcessedPDF.objects.filter(id=out.id).count(), 0)

    def test_history_delete_unowned_does_not_delete(self):
        from .models import ProcessedPDF

        other = ProcessedPDF.objects.create(
            session_key="not-mine",
            kind=ProcessedPDF.KIND_SPLIT,
            name="x.pdf",
            path="/tmp/nope.pdf",
            size=0,
        )
        self.client.get(reverse("history_delete", args=[other.id]))
        # The row still exists — we only deleted within our session scope.
        self.assertTrue(ProcessedPDF.objects.filter(id=other.id).exists())


class VerifySignatureViewTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_get_renders_form(self):
        resp = self.client.get(reverse("verify_signature"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Verify signature")

    def test_post_unsigned_pdf_warns(self):
        unsigned = SimpleUploadedFile("x.pdf", _multipage_pdf_bytes(1), content_type="application/pdf")
        resp = self.client.post(reverse("verify_signature"), {"pdf_file": unsigned}, follow=False)
        self.assertEqual(resp.status_code, 200)
        # Empty report → user sees a warning message
        self.assertContains(resp, "No signatures found")

    def test_post_signed_pdf_renders_report(self):
        # Build a signed PDF on disk
        import os
        import tempfile

        from .pdf_processor.ops import sign_pdf
        from .tests_ops import _make_self_signed_p12

        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(_multipage_pdf_bytes(1))
        try:
            p12, cert_pem = _make_self_signed_p12(b"pw")
            signed_path = sign_pdf(path, p12_bytes=p12, p12_password="pw")
            try:
                with open(signed_path, "rb") as sf:
                    pdf_blob = sf.read()
                upload = SimpleUploadedFile("signed.pdf", pdf_blob, content_type="application/pdf")
                trust = SimpleUploadedFile("trust.pem", cert_pem, content_type="application/x-pem-file")
                resp = self.client.post(
                    reverse("verify_signature"),
                    {"pdf_file": upload, "trust_certs": trust},
                )
                self.assertEqual(resp.status_code, 200)
                self.assertContains(resp, "Signature1")
                self.assertContains(resp, "Test Signer")
                # With trust anchor provided → "Valid & trusted"
                self.assertContains(resp, "Valid & trusted")
            finally:
                if os.path.exists(signed_path):
                    os.remove(signed_path)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_global_trust_anchor_makes_signature_trusted(self):
        import os
        import tempfile

        from .models import TrustAnchor
        from .pdf_processor.ops import sign_pdf
        from .tests_ops import _make_self_signed_p12

        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(_multipage_pdf_bytes(1))
        try:
            p12, cert_pem = _make_self_signed_p12(b"pw")
            signed_path = sign_pdf(path, p12_bytes=p12, p12_password="pw")
            try:
                TrustAnchor.objects.create(
                    name="Test Signer (global)",
                    cert_pem=cert_pem.decode("utf-8"),
                    is_active=True,
                )
                with open(signed_path, "rb") as sf:
                    pdf_blob = sf.read()
                upload = SimpleUploadedFile("signed.pdf", pdf_blob, content_type="application/pdf")
                # No per-request trust file — the global anchor must be enough.
                resp = self.client.post(reverse("verify_signature"), {"pdf_file": upload})
                self.assertEqual(resp.status_code, 200)
                self.assertContains(resp, "Valid & trusted")
            finally:
                if os.path.exists(signed_path):
                    os.remove(signed_path)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_inactive_trust_anchor_does_not_trust(self):
        import os
        import tempfile

        from .models import TrustAnchor
        from .pdf_processor.ops import sign_pdf
        from .tests_ops import _make_self_signed_p12

        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(_multipage_pdf_bytes(1))
        try:
            p12, cert_pem = _make_self_signed_p12(b"pw")
            signed_path = sign_pdf(path, p12_bytes=p12, p12_password="pw")
            try:
                TrustAnchor.objects.create(
                    name="Disabled anchor",
                    cert_pem=cert_pem.decode("utf-8"),
                    is_active=False,
                )
                with open(signed_path, "rb") as sf:
                    pdf_blob = sf.read()
                upload = SimpleUploadedFile("signed.pdf", pdf_blob, content_type="application/pdf")
                resp = self.client.post(reverse("verify_signature"), {"pdf_file": upload})
                self.assertEqual(resp.status_code, 200)
                self.assertContains(resp, "Intact but untrusted")
            finally:
                if os.path.exists(signed_path):
                    os.remove(signed_path)
        finally:
            if os.path.exists(path):
                os.remove(path)


class GenerateCertViewTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_get_renders_form(self):
        resp = self.client.get(reverse("generate_cert"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Generate test certificate")

    def test_post_returns_p12_download(self):
        resp = self.client.post(
            reverse("generate_cert"),
            {
                "common_name": "Test User",
                "passphrase": "test1234",
                "passphrase_confirm": "test1234",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["Content-Type"], "application/x-pkcs12")
        self.assertIn("attachment", resp["Content-Disposition"])
        self.assertGreater(len(resp.content), 1000)  # a real .p12 archive

    def test_post_passphrase_mismatch_errors(self):
        resp = self.client.post(
            reverse("generate_cert"),
            {
                "common_name": "Test User",
                "passphrase": "test1234",
                "passphrase_confirm": "different",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "do not match")
