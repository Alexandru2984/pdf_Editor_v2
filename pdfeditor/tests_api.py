"""Tests for the REST API (auth, PDF management, operations, throttling)."""

import io
import os
import shutil
import tempfile

import fitz
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from rest_framework.test import APIClient

from .models import ApiKey, ProcessedPDF, UploadedPDF

_MEDIA_ROOT = tempfile.mkdtemp(prefix="pdfeditor_api_test_")

User = get_user_model()


def _make_pdf_bytes(num_pages: int = 1) -> bytes:
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"Page {i + 1}", fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


@override_settings(MEDIA_ROOT=_MEDIA_ROOT)
class _ApiTestBase(TestCase):
    """Each subclass gets a fresh APIClient + an isolated MEDIA_ROOT."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.makedirs(os.path.join(_MEDIA_ROOT, "uploads"), exist_ok=True)
        os.makedirs(os.path.join(_MEDIA_ROOT, "processed"), exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(_MEDIA_ROOT, ignore_errors=True)
        super().tearDownClass()

    def setUp(self):
        cache.clear()  # reset throttle counters between tests
        self.client = APIClient()
        self.user = User.objects.create_user(username="alice", password="pw12345!")
        self.api_key_obj, self.token = ApiKey.create_for_user(self.user, label="test")
        self.client.credentials(HTTP_X_API_KEY=self.token)

    def _upload(self, num_pages=2, name="sample.pdf") -> UploadedPDF:
        f = SimpleUploadedFile(name, _make_pdf_bytes(num_pages), content_type="application/pdf")
        resp = self.client.post(reverse("api:pdf-list"), {"pdf_file": f}, format="multipart")
        self.assertEqual(resp.status_code, 201, resp.content)
        return UploadedPDF.objects.get(id=resp.json()["id"])


class ApiAuthTests(_ApiTestBase):
    def test_anonymous_request_rejected(self):
        client = APIClient()
        resp = client.get(reverse("api:pdf-list"))
        self.assertIn(resp.status_code, (401, 403))

    def test_invalid_key_rejected(self):
        client = APIClient()
        client.credentials(HTTP_X_API_KEY="not-a-real-token")
        resp = client.get(reverse("api:pdf-list"))
        self.assertEqual(resp.status_code, 401)

    def test_revoked_key_rejected(self):
        from django.utils import timezone

        self.api_key_obj.revoked_at = timezone.now()
        self.api_key_obj.save()
        resp = self.client.get(reverse("api:pdf-list"))
        self.assertEqual(resp.status_code, 401)

    def test_valid_key_authenticates(self):
        resp = self.client.get(reverse("api:pdf-list"))
        self.assertEqual(resp.status_code, 200)

    def test_last_used_at_updates(self):
        self.assertIsNone(self.api_key_obj.last_used_at)
        self.client.get(reverse("api:pdf-list"))
        self.api_key_obj.refresh_from_db()
        self.assertIsNotNone(self.api_key_obj.last_used_at)


class PdfManagementApiTests(_ApiTestBase):
    def test_upload_creates_uploaded_pdf(self):
        f = SimpleUploadedFile("hello.pdf", _make_pdf_bytes(1), content_type="application/pdf")
        resp = self.client.post(reverse("api:pdf-list"), {"pdf_file": f}, format="multipart")
        self.assertEqual(resp.status_code, 201)
        body = resp.json()
        self.assertEqual(body["name"], "hello.pdf")
        self.assertTrue(UploadedPDF.objects.filter(id=body["id"], user=self.user).exists())

    def test_upload_rejects_non_pdf(self):
        f = SimpleUploadedFile("evil.pdf", b"NOTAPDF" * 10, content_type="application/pdf")
        resp = self.client.post(reverse("api:pdf-list"), {"pdf_file": f}, format="multipart")
        self.assertEqual(resp.status_code, 400)

    def test_upload_missing_file_returns_400(self):
        resp = self.client.post(reverse("api:pdf-list"), {}, format="multipart")
        self.assertEqual(resp.status_code, 400)

    def test_list_only_returns_own_pdfs(self):
        self._upload()
        other = User.objects.create_user(username="bob", password="pw")
        UploadedPDF.objects.create(user=other, name="bob.pdf", path="/tmp/x.pdf", size=1)

        resp = self.client.get(reverse("api:pdf-list"))
        body = resp.json()
        rows = body["results"] if isinstance(body, dict) and "results" in body else body
        self.assertEqual(len(rows), 1)

    def test_delete_removes_pdf(self):
        pdf = self._upload()
        resp = self.client.delete(reverse("api:pdf-detail", args=[pdf.id]))
        self.assertEqual(resp.status_code, 204)
        self.assertFalse(UploadedPDF.objects.filter(id=pdf.id).exists())


class OperationsApiTests(_ApiTestBase):
    def test_compress_creates_output(self):
        pdf = self._upload(num_pages=2)
        resp = self.client.post(
            reverse("api:op-compress"),
            {"pdf_id": str(pdf.id), "quality": "medium"},
            format="json",
        )
        self.assertEqual(resp.status_code, 201, resp.content)
        body = resp.json()
        self.assertEqual(body["kind"], ProcessedPDF.KIND_COMPRESS)
        self.assertIn("download_url", body)

    def test_redact_with_terms(self):
        pdf = self._upload(num_pages=1)
        resp = self.client.post(
            reverse("api:op-redact"),
            {"pdf_id": str(pdf.id), "search_terms": ["Page"]},
            format="json",
        )
        self.assertEqual(resp.status_code, 201, resp.content)
        self.assertEqual(resp.json()["kind"], ProcessedPDF.KIND_REDACT)

    def test_pdfa_creates_output(self):
        if not shutil.which("gs"):
            self.skipTest("ghostscript not installed")
        pdf = self._upload(num_pages=1)
        resp = self.client.post(
            reverse("api:op-pdfa"),
            {"pdf_id": str(pdf.id), "version": "2b"},
            format="json",
        )
        self.assertEqual(resp.status_code, 201, resp.content)
        self.assertEqual(resp.json()["kind"], ProcessedPDF.KIND_PDFA)

    def test_op_with_missing_pdf_returns_404(self):
        resp = self.client.post(
            reverse("api:op-compress"),
            {"pdf_id": "00000000-0000-0000-0000-000000000000", "quality": "medium"},
            format="json",
        )
        self.assertEqual(resp.status_code, 404)

    def test_cannot_operate_on_another_users_pdf(self):
        other = User.objects.create_user(username="bob", password="pw")
        other_pdf = UploadedPDF.objects.create(user=other, name="b.pdf", path="/tmp/none.pdf", size=1)
        resp = self.client.post(
            reverse("api:op-compress"),
            {"pdf_id": str(other_pdf.id), "quality": "medium"},
            format="json",
        )
        self.assertEqual(resp.status_code, 404)

    def test_merge_requires_at_least_two_pdfs(self):
        a = self._upload(num_pages=1)
        b = self._upload(num_pages=1)
        resp = self.client.post(
            reverse("api:op-merge"),
            {"pdf_ids": [str(a.id), str(b.id)]},
            format="json",
        )
        self.assertEqual(resp.status_code, 201, resp.content)
        self.assertEqual(resp.json()["kind"], ProcessedPDF.KIND_MERGE)


class OutputApiTests(_ApiTestBase):
    def test_list_and_download_output(self):
        pdf = self._upload(num_pages=1)
        op = self.client.post(
            reverse("api:op-compress"),
            {"pdf_id": str(pdf.id), "quality": "medium"},
            format="json",
        )
        out_id = op.json()["id"]

        listing = self.client.get(reverse("api:output-list"))
        self.assertEqual(listing.status_code, 200)

        download = self.client.get(reverse("api:output-download", args=[out_id]))
        self.assertEqual(download.status_code, 200)
        self.assertIn("attachment", download["Content-Disposition"])


class ThrottleApiTests(_ApiTestBase):
    """Verify our per-API-key throttle generates distinct keys; the actual
    rate-limiting logic is upstream in DRF and well-covered there."""

    def test_throttle_key_is_per_api_key(self):
        from rest_framework.test import APIRequestFactory

        from .api.throttles import ApiKeyRateThrottle

        factory = APIRequestFactory()
        req1 = factory.get("/api/v1/pdfs/")
        req1.auth = self.api_key_obj

        other_user = User.objects.create_user(username="zoe", password="pw")
        other_key, _ = ApiKey.create_for_user(other_user)
        req2 = factory.get("/api/v1/pdfs/")
        req2.auth = other_key

        throttle = ApiKeyRateThrottle()
        key1 = throttle.get_cache_key(req1, view=None)
        key2 = throttle.get_cache_key(req2, view=None)
        self.assertIsNotNone(key1)
        self.assertIsNotNone(key2)
        self.assertNotEqual(key1, key2)
        self.assertIn("api_key", key1)

    def test_throttle_skips_session_auth(self):
        """When request.auth is not an ApiKey (e.g. session login), our
        throttle returns None so DRF falls back to the default throttles."""
        from rest_framework.test import APIRequestFactory

        from .api.throttles import ApiKeyRateThrottle

        req = APIRequestFactory().get("/api/v1/pdfs/")
        req.auth = None
        self.assertIsNone(ApiKeyRateThrottle().get_cache_key(req, view=None))


class OpenApiSchemaTests(_ApiTestBase):
    def test_schema_endpoint_returns_yaml(self):
        # Schema and docs are public — use anon client.
        anon = APIClient()
        resp = anon.get(reverse("api:schema"))
        self.assertEqual(resp.status_code, 200)
        body = resp.content.decode("utf-8")
        self.assertIn("PDF Editor API", body)
        self.assertIn("/api/v1/pdfs/", body)

    def test_swagger_renders(self):
        anon = APIClient()
        resp = anon.get(reverse("api:docs"))
        self.assertEqual(resp.status_code, 200)

    def test_redoc_renders(self):
        anon = APIClient()
        resp = anon.get(reverse("api:redoc"))
        self.assertEqual(resp.status_code, 200)
