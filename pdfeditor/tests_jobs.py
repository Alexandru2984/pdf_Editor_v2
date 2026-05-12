"""Tests for the async job flow (Celery tasks, web + API).

Celery runs in eager mode in tests (set in settings.py when 'test' is in
argv), so calling a task .delay() executes it synchronously and we can
assert on the resulting Job/ProcessedPDF rows immediately.
"""

import io
import os
import shutil
import tempfile

import fitz
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from rest_framework.test import APIClient

from .models import ApiKey, Job, ProcessedPDF, UploadedPDF

User = get_user_model()


def _pdf_bytes(num_pages: int = 1) -> bytes:
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"Page {i + 1}", fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


_MEDIA_ROOT = tempfile.mkdtemp(prefix="pdfeditor_jobs_test_")


@override_settings(MEDIA_ROOT=_MEDIA_ROOT)
class _JobTestBase(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.makedirs(os.path.join(_MEDIA_ROOT, "uploads"), exist_ok=True)
        os.makedirs(os.path.join(_MEDIA_ROOT, "processed"), exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(_MEDIA_ROOT, ignore_errors=True)
        super().tearDownClass()

    def _make_uploaded_pdf(self, name="src.pdf", num_pages=1) -> UploadedPDF:
        path = os.path.join(_MEDIA_ROOT, "uploads", name)
        with open(path, "wb") as f:
            f.write(_pdf_bytes(num_pages))
        return UploadedPDF.objects.create(user=self.user, name=name, path=path, size=os.path.getsize(path))


class JobTaskTests(_JobTestBase):
    def setUp(self):
        self.user = User.objects.create_user(username="alice", password="pw")

    def test_pdfa_task_marks_job_done_and_creates_output(self):
        if not shutil.which("gs"):
            self.skipTest("ghostscript not installed")
        from .tasks import run_pdfa_task

        src = self._make_uploaded_pdf()
        job = Job.objects.create(
            user=self.user, kind=ProcessedPDF.KIND_PDFA, source=src, params={"version": "2b"}
        )
        run_pdfa_task(str(job.id))
        job.refresh_from_db()
        self.assertEqual(job.status, Job.STATUS_DONE)
        self.assertIsNotNone(job.output_id)
        self.assertEqual(job.output.kind, ProcessedPDF.KIND_PDFA)
        self.assertTrue(os.path.exists(job.output.path))

    def test_compare_task_records_stats(self):
        from .tasks import run_compare_task

        a = self._make_uploaded_pdf("a.pdf")
        b = self._make_uploaded_pdf("b.pdf")
        job = Job.objects.create(user=self.user, kind=ProcessedPDF.KIND_COMPARE, source=a, second_source=b)
        run_compare_task(str(job.id))
        job.refresh_from_db()
        self.assertEqual(job.status, Job.STATUS_DONE)
        self.assertIn("stats", job.params)
        self.assertEqual(job.params["stats"]["pages_a"], 1)

    def test_task_failure_marks_job_failed_with_message(self):
        from .tasks import run_pdfa_task

        src = self._make_uploaded_pdf()
        # Bad version triggers ValueError inside convert_to_pdfa.
        job = Job.objects.create(
            user=self.user, kind=ProcessedPDF.KIND_PDFA, source=src, params={"version": "9z"}
        )
        run_pdfa_task(str(job.id))
        job.refresh_from_db()
        self.assertEqual(job.status, Job.STATUS_FAILED)
        self.assertIn("PDF/A version", job.error_message)
        self.assertIsNone(job.output_id)

    def test_missing_source_pdf_fails_gracefully(self):
        from .tasks import run_pdfa_task

        # Source row with no file on disk.
        src = UploadedPDF.objects.create(user=self.user, name="ghost.pdf", path="/nope.pdf", size=1)
        job = Job.objects.create(user=self.user, kind=ProcessedPDF.KIND_PDFA, source=src)
        run_pdfa_task(str(job.id))
        job.refresh_from_db()
        self.assertEqual(job.status, Job.STATUS_FAILED)


class JobWebViewTests(_JobTestBase):
    def setUp(self):
        self.user = User.objects.create_user(username="alice", password="pw")
        self.client = Client()
        self.client.force_login(self.user)

    def _upload(self, name="hello.pdf", num_pages=1) -> UploadedPDF:
        f = SimpleUploadedFile(name, _pdf_bytes(num_pages), content_type="application/pdf")
        self.client.post(reverse("upload"), {"pdf_file": f})
        return UploadedPDF.objects.filter(user=self.user).order_by("-uploaded_at").first()

    def test_searchable_view_creates_job_and_redirects(self):
        self._upload()
        resp = self.client.post(reverse("searchable"), {"language": "eng", "dpi": "150"})
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/jobs/", resp.url)
        self.assertTrue(Job.objects.filter(user=self.user, kind=ProcessedPDF.KIND_OCR_LAYER).exists())

    def test_job_status_endpoint_returns_json(self):
        if not shutil.which("gs"):
            self.skipTest("ghostscript not installed")
        self._upload()
        post = self.client.post(reverse("pdfa"), {"version": "2b"})
        job_id = post.url.rstrip("/").split("/")[-1]
        status = self.client.get(reverse("job_status", args=[job_id]))
        self.assertEqual(status.status_code, 200)
        body = status.json()
        # Eager mode: task already ran by the time post returns.
        self.assertIn(body["status"], ("done", "failed"))
        self.assertTrue(body["is_terminal"])

    def test_job_detail_renders(self):
        self._upload()
        post = self.client.post(reverse("searchable"), {"language": "eng", "dpi": "150"})
        job_id = post.url.rstrip("/").split("/")[-1]
        resp = self.client.get(reverse("job_detail", args=[job_id]))
        self.assertEqual(resp.status_code, 200)

    def test_jobs_list_renders(self):
        resp = self.client.get(reverse("jobs_list"))
        self.assertEqual(resp.status_code, 200)

    def test_cannot_see_another_users_job(self):
        other = User.objects.create_user(username="bob", password="pw")
        bob_src = UploadedPDF.objects.create(user=other, name="b.pdf", path="/none", size=1)
        bob_job = Job.objects.create(user=other, kind="pdfa", source=bob_src)
        resp = self.client.get(reverse("job_detail", args=[bob_job.id]))
        self.assertEqual(resp.status_code, 404)


class JobApiTests(_JobTestBase):
    def setUp(self):
        self.user = User.objects.create_user(username="alice", password="pw")
        _, self.token = ApiKey.create_for_user(self.user)
        self.client = APIClient()
        self.client.credentials(HTTP_X_API_KEY=self.token)

    def _upload(self) -> str:
        f = SimpleUploadedFile("hello.pdf", _pdf_bytes(1), content_type="application/pdf")
        resp = self.client.post(reverse("api:pdf-list"), {"pdf_file": f}, format="multipart")
        return resp.json()["id"]

    def test_ocr_endpoint_returns_202_with_job_id(self):
        pdf_id = self._upload()
        resp = self.client.post(
            reverse("api:op-searchable"), {"pdf_id": pdf_id, "language": "eng"}, format="json"
        )
        self.assertEqual(resp.status_code, 202)
        body = resp.json()
        self.assertIn("job_id", body)
        self.assertIn("status_url", body)
        self.assertTrue(Job.objects.filter(user=self.user, id=body["job_id"]).exists())

    def test_jobs_list_endpoint(self):
        pdf_id = self._upload()
        self.client.post(reverse("api:op-searchable"), {"pdf_id": pdf_id, "language": "eng"}, format="json")
        resp = self.client.get(reverse("api:job-list"))
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        rows = body["results"] if isinstance(body, dict) else body
        self.assertGreaterEqual(len(rows), 1)

    def test_pdfa_endpoint_returns_job_with_done_status_in_eager_mode(self):
        if not shutil.which("gs"):
            self.skipTest("ghostscript not installed")
        pdf_id = self._upload()
        resp = self.client.post(reverse("api:op-pdfa"), {"pdf_id": pdf_id, "version": "2b"}, format="json")
        self.assertEqual(resp.status_code, 202)
        job_id = resp.json()["job_id"]
        detail = self.client.get(reverse("api:job-detail", args=[job_id]))
        self.assertEqual(detail.status_code, 200)
        body = detail.json()
        self.assertEqual(body["status"], "done")
        self.assertIsNotNone(body["output_id"])
        self.assertIsNotNone(body["output_download_url"])

    def test_cannot_query_another_users_job_via_api(self):
        other = User.objects.create_user(username="bob", password="pw")
        bob_src = UploadedPDF.objects.create(user=other, name="b.pdf", path="/none", size=1)
        bob_job = Job.objects.create(user=other, kind="pdfa", source=bob_src)
        resp = self.client.get(reverse("api:job-detail", args=[bob_job.id]))
        self.assertEqual(resp.status_code, 404)
