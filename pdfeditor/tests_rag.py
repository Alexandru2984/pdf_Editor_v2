"""Tests for the RAG chat-with-PDF feature (chunking, indexing, retrieval).

These tests require Postgres + pgvector — they are skipped on SQLite (the
Embedding model uses a vector(384) column that SQLite can't store).
"""

import io
import json
import os
import shutil
import tempfile
from unittest.mock import patch

import fitz
from django.contrib.auth import get_user_model
from django.db import connection
from django.test import TestCase, override_settings
from django.urls import reverse
from rest_framework.test import APIClient

from .models import ApiKey, Embedding, Job, ProcessedPDF, UploadedPDF

User = get_user_model()
_HAS_PGVECTOR = connection.vendor == "postgresql"


def _pdf_bytes(pages_text: list[str]) -> bytes:
    doc = fitz.open()
    for txt in pages_text:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), txt, fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


_MEDIA_ROOT = tempfile.mkdtemp(prefix="pdfeditor_rag_test_")


def _fake_embed_texts(texts):
    """Stand-in for fastembed: returns a stable, content-derived vector per
    text. Same text → same vector; similar text → similar vectors (we hash
    each char into one of 384 buckets)."""
    out = []
    for text in texts:
        vec = [0.0] * 384
        for ch in text.lower():
            vec[ord(ch) % 384] += 1.0
        # Normalise so cosine distance works.
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        out.append([v / norm for v in vec])
    return out


def _fake_embed_query(text):
    return _fake_embed_texts([text])[0]


@override_settings(MEDIA_ROOT=_MEDIA_ROOT)
class _RagTestBase(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.makedirs(os.path.join(_MEDIA_ROOT, "uploads"), exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(_MEDIA_ROOT, ignore_errors=True)
        super().tearDownClass()

    def _make_pdf(self, name="src.pdf", pages_text=None) -> UploadedPDF:
        pages_text = pages_text or ["Hello world from page one"]
        path = os.path.join(_MEDIA_ROOT, "uploads", name)
        with open(path, "wb") as f:
            f.write(_pdf_bytes(pages_text))
        return UploadedPDF.objects.create(user=self.user, name=name, path=path, size=os.path.getsize(path))


class ChunkingTests(TestCase):
    """Pure-function tests — no DB needed."""

    def test_chunks_skip_empty_pages(self):
        from .pdf_processor.rag import chunk_pdf

        doc = fitz.open()
        doc.new_page()  # blank
        page2 = doc.new_page(width=595, height=842)
        page2.insert_text((72, 100), "Some content on page 2", fontsize=12)
        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        doc.save(path)
        doc.close()
        try:
            chunks = list(chunk_pdf(path))
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0][1], 2)  # page number
            self.assertIn("page 2", chunks[0][2])
        finally:
            os.remove(path)

    def test_missing_file_raises(self):
        from .pdf_processor.rag import chunk_pdf

        with self.assertRaises(ValueError):
            list(chunk_pdf("/no/such/file.pdf"))


class IndexTaskTests(_RagTestBase):
    def setUp(self):
        if not _HAS_PGVECTOR:
            self.skipTest("requires Postgres + pgvector")
        self.user = User.objects.create_user(username="alice", password="pw")

    @patch("pdfeditor.pdf_processor.rag.embed_texts", side_effect=_fake_embed_texts)
    def test_indexing_creates_embeddings_and_marks_job_done(self, _mock):
        from .tasks import run_chat_index_task

        src = self._make_pdf(pages_text=["Page 1 talks about cats and dogs", "Page 2 talks about birds"])
        job = Job.objects.create(user=self.user, kind=ProcessedPDF.KIND_CHAT_INDEX, source=src)
        run_chat_index_task(str(job.id))
        job.refresh_from_db()
        self.assertEqual(job.status, Job.STATUS_DONE)
        self.assertEqual(job.progress, 100)
        self.assertEqual(Embedding.objects.filter(uploaded_pdf=src).count(), 2)
        self.assertIn("chunk_count", job.params)

    @patch("pdfeditor.pdf_processor.rag.embed_texts", side_effect=_fake_embed_texts)
    def test_reindexing_replaces_old_embeddings(self, _mock):
        from .tasks import run_chat_index_task

        src = self._make_pdf(pages_text=["First text"])
        # Pre-existing stale embedding for the same PDF.
        Embedding.objects.create(
            uploaded_pdf=src, chunk_index=99, page_number=1, chunk_text="stale", embedding=[0.0] * 384
        )
        job = Job.objects.create(user=self.user, kind=ProcessedPDF.KIND_CHAT_INDEX, source=src)
        run_chat_index_task(str(job.id))
        # The stale row must be gone, replaced by the fresh chunk.
        kept = list(Embedding.objects.filter(uploaded_pdf=src))
        self.assertEqual(len(kept), 1)
        self.assertNotEqual(kept[0].chunk_index, 99)

    def test_empty_pdf_marks_job_failed(self):
        from .tasks import run_chat_index_task

        # PDF with one blank page only — chunk_pdf yields nothing.
        doc = fitz.open()
        doc.new_page()
        path = os.path.join(_MEDIA_ROOT, "uploads", "blank.pdf")
        doc.save(path)
        doc.close()
        src = UploadedPDF.objects.create(
            user=self.user, name="blank.pdf", path=path, size=os.path.getsize(path)
        )

        job = Job.objects.create(user=self.user, kind=ProcessedPDF.KIND_CHAT_INDEX, source=src)
        run_chat_index_task(str(job.id))
        job.refresh_from_db()
        self.assertEqual(job.status, Job.STATUS_FAILED)
        self.assertIn("OCR", job.error_message)


class ChatViewTests(_RagTestBase):
    def setUp(self):
        if not _HAS_PGVECTOR:
            self.skipTest("requires Postgres + pgvector")
        self.user = User.objects.create_user(username="alice", password="pw")
        self.client.force_login(self.user)

    def test_chat_redirects_to_index_job_when_no_embeddings(self):
        pdf = self._make_pdf()
        resp = self.client.get(reverse("chat", args=[pdf.id]))
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/jobs/", resp.url)
        self.assertTrue(
            Job.objects.filter(user=self.user, kind=ProcessedPDF.KIND_CHAT_INDEX, source=pdf).exists()
        )

    def test_chat_renders_when_indexed(self):
        pdf = self._make_pdf()
        Embedding.objects.create(
            uploaded_pdf=pdf, chunk_index=0, page_number=1, chunk_text="text", embedding=[0.1] * 384
        )
        resp = self.client.get(reverse("chat", args=[pdf.id]))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Chat with PDF")

    @patch("pdfeditor.pdf_processor.rag.embed_query", side_effect=_fake_embed_query)
    @patch("pdfeditor.views.chat._call_groq", return_value=("Cats are mentioned on page 1 [1].", None))
    def test_chat_message_returns_answer_and_citations(self, _groq, _embed):
        pdf = self._make_pdf()
        Embedding.objects.create(
            uploaded_pdf=pdf, chunk_index=0, page_number=1, chunk_text="cats on page", embedding=[0.1] * 384
        )
        resp = self.client.post(
            reverse("chat_message", args=[pdf.id]),
            data=json.dumps({"message": "What animals appear?"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("answer", body)
        self.assertIn("citations", body)
        self.assertEqual(len(body["citations"]), 1)
        self.assertEqual(body["citations"][0]["page"], 1)

    def test_chat_message_rejects_empty_input(self):
        pdf = self._make_pdf()
        Embedding.objects.create(
            uploaded_pdf=pdf, chunk_index=0, page_number=1, chunk_text="x", embedding=[0.1] * 384
        )
        resp = self.client.post(
            reverse("chat_message", args=[pdf.id]),
            data=json.dumps({"message": "   "}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_chat_message_returns_409_when_not_indexed(self):
        pdf = self._make_pdf()
        resp = self.client.post(
            reverse("chat_message", args=[pdf.id]),
            data=json.dumps({"message": "hello"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 409)

    def test_cannot_chat_with_another_users_pdf(self):
        other = User.objects.create_user(username="bob", password="pw")
        bob_pdf = UploadedPDF.objects.create(user=other, name="b.pdf", path="/none", size=1)
        resp = self.client.get(reverse("chat", args=[bob_pdf.id]))
        self.assertEqual(resp.status_code, 302)
        # Redirects to dashboard with error, not to a job page.
        self.assertEqual(resp.url, reverse("dashboard"))


class ChatApiTests(_RagTestBase):
    def setUp(self):
        if not _HAS_PGVECTOR:
            self.skipTest("requires Postgres + pgvector")
        self.user = User.objects.create_user(username="alice", password="pw")
        _, self.token = ApiKey.create_for_user(self.user)
        self.client = APIClient()
        self.client.credentials(HTTP_X_API_KEY=self.token)

    @patch("pdfeditor.pdf_processor.rag.embed_query", side_effect=_fake_embed_query)
    @patch("pdfeditor.views.chat._call_groq", return_value=("Answer with [1].", None))
    def test_api_chat_returns_answer(self, _groq, _embed):
        pdf = self._make_pdf()
        Embedding.objects.create(
            uploaded_pdf=pdf, chunk_index=0, page_number=2, chunk_text="content", embedding=[0.1] * 384
        )
        resp = self.client.post(
            reverse("api:op-chat"),
            {"pdf_id": str(pdf.id), "message": "hi"},
            format="json",
        )
        self.assertEqual(resp.status_code, 200, resp.content)
        body = resp.json()
        self.assertIn("answer", body)
        self.assertEqual(body["citations"][0]["page"], 2)

    def test_api_chat_returns_409_when_not_indexed(self):
        pdf = self._make_pdf()
        resp = self.client.post(
            reverse("api:op-chat"),
            {"pdf_id": str(pdf.id), "message": "hi"},
            format="json",
        )
        self.assertEqual(resp.status_code, 409)
