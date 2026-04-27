"""Tests for the cleanup_old_pdfs management command."""
import os
import shutil
import tempfile
from datetime import timedelta
from io import StringIO

import fitz
from django.core.management import call_command
from django.test import TestCase, override_settings
from django.utils import timezone

from .models import ProcessedPDF, UploadedPDF


def _media_tmp():
    return tempfile.mkdtemp(prefix="pdfedit_cleanup_")


def _make_file(directory, name="x.pdf"):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    doc = fitz.open()
    doc.new_page()
    doc.save(path)
    doc.close()
    return path


class CleanupCommandTests(TestCase):
    def setUp(self):
        self.media_root = _media_tmp()
        self.uploads = os.path.join(self.media_root, "uploads")
        self.processed = os.path.join(self.media_root, "processed")
        os.makedirs(self.uploads, exist_ok=True)
        os.makedirs(self.processed, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.media_root, ignore_errors=True)

    def _override_media(self):
        return override_settings(MEDIA_ROOT=self.media_root)

    def _run(self, hours=None):
        out = StringIO()
        if hours is None:
            call_command("cleanup_old_pdfs", stdout=out)
        else:
            call_command("cleanup_old_pdfs", "--hours", str(hours), stdout=out)
        return out.getvalue()

    def test_deletes_old_uploaded_pdf_row_and_file(self):
        path = _make_file(self.uploads, "old.pdf")
        old_pdf = UploadedPDF.objects.create(
            session_key="abc", name="old.pdf", path=path, size=os.path.getsize(path),
        )
        UploadedPDF.objects.filter(pk=old_pdf.pk).update(uploaded_at=timezone.now() - timedelta(hours=48))

        with self._override_media():
            output = self._run(hours=24)

        self.assertFalse(os.path.exists(path))
        self.assertFalse(UploadedPDF.objects.filter(pk=old_pdf.pk).exists())
        self.assertIn("Deleted", output)

    def test_keeps_recent_uploaded_pdf(self):
        path = _make_file(self.uploads, "fresh.pdf")
        recent = UploadedPDF.objects.create(
            session_key="abc", name="fresh.pdf", path=path, size=os.path.getsize(path),
        )

        with self._override_media():
            self._run(hours=24)

        self.assertTrue(os.path.exists(path))
        self.assertTrue(UploadedPDF.objects.filter(pk=recent.pk).exists())

    def test_deletes_old_processed_row(self):
        path = _make_file(self.processed, "out.pdf")
        proc = ProcessedPDF.objects.create(
            session_key="abc", kind=ProcessedPDF.KIND_SPLIT,
            name="out.pdf", path=path, size=os.path.getsize(path),
        )
        ProcessedPDF.objects.filter(pk=proc.pk).update(created_at=timezone.now() - timedelta(hours=48))

        with self._override_media():
            self._run(hours=24)

        self.assertFalse(os.path.exists(path))
        self.assertFalse(ProcessedPDF.objects.filter(pk=proc.pk).exists())

    def test_removes_orphan_files_older_than_cutoff(self):
        path = _make_file(self.uploads, "orphan.pdf")
        # Backdate the file's mtime past the cutoff.
        old_ts = (timezone.now() - timedelta(hours=48)).timestamp()
        os.utime(path, (old_ts, old_ts))

        with self._override_media():
            self._run(hours=24)

        self.assertFalse(os.path.exists(path))

    def test_keeps_orphan_files_inside_cutoff(self):
        path = _make_file(self.uploads, "fresh_orphan.pdf")
        # File mtime is "now" — younger than 24h.

        with self._override_media():
            self._run(hours=24)

        self.assertTrue(os.path.exists(path))

    def test_keeps_tracked_files_even_if_old_on_disk(self):
        path = _make_file(self.uploads, "tracked.pdf")
        old_ts = (timezone.now() - timedelta(hours=48)).timestamp()
        os.utime(path, (old_ts, old_ts))
        # Row exists with a fresh uploaded_at — file should be preserved as
        # tracked-but-recent.
        row = UploadedPDF.objects.create(
            session_key="abc", name="tracked.pdf", path=path, size=os.path.getsize(path),
        )

        with self._override_media():
            self._run(hours=24)

        self.assertTrue(os.path.exists(path))
        self.assertTrue(UploadedPDF.objects.filter(pk=row.pk).exists())

    def test_handles_missing_directories(self):
        # Empty MEDIA_ROOT — no uploads/ or processed/ subdirs.
        empty = tempfile.mkdtemp()
        try:
            with override_settings(MEDIA_ROOT=empty):
                output = self._run(hours=24)
            self.assertIn("Deleted 0", output)
        finally:
            shutil.rmtree(empty, ignore_errors=True)

    def test_default_hours_uses_settings(self):
        with self._override_media(), override_settings(PDF_CLEANUP_HOURS=24):
            output = self._run()  # no --hours argument
        self.assertIn("Deleted", output)
