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
            session_key="abc",
            name="old.pdf",
            path=path,
            size=os.path.getsize(path),
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
            session_key="abc",
            name="fresh.pdf",
            path=path,
            size=os.path.getsize(path),
        )

        with self._override_media():
            self._run(hours=24)

        self.assertTrue(os.path.exists(path))
        self.assertTrue(UploadedPDF.objects.filter(pk=recent.pk).exists())

    def test_deletes_old_processed_row(self):
        path = _make_file(self.processed, "out.pdf")
        proc = ProcessedPDF.objects.create(
            session_key="abc",
            kind=ProcessedPDF.KIND_SPLIT,
            name="out.pdf",
            path=path,
            size=os.path.getsize(path),
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
            session_key="abc",
            name="tracked.pdf",
            path=path,
            size=os.path.getsize(path),
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


class SweepOrphanFilesCommandTests(TestCase):
    """sweep_orphan_files is age-agnostic and defaults to dry-run."""

    def setUp(self):
        self.media_root = _media_tmp()
        self.uploads = os.path.join(self.media_root, "uploads")
        self.processed = os.path.join(self.media_root, "processed")
        self.thumbs = os.path.join(self.media_root, "thumbs")
        for d in (self.uploads, self.processed, self.thumbs):
            os.makedirs(d, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.media_root, ignore_errors=True)

    def _override(self):
        return override_settings(MEDIA_ROOT=self.media_root)

    def _run(self, *extra):
        out = StringIO()
        call_command("sweep_orphan_files", *extra, stdout=out)
        return out.getvalue()

    def _make_thumb(self, name):
        path = os.path.join(self.thumbs, name)
        with open(path, "wb") as fh:
            fh.write(b"\xff\xd8\xff\xe0FAKE")  # not a real JPEG but file exists
        return path

    def test_dry_run_keeps_files(self):
        orphan_upload = _make_file(self.uploads, "orphan.pdf")
        with self._override():
            output = self._run()  # no --apply

        self.assertTrue(os.path.exists(orphan_upload))
        self.assertIn("DRY-RUN", output)
        self.assertIn("Would delete", output)
        self.assertIn("Run with --apply", output)

    def test_apply_removes_orphan_uploads(self):
        orphan = _make_file(self.uploads, "orphan.pdf")
        with self._override():
            output = self._run("--apply")

        self.assertFalse(os.path.exists(orphan))
        self.assertIn("APPLY", output)
        self.assertIn("Removed 1 orphan", output)

    def test_apply_keeps_tracked_uploads(self):
        path = _make_file(self.uploads, "tracked.pdf")
        UploadedPDF.objects.create(
            session_key="abc", name="tracked.pdf", path=path, size=os.path.getsize(path)
        )
        with self._override():
            self._run("--apply")
        self.assertTrue(os.path.exists(path))

    def test_apply_removes_orphan_processed(self):
        orphan = _make_file(self.processed, "stale-out.pdf")
        with self._override():
            self._run("--apply")
        self.assertFalse(os.path.exists(orphan))

    def test_apply_removes_orphan_thumbnail(self):
        # Thumb whose UUID stem matches no live UploadedPDF row.
        thumb = self._make_thumb("11111111-1111-1111-1111-111111111111.jpg")
        with self._override():
            self._run("--apply")
        self.assertFalse(os.path.exists(thumb))

    def test_apply_keeps_thumbnail_for_existing_pdf(self):
        path = _make_file(self.uploads, "live.pdf")
        live = UploadedPDF.objects.create(
            session_key="abc", name="live.pdf", path=path, size=os.path.getsize(path)
        )
        thumb = self._make_thumb(f"{live.id}.jpg")
        with self._override():
            self._run("--apply")
        self.assertTrue(os.path.exists(thumb))

    def test_only_restricts_category(self):
        orphan_upload = _make_file(self.uploads, "u.pdf")
        orphan_proc = _make_file(self.processed, "p.pdf")
        with self._override():
            self._run("--apply", "--only", "uploads")
        self.assertFalse(os.path.exists(orphan_upload))
        # processed wasn't in --only → preserved
        self.assertTrue(os.path.exists(orphan_proc))

    def test_skips_non_jpg_files_in_thumbs(self):
        # Anything we don't recognise (e.g. a .gitkeep) must be left alone.
        keep = os.path.join(self.thumbs, ".gitkeep")
        with open(keep, "wb") as fh:
            fh.write(b"")
        with self._override():
            self._run("--apply")
        self.assertTrue(os.path.exists(keep))

    def test_handles_missing_directories(self):
        empty = tempfile.mkdtemp()
        try:
            with override_settings(MEDIA_ROOT=empty):
                output = self._run("--apply")
            self.assertIn("Removed 0", output)
        finally:
            shutil.rmtree(empty, ignore_errors=True)
