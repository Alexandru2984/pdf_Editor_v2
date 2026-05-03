"""Delete PDF files and DB rows older than PDF_CLEANUP_HOURS."""

import os
from datetime import timedelta

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from pdfeditor.models import ProcessedPDF, UploadedPDF


class Command(BaseCommand):
    help = "Delete PDF files (and matching DB rows) older than PDF_CLEANUP_HOURS."

    def add_arguments(self, parser):
        parser.add_argument(
            "--hours",
            type=int,
            default=None,
            help="Retention window in hours (default: settings.PDF_CLEANUP_HOURS).",
        )

    def handle(self, *args: object, **options: object) -> None:
        opt_hours = options.get("hours")
        if isinstance(opt_hours, int):
            cleanup_hours = opt_hours
        else:
            cleanup_hours = int(getattr(settings, "PDF_CLEANUP_HOURS", 24))
        cleanup_threshold = timezone.now() - timedelta(hours=cleanup_hours)

        rows_deleted = 0
        rows_deleted += self._delete_rows(
            UploadedPDF.objects.filter(uploaded_at__lt=cleanup_threshold),
            "upload",
        )
        rows_deleted += self._delete_rows(
            ProcessedPDF.objects.filter(created_at__lt=cleanup_threshold),
            "output",
        )

        orphans_deleted, orphans_bytes = self._delete_orphaned_files(cleanup_threshold)
        thumbs_deleted = self._delete_orphan_thumbnails()
        if thumbs_deleted:
            orphans_deleted += thumbs_deleted

        summary = (
            f"Deleted {rows_deleted} DB rows, "
            f"{orphans_deleted} orphaned files ({orphans_bytes / 1024 / 1024:.2f} MB)."
        )
        self.stdout.write(self.style.SUCCESS(summary))

    def _delete_rows(self, queryset, label: str) -> int:
        count = 0
        for row in queryset:
            if row.path and os.path.exists(row.path):
                try:
                    os.remove(row.path)
                except OSError as e:
                    self.stdout.write(self.style.WARNING(f'{label} "{row.path}": {e}'))
            row.delete()
            count += 1
        return count

    def _delete_orphan_thumbnails(self) -> int:
        """Remove thumbnails whose UploadedPDF row no longer exists."""
        from pdfeditor.views.upload import THUMB_SUBDIR

        thumbs_dir = os.path.join(settings.MEDIA_ROOT, THUMB_SUBDIR)
        if not os.path.isdir(thumbs_dir):
            return 0

        live_ids = {str(pid) for pid in UploadedPDF.objects.values_list("id", flat=True)}
        removed = 0
        for entry in os.listdir(thumbs_dir):
            stem, ext = os.path.splitext(entry)
            if ext.lower() != ".jpg" or stem in live_ids:
                continue
            try:
                os.remove(os.path.join(thumbs_dir, entry))
                removed += 1
            except OSError as e:
                self.stdout.write(self.style.WARNING(f'thumb "{entry}": {e}'))
        return removed

    def _delete_orphaned_files(self, cleanup_threshold):
        """Remove files on disk that no row references and are older than threshold."""
        tracked: set[str] = set()
        tracked.update(os.path.realpath(p) for p in UploadedPDF.objects.values_list("path", flat=True) if p)
        tracked.update(os.path.realpath(p) for p in ProcessedPDF.objects.values_list("path", flat=True) if p)

        cutoff_ts = cleanup_threshold.timestamp()
        deleted = 0
        deleted_bytes = 0

        for sub in ("uploads", "processed"):
            directory = os.path.join(settings.MEDIA_ROOT, sub)
            if not os.path.isdir(directory):
                continue
            for entry in os.listdir(directory):
                filepath = os.path.realpath(os.path.join(directory, entry))
                if not os.path.isfile(filepath):
                    continue
                if filepath in tracked:
                    continue
                if os.path.getmtime(filepath) >= cutoff_ts:
                    continue
                try:
                    size = os.path.getsize(filepath)
                    os.remove(filepath)
                    deleted += 1
                    deleted_bytes += size
                except OSError as e:
                    self.stdout.write(self.style.WARNING(f'orphan "{filepath}": {e}'))

        return deleted, deleted_bytes
