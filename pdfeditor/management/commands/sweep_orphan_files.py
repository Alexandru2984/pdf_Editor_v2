"""Find and (optionally) delete media files with no matching DB row.

This is the close-out command for the orphan-file problem documented in
``scripts/loadtest/benchmarks.md``: the ``post_delete`` signal in
``pdfeditor/signals.py`` now keeps the filesystem in lockstep with the
database for newly-deleted rows, but anything orphaned **before** the
signal landed is still sitting on disk.

Unlike ``cleanup_old_pdfs`` (which is a retention sweep — age threshold
+ deletes DB rows + their files), this command:

* Touches **only files**, never DB rows.
* Has **no age cutoff** — every orphan is fair game.
* **Dry-runs by default**. Run ``--apply`` to actually delete.

Use it once after the signal lands to close out legacy orphans; you can
also re-run it periodically as a paranoia check.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from django.conf import settings
from django.core.management.base import BaseCommand

from pdfeditor.models import ProcessedPDF, UploadedPDF
from pdfeditor.views.upload import THUMB_SUBDIR

_CATEGORIES = ("uploads", "processed", "thumbs")


class Command(BaseCommand):
    help = "Find (and optionally delete) media files that no DB row references."

    def add_arguments(self, parser):
        parser.add_argument(
            "--apply",
            action="store_true",
            help="Actually delete the orphans. Default is a dry-run report only.",
        )
        parser.add_argument(
            "--only",
            choices=_CATEGORIES,
            action="append",
            help=(
                "Restrict to one or more categories: uploads / processed / "
                "thumbs. Repeatable. Default: all three."
            ),
        )

    def handle(self, *args: object, **options: object) -> None:
        apply: bool = bool(options.get("apply"))
        only: Iterable[str] = options.get("only") or list(_CATEGORIES)  # type: ignore[assignment]

        verb = "Deleting" if apply else "Would delete"
        mode_label = "APPLY" if apply else "DRY-RUN"
        self.stdout.write(self.style.NOTICE(f"sweep_orphan_files [{mode_label}]"))

        total_files = 0
        total_bytes = 0
        for category in only:
            n_files, n_bytes = self._sweep_category(category, apply=apply, verb=verb)
            total_files += n_files
            total_bytes += n_bytes

        summary = (
            f"{'Removed' if apply else 'Found'} {total_files} orphan files "
            f"({total_bytes / 1024 / 1024:.2f} MB)."
        )
        if apply:
            self.stdout.write(self.style.SUCCESS(summary))
        else:
            self.stdout.write(self.style.WARNING(summary + "  Run with --apply to delete."))

    # --- per-category sweeps ------------------------------------------------

    def _sweep_category(self, category: str, *, apply: bool, verb: str) -> tuple[int, int]:
        if category == "uploads":
            return self._sweep_dir(
                os.path.join(settings.MEDIA_ROOT, "uploads"),
                self._live_upload_paths(),
                category,
                apply=apply,
                verb=verb,
            )
        if category == "processed":
            return self._sweep_dir(
                os.path.join(settings.MEDIA_ROOT, "processed"),
                self._live_processed_paths(),
                category,
                apply=apply,
                verb=verb,
            )
        if category == "thumbs":
            return self._sweep_thumbs(apply=apply, verb=verb)
        # Argparse's choices=... gates this, but be explicit.
        raise ValueError(f"Unknown category: {category}")

    @staticmethod
    def _live_upload_paths() -> set[str]:
        return {os.path.realpath(p) for p in UploadedPDF.objects.values_list("path", flat=True) if p}

    @staticmethod
    def _live_processed_paths() -> set[str]:
        return {os.path.realpath(p) for p in ProcessedPDF.objects.values_list("path", flat=True) if p}

    def _sweep_dir(
        self,
        directory: str,
        tracked: Iterable[str],
        label: str,
        *,
        apply: bool,
        verb: str,
    ) -> tuple[int, int]:
        if not os.path.isdir(directory):
            return 0, 0
        tracked_set = set(tracked)
        removed, removed_bytes = 0, 0
        for entry in os.listdir(directory):
            filepath = os.path.realpath(os.path.join(directory, entry))
            if not os.path.isfile(filepath):
                continue
            if filepath in tracked_set:
                continue
            try:
                size = os.path.getsize(filepath)
            except OSError:
                continue
            self.stdout.write(f"  [{label}] {verb}: {filepath}  ({size / 1024:.1f} KB)")
            if apply:
                try:
                    os.remove(filepath)
                except OSError as exc:
                    self.stdout.write(self.style.WARNING(f"    failed: {exc}"))
                    continue
            removed += 1
            removed_bytes += size
        return removed, removed_bytes

    def _sweep_thumbs(self, *, apply: bool, verb: str) -> tuple[int, int]:
        directory = os.path.join(settings.MEDIA_ROOT, THUMB_SUBDIR)
        if not os.path.isdir(directory):
            return 0, 0
        live_ids = {str(pid) for pid in UploadedPDF.objects.values_list("id", flat=True)}
        removed, removed_bytes = 0, 0
        for entry in os.listdir(directory):
            stem, ext = os.path.splitext(entry)
            if ext.lower() != ".jpg":
                # Unknown shape — skip rather than risk deleting something
                # legitimate that we don't recognise.
                continue
            if stem in live_ids:
                continue
            filepath = os.path.join(directory, entry)
            try:
                size = os.path.getsize(filepath)
            except OSError:
                continue
            self.stdout.write(f"  [thumbs] {verb}: {filepath}  ({size / 1024:.1f} KB)")
            if apply:
                try:
                    os.remove(filepath)
                except OSError as exc:
                    self.stdout.write(self.style.WARNING(f"    failed: {exc}"))
                    continue
            removed += 1
            removed_bytes += size
        return removed, removed_bytes
