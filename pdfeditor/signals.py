"""Filesystem cleanup signals.

`cleanup_old_pdfs` is the catch-all sweep that runs on a cron and removes
files older than the retention window. These signals are the *fast path* —
when a row is deleted in real time (user action, cascade from User delete,
admin delete), we delete the matching file/thumbnail immediately so the
filesystem never drifts ahead of the database.
"""

from __future__ import annotations

import logging
import os

from django.db.models.signals import post_delete
from django.dispatch import receiver

from .models import ProcessedPDF, UploadedPDF

logger = logging.getLogger(__name__)


def _unlink(path: str | None, label: str) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Failed to remove %s %s: %s", label, path, exc)


@receiver(post_delete, sender=UploadedPDF)
def _on_uploaded_pdf_delete(sender, instance: UploadedPDF, **_):
    _unlink(instance.path, "uploaded PDF")
    # Local import avoids circular: views.upload imports from models.
    from .views.upload import _thumbnail_path

    _unlink(_thumbnail_path(instance.id), "thumbnail")


@receiver(post_delete, sender=ProcessedPDF)
def _on_processed_pdf_delete(sender, instance: ProcessedPDF, **_):
    _unlink(instance.path, "processed PDF")
