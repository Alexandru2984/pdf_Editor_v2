"""Database models for PDF Editor.

PDFs are scoped per anonymous Django session via ``session_key``. There is no
user authentication; isolation comes from matching the requesting client's
session cookie against the stored key. File ownership is not transferable
across sessions.
"""

import os
import uuid

from django.db import models


class UploadedPDF(models.Model):
    """A PDF a user has uploaded in the current session."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session_key = models.CharField(max_length=64, db_index=True)
    name = models.CharField(max_length=255)
    path = models.CharField(max_length=500)
    size = models.BigIntegerField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return f"{self.name} ({self.session_key[:8]}…)"

    def exists_on_disk(self) -> bool:
        return bool(self.path) and os.path.exists(self.path)


class ProcessedPDF(models.Model):
    """A PDF produced by one of the editing operations."""

    KIND_FIND_REPLACE = "find_replace"
    KIND_SPLIT = "split"
    KIND_MERGE = "merge"
    KIND_COMPRESS = "compress"
    KIND_WATERMARK = "watermark"
    KIND_ROTATE = "rotate"
    KIND_PAGE_NUMBERS = "page_numbers"
    KIND_REPHRASE = "rephrase"
    KIND_FORM_FILL = "form_fill"

    KIND_CHOICES = [
        (KIND_FIND_REPLACE, "Find & Replace"),
        (KIND_SPLIT, "Split"),
        (KIND_MERGE, "Merge"),
        (KIND_COMPRESS, "Compress"),
        (KIND_WATERMARK, "Watermark"),
        (KIND_ROTATE, "Rotate"),
        (KIND_PAGE_NUMBERS, "Page Numbers"),
        (KIND_REPHRASE, "AI Rephrase"),
        (KIND_FORM_FILL, "Form Fill"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session_key = models.CharField(max_length=64, db_index=True)
    kind = models.CharField(max_length=20, choices=KIND_CHOICES)
    source = models.ForeignKey(
        UploadedPDF,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="outputs",
    )
    name = models.CharField(max_length=255)
    path = models.CharField(max_length=500)
    size = models.BigIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["session_key", "kind"])]

    def __str__(self):
        return f"{self.get_kind_display()} · {self.name}"

    def exists_on_disk(self) -> bool:
        return bool(self.path) and os.path.exists(self.path)
