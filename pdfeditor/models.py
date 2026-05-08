"""Database models for PDF Editor.

PDFs are scoped per ``user`` when the requester is authenticated, otherwise
per anonymous Django session via ``session_key``. Both fields exist on the
same row so the same model serves both flows; queries filter by whichever
applies to the current request.
"""

import os
import uuid

from django.conf import settings
from django.db import models


class UploadedPDF(models.Model):
    """A PDF a user has uploaded — owned by an authenticated user OR by an anonymous session."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="uploaded_pdfs",
    )
    session_key = models.CharField(max_length=64, db_index=True, blank=True)
    name = models.CharField(max_length=255)
    path = models.CharField(max_length=500)
    size = models.BigIntegerField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-uploaded_at"]
        indexes = [
            models.Index(fields=["user", "-uploaded_at"]),
        ]

    def __str__(self):
        owner = self.user.username if self.user_id else f"anon:{self.session_key[:8]}"
        return f"{self.name} ({owner})"

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
    KIND_PROTECT = "protect"
    KIND_SIGN = "sign"
    KIND_CONVERT = "convert"
    KIND_REORDER = "reorder"
    KIND_TO_IMAGES = "to_images"
    KIND_IMAGES_TO_PDF = "images_to_pdf"
    KIND_METADATA = "metadata"

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
        (KIND_PROTECT, "Password Protect"),
        (KIND_SIGN, "Digital Signature"),
        (KIND_CONVERT, "Convert to Word"),
        (KIND_REORDER, "Reorder/Delete Pages"),
        (KIND_TO_IMAGES, "PDF to Images"),
        (KIND_IMAGES_TO_PDF, "Images to PDF"),
        (KIND_METADATA, "Edit Metadata"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="processed_pdfs",
    )
    session_key = models.CharField(max_length=64, db_index=True, blank=True)
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
        indexes = [
            models.Index(fields=["session_key", "kind"]),
            models.Index(fields=["user", "-created_at"]),
        ]

    def __str__(self):
        owner = self.user.username if self.user_id else f"anon:{self.session_key[:8]}"
        return f"{self.get_kind_display()} · {self.name} ({owner})"

    def exists_on_disk(self) -> bool:
        return bool(self.path) and os.path.exists(self.path)


class TrustAnchor(models.Model):
    """A globally-trusted X.509 certificate used by the signature verifier.

    Anchors are merged with any per-request anchors uploaded at verify time.
    Managed via the Django admin (staff only).
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, help_text="Friendly label for this anchor.")
    cert_pem = models.TextField(help_text="PEM-encoded X.509 certificate.")
    is_active = models.BooleanField(default=True)
    added_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="trust_anchors",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        status = "active" if self.is_active else "disabled"
        return f"{self.name} ({status})"
