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
from pgvector.django import HnswIndex, VectorField

# Dimensionality of the multilingual MiniLM model we use (384 floats).
# Keep aligned with EMBEDDING_MODEL in pdf_processor.rag — changing this
# requires a new migration that rebuilds the column + HNSW index.
EMBEDDING_DIM = 384


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
    KIND_UNPROTECT = "unprotect"
    KIND_CROP = "crop"
    KIND_FLATTEN = "flatten"
    KIND_REDACT = "redact"
    KIND_OCR_LAYER = "ocr_layer"
    KIND_PDFA = "pdfa"
    KIND_COMPARE = "compare"
    KIND_OUTLINE = "outline"
    KIND_CHAT_INDEX = "chat_index"

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
        (KIND_UNPROTECT, "Remove Password"),
        (KIND_CROP, "Crop Pages"),
        (KIND_FLATTEN, "Flatten"),
        (KIND_REDACT, "Redact"),
        (KIND_OCR_LAYER, "OCR Layer"),
        (KIND_PDFA, "PDF/A"),
        (KIND_COMPARE, "Compare"),
        (KIND_OUTLINE, "Edit Outline"),
        (KIND_CHAT_INDEX, "Index for Chat"),
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


class AuditLog(models.Model):
    """Immutable record of a single PDF operation — who, what, when, from where."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="audit_entries",
    )
    session_key = models.CharField(max_length=64, db_index=True, blank=True)
    kind = models.CharField(max_length=20, db_index=True)
    source_name = models.CharField(max_length=255, blank=True)
    output_name = models.CharField(max_length=255, blank=True)
    output_size = models.BigIntegerField(default=0)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.CharField(max_length=300, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "-created_at"]),
            models.Index(fields=["kind", "-created_at"]),
        ]

    def __str__(self):
        owner = self.user.username if self.user_id else f"anon:{self.session_key[:8] or '?'}"
        return f"{self.kind} · {owner} · {self.created_at:%Y-%m-%d %H:%M}"


class Job(models.Model):
    """Tracks an async PDF operation run by a Celery worker.

    The view creates the row + enqueues a task. The task updates ``status``
    as it progresses and links ``output`` once a ProcessedPDF row exists. On
    failure, ``error_message`` carries the user-friendly summary.
    """

    STATUS_QUEUED = "queued"
    STATUS_RUNNING = "running"
    STATUS_DONE = "done"
    STATUS_FAILED = "failed"
    STATUS_CHOICES = [
        (STATUS_QUEUED, "Queued"),
        (STATUS_RUNNING, "Running"),
        (STATUS_DONE, "Done"),
        (STATUS_FAILED, "Failed"),
    ]
    TERMINAL_STATUSES = (STATUS_DONE, STATUS_FAILED)

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="jobs",
    )
    session_key = models.CharField(max_length=64, db_index=True, blank=True)
    kind = models.CharField(max_length=20, db_index=True)
    status = models.CharField(max_length=12, choices=STATUS_CHOICES, default=STATUS_QUEUED, db_index=True)
    source = models.ForeignKey(
        "UploadedPDF",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="jobs",
    )
    second_source = models.ForeignKey(
        "UploadedPDF",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="jobs_as_second",
        help_text="Optional second input (e.g. compare's revised PDF).",
    )
    output = models.ForeignKey(
        "ProcessedPDF",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="jobs",
    )
    params = models.JSONField(default=dict, blank=True)
    progress = models.PositiveSmallIntegerField(default=0)
    error_message = models.CharField(max_length=500, blank=True)
    # Celery's AsyncResult.id; populated by enqueue_job. Stored so the
    # cancel endpoint can call ``app.control.revoke`` on the right task.
    # Blank for jobs created before the field existed.
    celery_task_id = models.CharField(max_length=64, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "-created_at"]),
            models.Index(fields=["session_key", "-created_at"]),
        ]

    def is_terminal(self) -> bool:
        return self.status in self.TERMINAL_STATUSES

    def __str__(self):
        owner = self.user.username if self.user_id else f"anon:{self.session_key[:8] or '?'}"
        return f"{self.kind} · {self.status} · {owner}"


class Embedding(models.Model):
    """One chunk of a PDF with its dense vector embedding, for RAG retrieval.

    Chunks are deleted en masse when the source PDF goes away. The HNSW
    index uses cosine distance — match it in queries with
    ``order_by(CosineDistance(...))``.
    """

    id = models.BigAutoField(primary_key=True)
    uploaded_pdf = models.ForeignKey(
        "UploadedPDF",
        on_delete=models.CASCADE,
        related_name="embeddings",
    )
    chunk_index = models.PositiveIntegerField()
    page_number = models.PositiveSmallIntegerField()
    chunk_text = models.TextField()
    embedding = VectorField(dimensions=EMBEDDING_DIM)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["uploaded_pdf_id", "chunk_index"]
        indexes = [
            models.Index(fields=["uploaded_pdf", "chunk_index"]),
            HnswIndex(
                name="embedding_hnsw_cos",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
        ]

    def __str__(self):
        return f"chunk {self.chunk_index} p{self.page_number} of {self.uploaded_pdf_id}"


def _default_mfa_secret() -> str:
    import pyotp

    return pyotp.random_base32()


class MfaDevice(models.Model):
    """A user's TOTP authenticator. One per user; ``confirmed`` flips true
    only after the user proves they can generate a valid code, so a
    half-finished enrolment never gates login."""

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="mfa_device",
    )
    secret = models.CharField(max_length=64, default=_default_mfa_secret)
    confirmed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    confirmed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"MFA({self.user.username}, {'on' if self.confirmed else 'pending'})"


class MfaBackupCode(models.Model):
    """A single-use recovery code (stored only as a SHA-256 hash)."""

    device = models.ForeignKey(
        MfaDevice,
        on_delete=models.CASCADE,
        related_name="backup_codes",
    )
    code_hash = models.CharField(max_length=64, db_index=True)
    used_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        state = "used" if self.used_at else "unused"
        return f"backup:{self.code_hash[:8]} ({state})"


def _default_share_token() -> str:
    return uuid.uuid4().hex


def _default_api_key_token() -> str:
    import secrets

    return secrets.token_urlsafe(32)


class ApiKey(models.Model):
    """Per-user API key for programmatic access to the REST API.

    The plaintext token is shown ONCE on creation; only its SHA-256 hash is
    stored. Revoked keys are kept for audit but cannot authenticate. Each
    key is scoped to one user — the API never authenticates anonymous
    sessions.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="api_keys",
    )
    label = models.CharField(max_length=80, blank=True, help_text="Friendly name for this key.")
    key_hash = models.CharField(max_length=64, unique=True, db_index=True)
    prefix = models.CharField(max_length=12, db_index=True, help_text="First chars of the token for display.")
    last_used_at = models.DateTimeField(null=True, blank=True)
    revoked_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["user", "-created_at"])]

    def is_active(self) -> bool:
        return self.revoked_at is None

    def __str__(self):
        state = "active" if self.is_active() else "revoked"
        return f"{self.label or self.prefix} · {self.user.username} ({state})"

    @staticmethod
    def hash_token(token: str) -> str:
        import hashlib

        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @classmethod
    def create_for_user(cls, user, label: str = "") -> tuple["ApiKey", str]:
        """Create a new key. Returns (instance, plaintext_token). The plaintext
        is only returned here — it's not stored, only its hash."""
        token = _default_api_key_token()
        return (
            cls.objects.create(
                user=user,
                label=label or "",
                key_hash=cls.hash_token(token),
                prefix=token[:8],
            ),
            token,
        )


class ShareLink(models.Model):
    """A public, token-protected download link for a ProcessedPDF.

    Whoever has the link can download the underlying PDF until either the
    expiry passes or the download counter exceeds ``max_downloads``. Owner
    can revoke at any time by deleting the row.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    processed_pdf = models.ForeignKey(
        "ProcessedPDF",
        on_delete=models.CASCADE,
        related_name="share_links",
    )
    creator = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="share_links",
    )
    session_key = models.CharField(max_length=64, blank=True, db_index=True)
    token = models.CharField(max_length=64, unique=True, default=_default_share_token, db_index=True)
    expires_at = models.DateTimeField()
    max_downloads = models.PositiveIntegerField(default=0)
    download_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["creator", "-created_at"]),
            models.Index(fields=["session_key", "-created_at"]),
        ]

    def is_expired(self) -> bool:
        from django.utils import timezone

        return timezone.now() >= self.expires_at

    def is_exhausted(self) -> bool:
        return self.max_downloads > 0 and self.download_count >= self.max_downloads

    def is_usable(self) -> bool:
        return not self.is_expired() and not self.is_exhausted()

    def __str__(self):
        return f"share:{self.token[:8]} → {self.processed_pdf_id}"
