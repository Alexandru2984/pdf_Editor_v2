"""Cloudflare R2 (S3-compatible) mirror for processed outputs.

When enabled, every :class:`~pdfeditor.models.ProcessedPDF` is uploaded to the
R2 bucket by a Celery task right after creation, and downloads are served as
short-lived presigned URLs straight from R2 — offloading download bandwidth
from the app containers and keeping outputs durable even if the local media
volume is lost.

Local disk stays the source of truth and the fallback: every function here is
best-effort and returns ``None``/no-ops on failure or when R2 is not
configured, so the app degrades to local ``FileResponse`` serving.

R2 specifics vs vanilla S3: the endpoint is per-account
(``https://<account_id>.r2.cloudflarestorage.com``), the region must be the
literal ``"auto"``, and only signature v4 is supported.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from django.conf import settings

logger = logging.getLogger(__name__)


def enabled() -> bool:
    """True iff R2 is switched on AND fully configured."""
    return bool(
        getattr(settings, "R2_ENABLED", False)
        and getattr(settings, "R2_BUCKET", "")
        and getattr(settings, "R2_ENDPOINT_URL", "")
        and getattr(settings, "R2_ACCESS_KEY_ID", "")
        and getattr(settings, "R2_SECRET_ACCESS_KEY", "")
    )


@lru_cache(maxsize=1)
def _client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=settings.R2_ENDPOINT_URL,
        aws_access_key_id=settings.R2_ACCESS_KEY_ID,
        aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
    )


def object_key(processed) -> str:
    """Bucket key for a ProcessedPDF. The UUID pk keeps keys collision-free
    and unguessable; the original name keeps the bucket browsable."""
    return f"processed/{processed.id}/{processed.name}"


def mirror_processed(processed) -> str | None:
    """Upload ``processed``'s file to R2. Returns the object key, or None."""
    if not enabled():
        return None
    key = object_key(processed)
    try:
        _client().upload_file(
            processed.path,
            settings.R2_BUCKET,
            key,
            ExtraArgs={"ContentType": "application/pdf"},
        )
    except Exception as exc:  # noqa: BLE001 — mirroring must never break the op
        logger.warning("R2 mirror failed for %s: %s", processed.id, exc)
        return None
    return key


def presigned_download_url(processed) -> str | None:
    """Short-lived GET URL for a mirrored output, or None to fall back to disk."""
    if not (enabled() and getattr(processed, "r2_key", None)):
        return None
    # The filename lands inside a quoted Content-Disposition value — strip
    # quote/control characters so a crafted name can't break the header.
    safe_name = "".join(c for c in processed.name if c.isprintable() and c not in '";\\') or "output.pdf"
    try:
        return _client().generate_presigned_url(
            "get_object",
            Params={
                "Bucket": settings.R2_BUCKET,
                "Key": processed.r2_key,
                "ResponseContentDisposition": f'attachment; filename="{safe_name}"',
                "ResponseContentType": "application/pdf",
            },
            ExpiresIn=int(getattr(settings, "R2_PRESIGN_TTL", 300)),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("R2 presign failed for %s: %s", processed.id, exc)
        return None


def schedule_mirror(processed) -> None:
    """Queue the Celery mirror task for a fresh output. Never raises."""
    if not enabled():
        return
    try:
        from .tasks import mirror_output_to_r2

        mirror_output_to_r2.delay(str(processed.id))
    except Exception as exc:  # noqa: BLE001 — a dead broker must not fail the op
        logger.warning("R2 mirror enqueue failed for %s: %s", processed.id, exc)


def delete_object(key: str) -> None:
    """Best-effort delete of a mirrored object (row deletion fast path)."""
    if not (enabled() and key):
        return
    try:
        _client().delete_object(Bucket=settings.R2_BUCKET, Key=key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("R2 delete failed for %s: %s", key, exc)
