"""Outbound webhook helpers: URL validation, signing, and payload building.

The Celery delivery task itself lives in :mod:`pdfeditor.tasks` (so Celery's
autodiscovery picks it up); this module holds the pure, importable pieces it
uses, plus the SSRF check for user-supplied delivery URLs.
"""

from __future__ import annotations

import hashlib
import hmac
from urllib.parse import urlparse

from .pdf_processor import ssrf_guard

# Headers every delivery carries. Receivers verify SIGNATURE_HEADER with the
# webhook's secret; the id/event headers let them route without parsing the body.
SIGNATURE_HEADER = "X-PDF-Signature"
EVENT_HEADER = "X-PDF-Event"
ID_HEADER = "X-PDF-Webhook-Id"

# Consecutive fully-failed deliveries (after retries) before the endpoint is
# auto-disabled, so a permanently-dead URL isn't hammered forever.
MAX_CONSECUTIVE_FAILURES = 15


class InvalidWebhookURL(ValueError):
    """Raised when a webhook target is not a public https URL."""


def validate_webhook_url(url: str) -> None:
    """Raise :class:`InvalidWebhookURL` unless ``url`` is https to a public IP.

    Webhook targets are user-controlled, so this is the anti-SSRF gate: it
    reuses the certificate-fetch guard's public-IP resolver so a user can't
    point a webhook at ``127.0.0.1``, an RFC1918 host, the cloud-metadata
    endpoint, or an internal service. https is required — deliveries are signed
    but the payload still shouldn't cross the wire in the clear. Called on save
    *and* again at delivery time (the resolve-time check alone is DNS-rebind
    -able).
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise InvalidWebhookURL("Webhook URL must use https.")
    host = parsed.hostname
    if not host:
        raise InvalidWebhookURL("Webhook URL has no host.")
    port = parsed.port or 443
    if not ssrf_guard.hostname_resolves_public(host, port):
        raise InvalidWebhookURL("Webhook URL must resolve to a public address.")


def sign_payload(secret: str, body: bytes) -> str:
    """HMAC-SHA256 of the raw request body, hex-encoded. The receiver recomputes
    this over the bytes it receives and compares (constant-time) to the header."""
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def build_job_payload(job, event: str, delivered_at: str) -> dict:
    """The JSON body delivered for a terminal job event."""
    return {
        "event": event,
        "delivered_at": delivered_at,
        "job": {
            "id": str(job.id),
            "kind": job.kind,
            "status": job.status,
            "output_id": str(job.output_id) if job.output_id else None,
            "error_message": job.error_message or None,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        },
    }
