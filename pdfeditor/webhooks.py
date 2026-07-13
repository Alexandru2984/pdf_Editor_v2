"""Outbound webhook helpers: URL validation, signing, and payload building.

The Celery delivery task itself lives in :mod:`pdfeditor.tasks` (so Celery's
autodiscovery picks it up); this module holds the pure, importable pieces it
uses, plus the SSRF check for user-supplied delivery URLs.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from urllib.parse import urlparse

import requests

from .pdf_processor import ssrf_guard

# Headers every delivery carries. Receivers verify SIGNATURE_HEADER with the
# webhook's secret; the id/event headers let them route without parsing the body.
SIGNATURE_HEADER = "X-PDF-Signature"
EVENT_HEADER = "X-PDF-Event"
ID_HEADER = "X-PDF-Webhook-Id"

# Consecutive fully-failed deliveries (after retries) before the endpoint is
# auto-disabled, so a permanently-dead URL isn't hammered forever.
MAX_CONSECUTIVE_FAILURES = 15

# Per-user endpoint cap. Each terminal job POSTs to every active endpoint, so
# this doubles as an amplification guard. Shared by the web UI and the API.
MAX_WEBHOOKS_PER_USER = 10

# Per-delivery HTTP timeout (seconds).
DELIVERY_TIMEOUT = 10

# Terminal deliveries retained per webhook (oldest pruned on insert) so the
# delivery-history table stays bounded without a cron.
MAX_DELIVERY_LOG = 25


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


def deliver_once(hook, event: str, payload: dict) -> tuple[bool, str]:
    """Make a single signed POST to ``hook``. Returns ``(ok, status)``.

    Validates the URL (SSRF) first — a ``"blocked: …"`` status means the target
    is no longer public and the caller should disable it. Never raises; ``ok``
    is True only on a 2xx. No retry — the Celery task layers backoff on top,
    while the synchronous ``test`` endpoint uses this directly for one attempt.
    """
    try:
        validate_webhook_url(hook.url)
    except InvalidWebhookURL as exc:
        return False, f"blocked: {exc}"[:50]

    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "PDFEditor-Webhook/1",
        SIGNATURE_HEADER: f"sha256={sign_payload(hook.secret, body)}",
        EVENT_HEADER: event,
        ID_HEADER: str(hook.id),
    }
    try:
        # allow_redirects=False: a 3xx could bounce the signed payload to an
        # internal address that would sidestep the SSRF check above.
        resp = requests.post(
            hook.url, data=body, headers=headers, timeout=DELIVERY_TIMEOUT, allow_redirects=False
        )
        return (200 <= resp.status_code < 300), str(resp.status_code)
    except requests.RequestException as exc:
        return False, f"error: {type(exc).__name__}"


def record_delivery(hook, event: str, ok: bool, status: str) -> None:
    """Append a terminal delivery outcome to the webhook's history, then prune
    it back to ``MAX_DELIVERY_LOG`` newest rows. Best-effort — logging history
    must never break the delivery itself."""
    from .models import WebhookDelivery

    try:
        WebhookDelivery.objects.create(webhook=hook, event=event, ok=ok, status=status[:50])
        keep = list(
            WebhookDelivery.objects.filter(webhook=hook)
            .order_by("-created_at")
            .values_list("id", flat=True)[:MAX_DELIVERY_LOG]
        )
        WebhookDelivery.objects.filter(webhook=hook).exclude(id__in=keep).delete()
    except Exception:  # noqa: BLE001 — history is a nicety, not load-bearing
        pass


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
