"""Canonical client-IP derivation, resistant to ``X-Forwarded-For`` spoofing.

A single source of truth shared by the rate limiter, login lockout (axes) and
the ``/metrics`` allowlist, so the three can never key on different addresses.

We trust exactly ``settings.TRUSTED_PROXY_COUNT`` reverse proxies in front of
the app (prod: Cloudflare → host nginx → docker nginx = 3). Each one appends a
single entry to ``X-Forwarded-For``, so the real client is the Nth entry
counted *from the right*; anything further left is supplied by the client and
must not be trusted. When the header is absent or shorter than the trusted
depth we refuse to guess and fall back to ``REMOTE_ADDR``.
"""

from __future__ import annotations

from django.conf import settings


def trusted_proxy_count() -> int:
    try:
        return max(0, int(getattr(settings, "TRUSTED_PROXY_COUNT", 0)))
    except (TypeError, ValueError):
        return 0


def client_ip(request) -> str | None:
    """Best-effort real client IP for ``request`` (None if undeterminable).

    Also accepts axes' AccessAttempt-style objects: anything exposing a
    ``META`` mapping works, so this can back ``AXES_CLIENT_IP_CALLABLE``.
    """
    meta = getattr(request, "META", None) or {}
    n = trusted_proxy_count()
    if n:
        parts = [p.strip() for p in meta.get("HTTP_X_FORWARDED_FOR", "").split(",") if p.strip()]
        if len(parts) >= n:
            # parts[-n]: the address the outermost trusted proxy recorded for
            # whoever connected to it — the real client when the count is right.
            return parts[-n]
        # Header shorter than the configured proxy depth: don't trust a
        # client-supplied leftmost value; fall through to REMOTE_ADDR.
    return meta.get("REMOTE_ADDR") or None
