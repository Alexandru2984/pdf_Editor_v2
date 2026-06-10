"""Request-scoped instrumentation middleware.

Generates an X-Request-Id for every request and exposes it to the logging
subsystem via a ``contextvars.ContextVar``. Logs emitted while a request is
in flight are tagged with that ID, making it trivial to correlate a user
report ("the page broke at 12:04") with a specific request in the logs.
"""

from __future__ import annotations

import contextvars
import logging
import re
import secrets
import uuid
from collections.abc import Callable

from django.http import HttpRequest, HttpResponse

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# An inbound X-Request-Id is echoed into logs and the response header, so it
# must be inert: bounded length + no CR/LF or control chars. Anything that
# doesn't match is discarded in favour of a fresh UUID (blocks log forging
# and response-header injection).
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def get_current_request_id() -> str:
    return _request_id_var.get()


class RequestIDMiddleware:
    """Attach an X-Request-Id to every request and propagate it via contextvar + response header.

    Honours an inbound ``X-Request-Id`` header (so reverse proxies / clients
    can correlate their own logs); generates a fresh UUID4 otherwise.
    """

    HEADER = "HTTP_X_REQUEST_ID"

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        incoming = request.META.get(self.HEADER, "").strip()
        rid = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        token = _request_id_var.set(rid)
        request.request_id = rid
        try:
            response = self.get_response(request)
        finally:
            _request_id_var.reset(token)
        response["X-Request-Id"] = rid
        return response


# Disable browser features the app never uses. Set at the Django layer (not
# just the host nginx) so the header is present no matter what fronts the app
# — including the docker-compose-only deploy, whose nginx sets no security
# headers. CSP is intentionally NOT set here: the host nginx already sends one
# and two CSP headers would both be enforced. Override via settings if needed.
_DEFAULT_PERMISSIONS_POLICY = (
    "accelerometer=(), autoplay=(), camera=(), display-capture=(), "
    "encrypted-media=(), fullscreen=(self), geolocation=(), gyroscope=(), "
    "magnetometer=(), microphone=(), midi=(), payment=(), usb=(), "
    "interest-cohort=()"
)

# Strict Content-Security-Policy for the app's own pages. Inline scripts are
# allowed only via the per-request nonce ('{nonce}' is substituted at runtime)
# — no 'unsafe-inline' and no 'unsafe-eval' for scripts. PDF.js is self-hosted
# under /static/vendor/pdfjs/ with isEvalSupported:false, so no CDN script
# origins are needed. 'unsafe-inline' stays for STYLES (the app has many
# inline style= attributes/blocks).
_DEFAULT_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'nonce-{nonce}' "
    "https://analytics.micutu.com https://static.cloudflareinsights.com https://*.cloudflare.com; "
    "script-src-elem 'self' 'nonce-{nonce}' "
    "https://analytics.micutu.com https://static.cloudflareinsights.com https://*.cloudflare.com; "
    "worker-src 'self' blob:; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob: https:; "
    "font-src 'self' data:; "
    "connect-src 'self' https://analytics.micutu.com https://*.cloudflare.com https://cloudflareinsights.com; "
    "object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'; "
    # Browsers POST violation reports here (views/csp.py) — log + Prometheus
    # counter, so a policy regression is an alert, not silent breakage.
    "report-uri /csp-report/"
)
# Appended only when serving over HTTPS (SECURE_SSL). On the plain-HTTP dev
# server / e2e it would force subresources to https://localhost and break them.
_CSP_HTTPS_SUFFIX = "; upgrade-insecure-requests"

# Paths skipped by the strict CSP: Django admin and the DRF/Swagger API UIs
# render third-party inline scripts we don't control. They keep relying on the
# host nginx CSP (which still carries 'unsafe-inline'); applying the nonce
# policy here would break them.
_CSP_SKIP_PREFIXES = ("/admin/", "/api/")


class SecurityHeadersMiddleware:
    """Per-request CSP nonce + Permissions-Policy.

    Exposes ``request.csp_nonce`` (used by inline ``<script nonce>`` tags) and
    sets a strict, nonce-based Content-Security-Policy on the app's own pages.
    """

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        self.get_response = get_response
        from django.conf import settings

        self.permissions_policy = getattr(settings, "PERMISSIONS_POLICY", _DEFAULT_PERMISSIONS_POLICY)
        self.csp_template = getattr(settings, "CSP_POLICY", _DEFAULT_CSP)
        if getattr(settings, "SECURE_SSL", False):
            self.csp_template += _CSP_HTTPS_SUFFIX
        self.csp_skip_prefixes = tuple(getattr(settings, "CSP_SKIP_PREFIXES", _CSP_SKIP_PREFIXES))

    def __call__(self, request: HttpRequest) -> HttpResponse:
        nonce = secrets.token_urlsafe(16)
        request.csp_nonce = nonce
        response = self.get_response(request)
        if self.permissions_policy:
            response.setdefault("Permissions-Policy", self.permissions_policy)
        if self.csp_template and not request.path.startswith(self.csp_skip_prefixes):
            response.setdefault("Content-Security-Policy", self.csp_template.replace("{nonce}", nonce))
        return response


class RequestIDLogFilter(logging.Filter):
    """Inject ``request_id`` onto every LogRecord so the formatter can render it."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = _request_id_var.get()
        return True
