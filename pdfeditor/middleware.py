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


class SecurityHeadersMiddleware:
    """Add deploy-independent response security headers (Permissions-Policy)."""

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        self.get_response = get_response
        from django.conf import settings

        self.permissions_policy = getattr(settings, "PERMISSIONS_POLICY", _DEFAULT_PERMISSIONS_POLICY)

    def __call__(self, request: HttpRequest) -> HttpResponse:
        response = self.get_response(request)
        if self.permissions_policy:
            response.setdefault("Permissions-Policy", self.permissions_policy)
        return response


class RequestIDLogFilter(logging.Filter):
    """Inject ``request_id`` onto every LogRecord so the formatter can render it."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = _request_id_var.get()
        return True
