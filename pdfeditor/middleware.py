"""Request-scoped instrumentation middleware.

Generates an X-Request-Id for every request and exposes it to the logging
subsystem via a ``contextvars.ContextVar``. Logs emitted while a request is
in flight are tagged with that ID, making it trivial to correlate a user
report ("the page broke at 12:04") with a specific request in the logs.
"""

from __future__ import annotations

import contextvars
import logging
import uuid
from collections.abc import Callable
from typing import Optional

from django.http import HttpRequest, HttpResponse

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


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
        rid = incoming or uuid.uuid4().hex
        token = _request_id_var.set(rid)
        request.request_id = rid
        try:
            response = self.get_response(request)
        finally:
            _request_id_var.reset(token)
        response["X-Request-Id"] = rid
        return response


class RequestIDLogFilter(logging.Filter):
    """Inject ``request_id`` onto every LogRecord so the formatter can render it."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = _request_id_var.get()
        return True
