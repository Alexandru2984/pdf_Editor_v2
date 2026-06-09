"""Optional ClamAV malware scanning for uploads.

Disabled by default (``CLAMAV_ENABLED=0``). When enabled, an uploaded file is
streamed to a clamd daemon (INSTREAM) *before* it is written to disk, so
malware embedded in a PDF/image never lands on the server or gets served to
other users through a share link.

clamd connection details come from settings/env:
  CLAMAV_ENABLED    – master switch (default False)
  CLAMAV_HOST/PORT  – clamd network socket (default clamav:3310)
  CLAMAV_TIMEOUT    – socket timeout, seconds (default 30)
  CLAMAV_FAIL_OPEN  – when clamd is unreachable, allow the upload instead of
                      rejecting it (default False = fail closed / reject).
"""

from __future__ import annotations

import contextlib
import logging
from typing import BinaryIO

from django.conf import settings

logger = logging.getLogger(__name__)


class UploadBlocked(Exception):
    """Raised when an upload must be refused. ``reason`` is a stable code
    (``"infected"`` or ``"scanner-unavailable"``); ``detail`` is the clamd
    signature or error string for logs/UI."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


def scanning_enabled() -> bool:
    return bool(getattr(settings, "CLAMAV_ENABLED", False))


def _build_client():
    # Imported lazily so the dependency is only needed when scanning is on.
    import clamd

    return clamd.ClamdNetworkSocket(
        host=getattr(settings, "CLAMAV_HOST", "clamav"),
        port=int(getattr(settings, "CLAMAV_PORT", 3310)),
        timeout=float(getattr(settings, "CLAMAV_TIMEOUT", 30)),
    )


def scan_fileobj(fileobj: BinaryIO) -> None:
    """Scan an open binary file object. No-op when scanning is disabled.

    Raises :class:`UploadBlocked` if clamd reports malware, or — unless
    ``CLAMAV_FAIL_OPEN`` is set — if clamd cannot be reached. The stream
    position is restored to 0 on the way out so the caller can still save it.
    """
    if not scanning_enabled():
        return

    try:
        client = _build_client()
        fileobj.seek(0)
        result = client.instream(fileobj)
    except UploadBlocked:
        raise
    except Exception as exc:  # noqa: BLE001 — any clamd/socket error is "unavailable"
        if getattr(settings, "CLAMAV_FAIL_OPEN", False):
            logger.warning("ClamAV unreachable, allowing upload (fail-open): %s", exc)
            _safe_seek0(fileobj)
            return
        logger.error("ClamAV unreachable, rejecting upload (fail-closed): %s", exc)
        _safe_seek0(fileobj)
        raise UploadBlocked("scanner-unavailable", str(exc)) from exc

    _safe_seek0(fileobj)

    # clamd INSTREAM result: {"stream": ("OK"|"FOUND", signature_or_None)}.
    status, signature = (result or {}).get("stream", ("OK", None))
    if status == "FOUND":
        logger.warning("ClamAV blocked an upload: %s", signature)
        raise UploadBlocked("infected", signature or "unknown")


def _safe_seek0(fileobj: BinaryIO) -> None:
    # Non-seekable stream is fine; the caller handles a missing rewind.
    with contextlib.suppress(Exception):
        fileobj.seek(0)
