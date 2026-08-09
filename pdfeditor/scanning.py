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
    (``"infected"``, ``"scanner-unavailable"``, or ``"scanner-error"``);
    ``detail`` is the clamd signature or error string for logs/UI."""

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
    # Only an explicit OK is clean. Treat ERROR, an unknown status, and any
    # malformed/empty response like scanner failure; accepting those would
    # silently turn a fail-closed deployment into fail-open.
    try:
        stream_result = result["stream"]
        if not isinstance(stream_result, (tuple, list)) or len(stream_result) != 2:
            raise ValueError("malformed stream result")
        status, signature = stream_result
    except (KeyError, TypeError, ValueError) as exc:
        _handle_scanner_error("malformed ClamAV response", exc)
        return

    if status == "OK":
        return
    if status == "FOUND":
        logger.warning("ClamAV blocked an upload: %s", signature)
        raise UploadBlocked("infected", signature or "unknown")

    _handle_scanner_error(f"ClamAV returned status {status!r}", signature)


def _handle_scanner_error(message: str, detail: object = "") -> None:
    full_detail = f"{message}: {detail}" if detail else message
    if getattr(settings, "CLAMAV_FAIL_OPEN", False):
        logger.warning("%s; allowing upload because fail-open is enabled", full_detail)
        return
    logger.error("%s; rejecting upload (fail-closed)", full_detail)
    raise UploadBlocked("scanner-error", full_detail)


def _safe_seek0(fileobj: BinaryIO) -> None:
    # Non-seekable stream is fine; the caller handles a missing rewind.
    with contextlib.suppress(Exception):
        fileobj.seek(0)
