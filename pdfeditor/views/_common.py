"""Shared helpers: ownership-scoped PDF lookup and guarded media serving.

Ownership rule: when the request is authenticated, the PDF must be linked to
``request.user``; otherwise it must match the request's anonymous
``session_key`` AND have ``user__isnull=True``. Authenticated and anonymous
rows never alias each other.
"""

import mimetypes
import os

from django.conf import settings
from django.db import models
from django.db.models import Q
from django.http import FileResponse, Http404, HttpRequest

from .. import objectstore
from ..models import AuditLog, ProcessedPDF, UploadedPDF
from ..netutils import client_ip

_SAFE_MEDIA_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".zip": "application/zip",
    ".png": "image/png",
}


def storage_usage(request: HttpRequest) -> int:
    """Total bytes occupied by the requester's uploaded and processed PDFs."""
    f = owner_filter(request)
    uploaded_total = UploadedPDF.objects.filter(f).aggregate(total=models.Sum("size"))["total"] or 0
    processed_total = ProcessedPDF.objects.filter(f).aggregate(total=models.Sum("size"))["total"] or 0
    return int(uploaded_total) + int(processed_total)


def storage_quota(request: HttpRequest) -> int:
    """Per-owner upload quota in bytes (0 = unlimited)."""
    if getattr(request, "user", None) is not None and request.user.is_authenticated:
        return int(getattr(settings, "PDF_QUOTA_USER_BYTES", 0) or 0)
    return int(getattr(settings, "PDF_QUOTA_ANON_BYTES", 0) or 0)


def _client_ip(request: HttpRequest) -> str | None:
    # Audit-trail IP must use the same spoof-safe XFF derivation as the rate
    # limiter and axes — a client-supplied leftmost XFF entry must not be able
    # to forge AuditLog rows.
    return client_ip(request)


def ensure_session_key(request: HttpRequest) -> str:
    """Guarantee that the request has a persistent session_key and return it."""
    if not request.session.session_key:
        request.session.save()
    return request.session.session_key


def owner_filter(request: HttpRequest) -> Q:
    """Q expression matching PDFs owned by the current requester (user or anon session)."""
    if getattr(request, "user", None) is not None and request.user.is_authenticated:
        return Q(user=request.user)

    # Read-only requests from a new anonymous visitor must not create a
    # database-backed session.  Write paths call ``_owner_kwargs`` below,
    # which deliberately creates a session before persisting owned data.
    session_key = request.session.session_key
    if not session_key:
        return Q(pk__isnull=True)
    return Q(user__isnull=True, session_key=session_key)


def _owner_kwargs(request: HttpRequest) -> dict:
    """kwargs for ``Model.objects.create()`` that set ownership correctly."""
    if getattr(request, "user", None) is not None and request.user.is_authenticated:
        return {"user": request.user, "session_key": ""}
    return {"user": None, "session_key": ensure_session_key(request)}


def get_uploaded_pdfs(request: HttpRequest, *, include_encrypted: bool = False):
    """Return owned uploads without mutating state.

    Missing files are reconciled by explicit retention/repair jobs.  Deleting
    rows during a GET would turn a transient media-volume outage into
    permanent metadata loss. Encrypted uploads are quarantined from normal
    tools unless a caller explicitly requests them for the unprotect flow.
    """
    queryset = UploadedPDF.objects.filter(owner_filter(request))
    if not include_encrypted:
        queryset = queryset.filter(ingest_status=UploadedPDF.INGEST_SANITIZED)
    return list(queryset)


def get_pdf_by_id(request: HttpRequest, pdf_id, *, include_encrypted: bool = False):
    queryset = UploadedPDF.objects.filter(owner_filter(request), id=pdf_id)
    if not include_encrypted:
        queryset = queryset.filter(ingest_status=UploadedPDF.INGEST_SANITIZED)
    return queryset.first()


def record_output(request: HttpRequest, *, kind: str, path: str, source=None) -> ProcessedPDF:
    """Create a ProcessedPDF row tracking a generated output file.

    Also writes an immutable AuditLog entry capturing who ran the op and
    where from. Audit failures are swallowed — they must never block the
    user's successful operation."""
    from ..metrics import OP_TOTAL

    OP_TOTAL.labels(kind=kind, outcome="success").inc()
    output = ProcessedPDF.objects.create(
        **_owner_kwargs(request),
        kind=kind,
        source=source,
        name=os.path.basename(path),
        path=path,
        size=os.path.getsize(path) if os.path.exists(path) else 0,
    )

    try:
        owner = _owner_kwargs(request)
        AuditLog.objects.create(
            user=owner.get("user"),
            session_key=owner.get("session_key", "") or "",
            kind=kind,
            source_name=getattr(source, "name", "") or "",
            output_name=output.name,
            output_size=output.size,
            ip_address=_client_ip(request),
            user_agent=request.META.get("HTTP_USER_AGENT", "")[:300],
        )
    except Exception:  # noqa: BLE001 — audit must never break the user flow
        pass

    objectstore.schedule_mirror(output)
    return output


def latest_output(request: HttpRequest, kind: str):
    return ProcessedPDF.objects.filter(owner_filter(request), kind=kind).first()


def _is_owned_media_path(request: HttpRequest, requested_real: str) -> bool:
    """Check whether ``requested_real`` belongs to the current requester without full-table realpath scans."""
    f = owner_filter(request)
    if (
        UploadedPDF.objects.filter(
            f,
            ingest_status=UploadedPDF.INGEST_SANITIZED,
            path=requested_real,
        ).exists()
        or ProcessedPDF.objects.filter(f, path=requested_real).exists()
    ):
        return True
    # Fallback for rows whose stored path uses a non-canonical prefix (e.g. symlinked MEDIA_ROOT in tests).
    for raw_path in UploadedPDF.objects.filter(f, ingest_status=UploadedPDF.INGEST_SANITIZED).values_list(
        "path", flat=True
    ):
        if os.path.realpath(raw_path) == requested_real:
            return True
    for raw_path in ProcessedPDF.objects.filter(f).values_list("path", flat=True):
        if os.path.realpath(raw_path) == requested_real:
            return True
    return False


def serve_media_view(request: HttpRequest, rel_path: str) -> FileResponse:
    """Serve a MEDIA_ROOT file iff it belongs to the current requester."""
    media_root = os.path.realpath(settings.MEDIA_ROOT)
    requested = os.path.realpath(os.path.join(media_root, rel_path))

    if not requested.startswith(media_root + os.sep):
        raise Http404()
    if not _is_owned_media_path(request, requested):
        raise Http404()
    if not os.path.exists(requested):
        raise Http404()

    ext = os.path.splitext(requested)[1].lower()
    content_type = (
        _SAFE_MEDIA_MIME_TYPES.get(ext) or mimetypes.guess_type(requested)[0] or "application/octet-stream"
    )
    as_attachment = ext not in (".pdf", ".png")
    response = FileResponse(
        open(requested, "rb"),  # noqa: SIM115 — FileResponse owns the handle
        as_attachment=as_attachment,
        filename=os.path.basename(requested),
        content_type=content_type,
    )
    response["Cache-Control"] = "private, no-store"
    response["X-Content-Type-Options"] = "nosniff"
    return response


def attachment_response(path: str, not_found_message: str = "File not found.") -> FileResponse:
    if not path or not os.path.exists(path):
        raise Http404(not_found_message)
    response = FileResponse(
        open(path, "rb"),  # noqa: SIM115 — FileResponse owns the handle
        as_attachment=True,
        filename=os.path.basename(path),
    )
    response["Cache-Control"] = "private, no-store"
    response["X-Content-Type-Options"] = "nosniff"
    return response
