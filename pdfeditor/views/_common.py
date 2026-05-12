"""Shared helpers: ownership-scoped PDF lookup and guarded media serving.

Ownership rule: when the request is authenticated, the PDF must be linked to
``request.user``; otherwise it must match the request's anonymous
``session_key`` AND have ``user__isnull=True``. Authenticated and anonymous
rows never alias each other.
"""

import os

from django.conf import settings
from django.db import models
from django.db.models import Q
from django.http import FileResponse, Http404, HttpRequest

from ..models import AuditLog, ProcessedPDF, UploadedPDF


def storage_usage(request: HttpRequest) -> int:
    """Total bytes occupied by the requester's uploaded PDFs."""
    return UploadedPDF.objects.filter(owner_filter(request)).aggregate(total=models.Sum("size"))["total"] or 0


def storage_quota(request: HttpRequest) -> int:
    """Per-owner upload quota in bytes (0 = unlimited)."""
    if getattr(request, "user", None) is not None and request.user.is_authenticated:
        return int(getattr(settings, "PDF_QUOTA_USER_BYTES", 0) or 0)
    return int(getattr(settings, "PDF_QUOTA_ANON_BYTES", 0) or 0)


def _client_ip(request: HttpRequest) -> str | None:
    forwarded = request.META.get("HTTP_X_FORWARDED_FOR", "")
    if forwarded:
        return forwarded.split(",")[0].strip() or None
    return request.META.get("REMOTE_ADDR") or None


def ensure_session_key(request: HttpRequest) -> str:
    """Guarantee that the request has a persistent session_key and return it."""
    if not request.session.session_key:
        request.session.save()
    return request.session.session_key


def owner_filter(request: HttpRequest) -> Q:
    """Q expression matching PDFs owned by the current requester (user or anon session)."""
    if getattr(request, "user", None) is not None and request.user.is_authenticated:
        return Q(user=request.user)
    return Q(user__isnull=True, session_key=ensure_session_key(request))


def _owner_kwargs(request: HttpRequest) -> dict:
    """kwargs for ``Model.objects.create()`` that set ownership correctly."""
    if getattr(request, "user", None) is not None and request.user.is_authenticated:
        return {"user": request.user, "session_key": ""}
    return {"user": None, "session_key": ensure_session_key(request)}


def get_uploaded_pdfs(request: HttpRequest):
    """Return UploadedPDFs owned by the requester. Prunes rows whose file vanished."""
    rows = list(UploadedPDF.objects.filter(owner_filter(request)))

    stale_ids = [pdf.id for pdf in rows if not pdf.exists_on_disk()]
    if stale_ids:
        UploadedPDF.objects.filter(id__in=stale_ids).delete()
        rows = [pdf for pdf in rows if pdf.id not in stale_ids]

    return rows


def get_pdf_by_id(request: HttpRequest, pdf_id):
    return UploadedPDF.objects.filter(owner_filter(request), id=pdf_id).first()


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

    return output


def latest_output(request: HttpRequest, kind: str):
    return ProcessedPDF.objects.filter(owner_filter(request), kind=kind).first()


def _allowed_media_paths(request: HttpRequest) -> set[str]:
    """Absolute paths the current requester is allowed to read from MEDIA_ROOT."""
    f = owner_filter(request)
    allowed: set[str] = set()
    for obj in UploadedPDF.objects.filter(f):
        allowed.add(os.path.realpath(obj.path))
    for obj in ProcessedPDF.objects.filter(f):
        allowed.add(os.path.realpath(obj.path))
    return allowed


def serve_media_view(request: HttpRequest, rel_path: str) -> FileResponse:
    """Serve a MEDIA_ROOT file iff it belongs to the current requester."""
    media_root = os.path.realpath(settings.MEDIA_ROOT)
    requested = os.path.realpath(os.path.join(media_root, rel_path))

    if not requested.startswith(media_root + os.sep):
        raise Http404()
    if requested not in _allowed_media_paths(request):
        raise Http404()
    if not os.path.exists(requested):
        raise Http404()

    return FileResponse(open(requested, "rb"), content_type="application/pdf")


def attachment_response(path: str, not_found_message: str = "File not found.") -> FileResponse:
    if not path or not os.path.exists(path):
        raise Http404(not_found_message)
    return FileResponse(
        open(path, "rb"),
        as_attachment=True,
        filename=os.path.basename(path),
    )
