"""Shared helpers: session-scoped PDF lookup and guarded media serving."""
import os

from django.conf import settings
from django.http import FileResponse, Http404

from ..models import ProcessedPDF, UploadedPDF


def ensure_session_key(request) -> str:
    """Guarantee that the request has a persistent session_key and return it."""
    if not request.session.session_key:
        request.session.save()
    return request.session.session_key


def get_uploaded_pdfs(request):
    """Return UploadedPDF objects for the current session. Prunes stale rows."""
    session_key = ensure_session_key(request)
    rows = list(UploadedPDF.objects.filter(session_key=session_key))

    stale_ids = [pdf.id for pdf in rows if not pdf.exists_on_disk()]
    if stale_ids:
        UploadedPDF.objects.filter(id__in=stale_ids).delete()
        rows = [pdf for pdf in rows if pdf.id not in stale_ids]

    return rows


def get_pdf_by_id(request, pdf_id):
    return UploadedPDF.objects.filter(
        session_key=ensure_session_key(request),
        id=pdf_id,
    ).first()


def record_output(request, *, kind: str, path: str, source=None) -> ProcessedPDF:
    """Create a ProcessedPDF row tracking a generated output file."""
    return ProcessedPDF.objects.create(
        session_key=ensure_session_key(request),
        kind=kind,
        source=source,
        name=os.path.basename(path),
        path=path,
        size=os.path.getsize(path) if os.path.exists(path) else 0,
    )


def latest_output(request, kind: str):
    return ProcessedPDF.objects.filter(
        session_key=ensure_session_key(request),
        kind=kind,
    ).first()


def _allowed_media_paths(request):
    """Absolute paths the current session is allowed to read from MEDIA_ROOT."""
    session_key = ensure_session_key(request)
    allowed = set()
    for obj in UploadedPDF.objects.filter(session_key=session_key):
        allowed.add(os.path.realpath(obj.path))
    for obj in ProcessedPDF.objects.filter(session_key=session_key):
        allowed.add(os.path.realpath(obj.path))
    return allowed


def serve_media_view(request, rel_path):
    """Serve a MEDIA_ROOT file iff it belongs to the current session."""
    media_root = os.path.realpath(settings.MEDIA_ROOT)
    requested = os.path.realpath(os.path.join(media_root, rel_path))

    if not requested.startswith(media_root + os.sep):
        raise Http404()
    if requested not in _allowed_media_paths(request):
        raise Http404()
    if not os.path.exists(requested):
        raise Http404()

    return FileResponse(open(requested, 'rb'), content_type='application/pdf')


def attachment_response(path, not_found_message='File not found.'):
    if not path or not os.path.exists(path):
        raise Http404(not_found_message)
    return FileResponse(
        open(path, 'rb'),
        as_attachment=True,
        filename=os.path.basename(path),
    )
