"""Shared helpers: session PDF lookup + session-scoped media serving."""
import os

from django.conf import settings
from django.http import FileResponse, Http404


def get_uploaded_pdfs(request):
    """Return uploaded PDFs from session, filtering out files that have been deleted on disk."""
    uploaded_pdfs = request.session.get('uploaded_pdfs', [])
    valid_pdfs = [pdf for pdf in uploaded_pdfs if os.path.exists(pdf['path'])]
    if len(valid_pdfs) != len(uploaded_pdfs):
        request.session['uploaded_pdfs'] = valid_pdfs
    return valid_pdfs


def get_pdf_by_id(request, pdf_id):
    for pdf in get_uploaded_pdfs(request):
        if pdf['id'] == pdf_id:
            return pdf
    return None


_SESSION_PDF_KEYS = (
    'processed_pdf_path', 'merged_pdf_path', 'compressed_pdf_path',
    'watermarked_pdf_path', 'rotated_pdf_path', 'numbered_pdf_path',
    'rephrased_pdf_path',
)


def _allowed_media_paths(request):
    """Absolute paths the current session is allowed to read from MEDIA_ROOT."""
    allowed = set()
    for pdf in request.session.get('uploaded_pdfs', []) or []:
        p = pdf.get('path')
        if p:
            allowed.add(os.path.realpath(p))
    for key in _SESSION_PDF_KEYS:
        p = request.session.get(key)
        if p:
            allowed.add(os.path.realpath(p))
    for p in request.session.get('split_files', []) or []:
        if p:
            allowed.add(os.path.realpath(p))
    return allowed


def serve_media_view(request, rel_path):
    """Serve files from MEDIA_ROOT only if they belong to the current session."""
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
    """Return a FileResponse for download, or raise Http404 if missing."""
    if not path or not os.path.exists(path):
        raise Http404(not_found_message)
    return FileResponse(
        open(path, 'rb'),
        as_attachment=True,
        filename=os.path.basename(path),
    )
