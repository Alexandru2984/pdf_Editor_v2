"""Dashboard, upload, delete."""
import logging
import os

import fitz
from django.conf import settings
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from django.shortcuts import redirect, render

from ..models import UploadedPDF
from ..pdf_processor import check_pdf_has_text
from ._common import ensure_session_key, get_uploaded_pdfs

logger = logging.getLogger(__name__)


def _count_pages_safely(file_path):
    """Return page count or None if the PDF cannot be opened."""
    try:
        with fitz.open(file_path) as doc:
            return len(doc)
    except Exception as exc:
        logger.warning("Failed to inspect uploaded PDF %s: %s", file_path, exc)
        return None


def dashboard_view(request):
    return render(request, 'pdfeditor/dashboard.html', {
        'uploaded_pdfs': get_uploaded_pdfs(request),
    })


def upload_view(request):
    if request.method != 'POST':
        return render(request, 'pdfeditor/upload.html')

    uploaded_files = request.FILES.getlist('pdf_file')
    if not uploaded_files:
        messages.error(request, 'Please select at least one PDF file.')
        return render(request, 'pdfeditor/upload.html')

    session_key = ensure_session_key(request)
    max_bytes = getattr(settings, 'PDF_MAX_UPLOAD_BYTES', 10 * 1024 * 1024)
    fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploads'))

    created = []
    for uploaded_file in uploaded_files:
        if not uploaded_file.name.lower().endswith('.pdf'):
            messages.warning(request, f'Skipped "{uploaded_file.name}" - only PDF files are accepted.')
            continue
        if uploaded_file.size > max_bytes:
            messages.warning(request, f'Skipped "{uploaded_file.name}" - exceeds {max_bytes // (1024 * 1024)} MB limit.')
            continue

        header = uploaded_file.read(5)
        uploaded_file.seek(0)
        if header != b'%PDF-':
            messages.warning(request, f'Skipped "{uploaded_file.name}" - not a valid PDF file.')
            continue

        safe_name = os.path.basename(uploaded_file.name)
        filename = fs.save(safe_name, uploaded_file)
        file_path = fs.path(filename)

        max_pages = getattr(settings, 'PDF_MAX_PAGES', 500)
        page_count = _count_pages_safely(file_path)
        if page_count is None:
            os.remove(file_path)
            messages.warning(request, f'Skipped "{uploaded_file.name}" - could not be parsed as a PDF.')
            continue
        if page_count > max_pages:
            os.remove(file_path)
            messages.warning(
                request,
                f'Skipped "{uploaded_file.name}" - {page_count} pages exceeds the {max_pages}-page limit.',
            )
            continue

        has_text, message = check_pdf_has_text(file_path)
        if not has_text:
            messages.warning(request, f'{uploaded_file.name}: {message}')

        created.append(UploadedPDF.objects.create(
            session_key=session_key,
            name=uploaded_file.name,
            path=file_path,
            size=uploaded_file.size,
        ))

    if len(created) == 1:
        messages.success(request, f'PDF "{created[0].name}" uploaded successfully! Choose an operation below.')
        return redirect('dashboard')
    if len(created) > 1:
        messages.success(request, f'{len(created)} PDFs uploaded successfully! Choose an operation below.')
        return redirect('dashboard')

    messages.error(request, 'No valid PDF files were uploaded.')
    return render(request, 'pdfeditor/upload.html')


def delete_pdf_view(request, pdf_id):
    deleted, _ = UploadedPDF.objects.filter(
        session_key=ensure_session_key(request),
        id=pdf_id,
    ).delete()

    if deleted:
        messages.success(request, 'PDF removed successfully.')
    else:
        messages.error(request, 'PDF not found.')

    return redirect('dashboard')
