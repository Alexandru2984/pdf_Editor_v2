"""Dashboard, upload, delete."""

import logging
import os

import fitz
from django.conf import settings
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from django.http import FileResponse, Http404
from django.shortcuts import redirect, render
from django.utils.translation import gettext as _

from ..models import UploadedPDF
from ..pdf_processor import check_pdf_has_text
from ._common import _owner_kwargs, get_uploaded_pdfs, owner_filter

logger = logging.getLogger(__name__)

THUMB_SUBDIR = "thumbs"
THUMB_TARGET_WIDTH = 480


def _count_pages_safely(file_path):
    """Return page count or None if the PDF cannot be opened."""
    try:
        with fitz.open(file_path) as doc:
            return len(doc)
    except Exception as exc:
        logger.warning("Failed to inspect uploaded PDF %s: %s", file_path, exc)
        return None


def _thumbnail_path(pdf_id) -> str:
    return os.path.join(settings.MEDIA_ROOT, THUMB_SUBDIR, f"{pdf_id}.jpg")


def _generate_thumbnail(pdf_path: str, thumb_path: str) -> bool:
    """Render the first page of a PDF to ``thumb_path`` as JPEG. Returns True on success."""
    try:
        with fitz.open(pdf_path) as doc:
            if len(doc) == 0:
                return False
            page = doc.load_page(0)
            zoom = THUMB_TARGET_WIDTH / max(page.rect.width, 1)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
            pix.save(thumb_path)
        return True
    except Exception as exc:
        logger.warning("Thumbnail generation failed for %s: %s", pdf_path, exc)
        return False


def dashboard_view(request):
    return render(
        request,
        "pdfeditor/dashboard.html",
        {
            "uploaded_pdfs": get_uploaded_pdfs(request),
        },
    )


def upload_view(request):
    if request.method != "POST":
        return render(request, "pdfeditor/upload.html")

    uploaded_files = request.FILES.getlist("pdf_file")
    if not uploaded_files:
        messages.error(request, _("Please select at least one PDF file."))
        return render(request, "pdfeditor/upload.html")

    max_bytes = getattr(settings, "PDF_MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
    fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, "uploads"))

    created = []
    for uploaded_file in uploaded_files:
        if not uploaded_file.name.lower().endswith(".pdf"):
            messages.warning(
                request,
                _('Skipped "%(name)s" - only PDF files are accepted.') % {"name": uploaded_file.name},
            )
            continue
        if uploaded_file.size > max_bytes:
            messages.warning(
                request,
                _('Skipped "%(name)s" - exceeds %(mb)d MB limit.')
                % {"name": uploaded_file.name, "mb": max_bytes // (1024 * 1024)},
            )
            continue

        header = uploaded_file.read(5)
        uploaded_file.seek(0)
        if header != b"%PDF-":
            messages.warning(
                request,
                _('Skipped "%(name)s" - not a valid PDF file.') % {"name": uploaded_file.name},
            )
            continue

        safe_name = os.path.basename(uploaded_file.name)
        filename = fs.save(safe_name, uploaded_file)
        file_path = fs.path(filename)

        max_pages = getattr(settings, "PDF_MAX_PAGES", 500)
        page_count = _count_pages_safely(file_path)
        if page_count is None:
            os.remove(file_path)
            messages.warning(
                request,
                _('Skipped "%(name)s" - could not be parsed as a PDF.') % {"name": uploaded_file.name},
            )
            continue
        if page_count > max_pages:
            os.remove(file_path)
            messages.warning(
                request,
                _('Skipped "%(name)s" - %(pages)d pages exceeds the %(max)d-page limit.')
                % {"name": uploaded_file.name, "pages": page_count, "max": max_pages},
            )
            continue

        has_text, message = check_pdf_has_text(file_path)
        if not has_text:
            messages.warning(request, f"{uploaded_file.name}: {message}")

        pdf_obj = UploadedPDF.objects.create(
            **_owner_kwargs(request),
            name=uploaded_file.name,
            path=file_path,
            size=uploaded_file.size,
        )
        # Best-effort: failure here only means the dashboard falls back to
        # the lazy-regen path in thumbnail_view.
        _generate_thumbnail(file_path, _thumbnail_path(pdf_obj.id))
        created.append(pdf_obj)

    if len(created) == 1:
        messages.success(
            request,
            _('PDF "%(name)s" uploaded successfully! Choose an operation below.') % {"name": created[0].name},
        )
        return redirect("dashboard")
    if len(created) > 1:
        messages.success(
            request,
            _("%(count)d PDFs uploaded successfully! Choose an operation below.") % {"count": len(created)},
        )
        return redirect("dashboard")

    messages.error(request, _("No valid PDF files were uploaded."))
    return render(request, "pdfeditor/upload.html")


def delete_pdf_view(request, pdf_id):
    pdf = UploadedPDF.objects.filter(owner_filter(request), id=pdf_id).first()
    if pdf is None:
        messages.error(request, _("PDF not found."))
        return redirect("dashboard")

    thumb_path = _thumbnail_path(pdf.id)
    if os.path.exists(thumb_path):
        try:
            os.remove(thumb_path)
        except OSError as exc:
            logger.warning("Failed to remove thumbnail %s: %s", thumb_path, exc)

    pdf.delete()
    messages.success(request, _("PDF removed successfully."))
    return redirect("dashboard")


def thumbnail_view(request, pdf_id):
    """Serve the first-page JPEG thumbnail. Owner-scoped, regenerated lazily."""
    pdf = UploadedPDF.objects.filter(owner_filter(request), id=pdf_id).first()
    if pdf is None or not pdf.exists_on_disk():
        raise Http404
    thumb_path = _thumbnail_path(pdf.id)
    if not os.path.exists(thumb_path) and not _generate_thumbnail(pdf.path, thumb_path):
        raise Http404
    response = FileResponse(open(thumb_path, "rb"), content_type="image/jpeg")  # noqa: SIM115
    response["Cache-Control"] = "private, max-age=3600"
    return response
