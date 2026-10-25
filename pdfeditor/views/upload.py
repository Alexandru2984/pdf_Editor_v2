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
from django.views.decorators.http import require_POST

from ..models import UploadedPDF
from ..pdf_processor import check_pdf_has_text, inspect_pdf, sanitize_pdf
from ..pdf_processor.ops import _guard_pixmap_memory
from ..ratelimiting import auth_aware_ratelimit
from ..scanning import UploadBlocked, scan_fileobj
from ..tool_catalog import is_upload_destination_allowed
from ._common import _owner_kwargs, get_uploaded_pdfs, owner_filter, storage_quota, storage_usage
from ._localization import localized_public_view

logger = logging.getLogger(__name__)

THUMB_SUBDIR = "thumbs"
THUMB_TARGET_WIDTH = 480


def _thumbnail_path(pdf_id) -> str:
    return os.path.join(settings.MEDIA_ROOT, THUMB_SUBDIR, f"{pdf_id}.jpg")


def _generate_thumbnail(pdf_path: str, thumb_path: str) -> bool:
    """Render the first page of a PDF to ``thumb_path`` as JPEG. Returns True on success."""
    try:
        with fitz.open(pdf_path) as doc:
            if len(doc) == 0:
                return False
            page = doc.load_page(0)
            zoom = min(
                THUMB_TARGET_WIDTH / max(page.rect.width, 1),
                (THUMB_TARGET_WIDTH * 2) / max(page.rect.height, 1),
            )
            _guard_pixmap_memory(page, zoom, 1)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
            pix.save(thumb_path)
        return True
    except Exception as exc:
        logger.warning("Thumbnail generation failed for %s: %s", pdf_path, exc)
        return False


@localized_public_view("dashboard", redirect_only_fresh_visits=True)
def dashboard_view(request):
    quota = storage_quota(request)
    used = storage_usage(request)
    return render(
        request,
        "pdfeditor/dashboard.html",
        {
            "uploaded_pdfs": get_uploaded_pdfs(request, include_encrypted=True),
            "storage_used": used,
            "storage_quota": quota,
            "storage_percent": (used * 100 // quota) if quota else 0,
        },
    )


MAX_FILES_PER_UPLOAD = 20


@auth_aware_ratelimit(anon_rate="20/h", user_rate="120/h", method="POST")
def upload_view(request):
    if request.method != "POST":
        return render(request, "pdfeditor/upload.html")

    uploaded_files = request.FILES.getlist("pdf_file")
    if not uploaded_files:
        messages.error(request, _("Please select at least one PDF file."))
        return render(request, "pdfeditor/upload.html")
    if len(uploaded_files) > MAX_FILES_PER_UPLOAD:
        messages.warning(
            request,
            _("Only the first %(max)d files in a single upload were processed.")
            % {"max": MAX_FILES_PER_UPLOAD},
        )
        uploaded_files = uploaded_files[:MAX_FILES_PER_UPLOAD]

    max_bytes = getattr(settings, "PDF_MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
    quota = storage_quota(request)
    quota_used = storage_usage(request)
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
        if quota and quota_used + uploaded_file.size > quota:
            messages.warning(
                request,
                _('Skipped "%(name)s" - would exceed your %(mb)d MB storage quota.')
                % {"name": uploaded_file.name, "mb": quota // (1024 * 1024)},
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

        # Malware scan before the file is written to disk (no-op unless
        # CLAMAV_ENABLED). Blocks infected files — and, when fail-closed,
        # uploads while clamd is unreachable.
        try:
            scan_fileobj(uploaded_file)
        except UploadBlocked as exc:
            logger.warning(
                "Upload blocked by scanner: %s (%s/%s)", uploaded_file.name, exc.reason, exc.detail
            )
            messages.warning(
                request,
                _('Skipped "%(name)s" - failed the malware scan.') % {"name": uploaded_file.name},
            )
            continue

        safe_name = os.path.basename(uploaded_file.name)
        filename = fs.save(safe_name, uploaded_file)
        file_path = fs.path(filename)

        max_pages = getattr(settings, "PDF_MAX_PAGES", 500)
        inspection = inspect_pdf(file_path)
        if inspection is None or inspection.page_count < 1:
            os.remove(file_path)
            messages.warning(
                request,
                _('Skipped "%(name)s" - could not be parsed as a PDF.') % {"name": uploaded_file.name},
            )
            continue
        if inspection.page_count > max_pages:
            os.remove(file_path)
            messages.warning(
                request,
                _('Skipped "%(name)s" - %(pages)d pages exceeds the %(max)d-page limit.')
                % {"name": uploaded_file.name, "pages": inspection.page_count, "max": max_pages},
            )
            continue

        ingest_status = UploadedPDF.INGEST_ENCRYPTED
        if not inspection.encrypted:
            # A sanitizer error is security-significant: never persist or
            # process the ambiguous original.
            if not sanitize_pdf(file_path):
                os.remove(file_path)
                messages.warning(
                    request,
                    _('Skipped "%(name)s" - could not be sanitized safely.') % {"name": uploaded_file.name},
                )
                continue
            ingest_status = UploadedPDF.INGEST_SANITIZED

            has_text, message = check_pdf_has_text(file_path)
            if not has_text:
                messages.warning(request, f"{uploaded_file.name}: {message}")

        pdf_obj = UploadedPDF.objects.create(
            **_owner_kwargs(request),
            name=uploaded_file.name,
            path=file_path,
            # On-disk size after sanitization, not the original upload size.
            size=os.path.getsize(file_path),
            ingest_status=ingest_status,
        )
        quota_used += pdf_obj.size
        if pdf_obj.is_processable:
            # Best-effort: failure here only means the dashboard falls back to
            # the lazy-regen path in thumbnail_view.
            _generate_thumbnail(file_path, _thumbnail_path(pdf_obj.id))
        else:
            messages.info(
                request,
                _('"%(name)s" is encrypted and is available only in Remove Password.')
                % {"name": uploaded_file.name},
            )
        created.append(pdf_obj)

    if len(created) == 1:
        messages.success(
            request,
            _('PDF "%(name)s" uploaded successfully! Choose an operation below.') % {"name": created[0].name},
        )
        destination = request.GET.get("next", "")
        return redirect(destination if is_upload_destination_allowed(destination) else "dashboard")
    if len(created) > 1:
        messages.success(
            request,
            _("%(count)d PDFs uploaded successfully! Choose an operation below.") % {"count": len(created)},
        )
        destination = request.GET.get("next", "")
        return redirect(destination if is_upload_destination_allowed(destination) else "dashboard")

    messages.error(request, _("No valid PDF files were uploaded."))
    return render(request, "pdfeditor/upload.html")


@require_POST
def delete_pdf_view(request, pdf_id):
    pdf = UploadedPDF.objects.filter(owner_filter(request), id=pdf_id).first()
    if pdf is None:
        messages.error(request, _("PDF not found."))
        return redirect("dashboard")

    # post_delete signal in pdfeditor/signals.py cleans the file + thumbnail.
    pdf.delete()
    messages.success(request, _("PDF removed successfully."))
    return redirect("dashboard")


def thumbnail_view(request, pdf_id):
    """Serve the first-page JPEG thumbnail. Owner-scoped, regenerated lazily."""
    pdf = UploadedPDF.objects.filter(
        owner_filter(request), id=pdf_id, ingest_status=UploadedPDF.INGEST_SANITIZED
    ).first()
    if pdf is None or not pdf.exists_on_disk():
        raise Http404
    thumb_path = _thumbnail_path(pdf.id)
    if not os.path.exists(thumb_path) and not _generate_thumbnail(pdf.path, thumb_path):
        raise Http404
    response = FileResponse(open(thumb_path, "rb"), content_type="image/jpeg")  # noqa: SIM115
    response["Cache-Control"] = "private, max-age=3600"
    return response


def uploaded_pdf_preview_view(request, pdf_id):
    """Stream a sanitized source PDF through an opaque, owner-scoped URL."""
    pdf = UploadedPDF.objects.filter(
        owner_filter(request),
        id=pdf_id,
        ingest_status=UploadedPDF.INGEST_SANITIZED,
    ).first()
    if pdf is None or not pdf.exists_on_disk():
        raise Http404

    response = FileResponse(
        open(pdf.path, "rb"),  # noqa: SIM115 — FileResponse owns the handle
        content_type="application/pdf",
        filename=pdf.name,
    )
    response["Cache-Control"] = "private, no-store"
    return response
