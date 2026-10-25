"""Operation history view — list every ProcessedPDF the requester owns."""

import os
import shutil

from django.conf import settings
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from django.http import Http404
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from .. import objectstore
from ..models import ProcessedPDF, UploadedPDF
from ..pdf_processor import inspect_pdf
from ..ratelimiting import auth_aware_ratelimit
from ._common import _owner_kwargs, attachment_response, owner_filter, storage_quota, storage_usage
from .upload import _generate_thumbnail, _thumbnail_path

_CHAINABLE_TOOLS = frozenset(
    {
        "edit",
        "rephrase",
        "summarize",
        "split",
        "merge",
        "compress",
        "watermark",
        "rotate",
        "page_numbers",
        "reorder",
        "crop",
        "sign",
        "protect",
        "unprotect",
        "flatten",
        "redact",
        "searchable",
        "pdfa",
        "outline",
        "metadata",
        "form_fill",
        "convert",
        "to_images",
        "chat",
        "share_links",
        "qr_code",
    }
)


def history_view(request):
    """Show every ProcessedPDF the current requester has produced, newest first."""
    rows = list(ProcessedPDF.objects.filter(owner_filter(request)).select_related("source"))

    history_items = []
    for row in rows:
        local_available = row.exists_on_disk()
        is_pdf = row.name.lower().endswith(".pdf") and row.kind != ProcessedPDF.KIND_PROTECT
        history_items.append(
            {
                "id": row.id,
                "kind": row.kind,
                "kind_display": row.get_kind_display(),
                "name": row.name,
                "size": row.size,
                "created_at": row.created_at,
                "source_name": row.source.name if row.source else None,
                "path_relative": os.path.relpath(row.path, settings.MEDIA_ROOT) if local_available else "",
                "local_available": local_available,
                "is_pdf": is_pdf,
                # A stored object key is enough to offer the download action;
                # the download view performs the authoritative check.
                "download_available": local_available or bool(row.r2_key),
            }
        )

    return render(
        request,
        "pdfeditor/history.html",
        {
            "history_items": history_items,
            "PDF_CLEANUP_HOURS": getattr(settings, "PDF_CLEANUP_HOURS", 24),
        },
    )


def history_download_view(request, output_id):
    """Download any past output, gated by ownership (user or anonymous session)."""
    output = ProcessedPDF.objects.filter(owner_filter(request), id=output_id).first()
    r2_url = objectstore.presigned_download_url(output) if output else None
    if not output or not (output.exists_on_disk() or r2_url):
        messages.error(request, _("File not found or no longer available."))
        return redirect("history")
    if r2_url:
        return redirect(r2_url)
    try:
        return attachment_response(output.path)
    except Http404:
        messages.error(request, _("File not found."))
        return redirect("history")


@require_POST
def history_delete_view(request, output_id):
    """Remove a past output (DB row + file) owned by the current requester."""
    deleted_count, _ = ProcessedPDF.objects.filter(owner_filter(request), id=output_id).delete()
    if deleted_count:
        messages.success(request, _("Output removed from history."))
    else:
        messages.error(request, _("Output not found."))
    return redirect("history")


@require_POST
@auth_aware_ratelimit(anon_rate="30/h", user_rate="180/h", method="POST")
def history_use_view(request, output_id=None):
    """Promote a ProcessedPDF output into an active UploadedPDF and open it in the chosen tool."""
    resolved_id = output_id or request.session.get("last_processed_pdf_id")
    if not resolved_id:
        messages.error(request, _("No processed PDF found in this session."))
        return redirect("dashboard")

    output = ProcessedPDF.objects.filter(owner_filter(request), id=resolved_id).first()
    if not output or not output.exists_on_disk():
        messages.error(request, _("Processed file not found or no longer available on disk."))
        return redirect("history")

    next_tool = (request.POST.get("tool") or "").strip()
    if next_tool == "share_links":
        return redirect(f"{reverse('share_links')}?output={output.id}")

    if not output.path.lower().endswith(".pdf"):
        messages.error(request, _("Only PDF outputs can be chained into another PDF tool."))
        return redirect("history")

    quota = storage_quota(request)
    used = storage_usage(request)
    output_size = os.path.getsize(output.path)
    if quota and used + output_size > quota:
        messages.error(
            request,
            _("Cannot chain PDF - would exceed your %(mb)d MB storage quota.")
            % {"mb": quota // (1024 * 1024)},
        )
        return redirect("history")

    inspection = inspect_pdf(output.path)
    if inspection is None or inspection.page_count < 1:
        messages.error(request, _("Processed file could not be opened as a valid PDF."))
        return redirect("history")

    fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, "uploads"))
    os.makedirs(fs.location, exist_ok=True)
    safe_name = os.path.basename(output.name)
    avail_name = fs.get_available_name(safe_name)
    dest_path = fs.path(avail_name)
    shutil.copy2(output.path, dest_path)

    ingest_status = UploadedPDF.INGEST_ENCRYPTED if inspection.encrypted else UploadedPDF.INGEST_SANITIZED
    pdf_obj = UploadedPDF.objects.create(
        **_owner_kwargs(request),
        name=safe_name,
        path=dest_path,
        size=os.path.getsize(dest_path),
        ingest_status=ingest_status,
    )
    if pdf_obj.is_processable:
        _generate_thumbnail(dest_path, _thumbnail_path(pdf_obj.id))

    messages.success(
        request,
        _('Loaded "%(name)s" for your next operation.') % {"name": pdf_obj.name},
    )

    if next_tool == "chat":
        return redirect("chat", pdf_id=pdf_obj.id)
    if next_tool in _CHAINABLE_TOOLS:
        try:
            return redirect(f"{reverse(next_tool)}?pdf={pdf_obj.id}")
        except Exception:  # noqa: BLE001
            pass
    return redirect("dashboard")
