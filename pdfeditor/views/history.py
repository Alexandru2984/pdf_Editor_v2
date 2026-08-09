"""Operation history view — list every ProcessedPDF the requester owns."""

import os

from django.conf import settings
from django.contrib import messages
from django.http import Http404
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

from .. import objectstore
from ..models import ProcessedPDF
from ._common import attachment_response, owner_filter


def history_view(request):
    """Show every ProcessedPDF the current requester has produced, newest first."""
    rows = list(ProcessedPDF.objects.filter(owner_filter(request)).select_related("source"))

    history_items = []
    for row in rows:
        local_available = row.exists_on_disk()
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
        messages.error(request, "File not found or no longer available.")
        return redirect("history")
    if r2_url:
        return redirect(r2_url)
    try:
        return attachment_response(output.path)
    except Http404:
        messages.error(request, "File not found.")
        return redirect("history")


@require_POST
def history_delete_view(request, output_id):
    """Remove a past output (DB row + file) owned by the current requester."""
    deleted_count, _ = ProcessedPDF.objects.filter(owner_filter(request), id=output_id).delete()
    if deleted_count:
        messages.success(request, "Output removed from history.")
    else:
        messages.error(request, "Output not found.")
    return redirect("history")
