"""Public token-based share links for processed PDFs."""

import os
from datetime import timedelta

from django.contrib import messages
from django.db.models import F
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.translation import gettext as _
from django.views.decorators.http import require_http_methods

from ..forms import ShareLinkForm
from ..models import ProcessedPDF, ShareLink
from ..ratelimiting import auth_aware_ratelimit
from ._common import owner_filter


@auth_aware_ratelimit(anon_rate="10/h", user_rate="60/h", method="POST")
@require_http_methods(["POST"])
def create_share_link_view(request):
    form = ShareLinkForm(request.POST)
    if not form.is_valid():
        messages.error(request, _("Invalid share link request."))
        return redirect("share_links")

    processed = ProcessedPDF.objects.filter(
        owner_filter(request),
        id=form.cleaned_data["processed_pdf_id"],
    ).first()
    if not processed or not processed.exists_on_disk():
        messages.error(request, _("PDF not found."))
        return redirect("share_links")

    ttl_hours = int(form.cleaned_data["ttl_hours"])
    expires_at = timezone.now() + timedelta(hours=ttl_hours)

    creator = request.user if request.user.is_authenticated else None
    session_key = "" if creator else (request.session.session_key or "")
    if not creator and not session_key:
        request.session.save()
        session_key = request.session.session_key or ""

    link = ShareLink.objects.create(
        processed_pdf=processed,
        creator=creator,
        session_key=session_key,
        expires_at=expires_at,
        max_downloads=form.cleaned_data["max_downloads"],
    )

    messages.success(request, _("Share link created."))
    request.session["last_share_token"] = link.token
    return redirect("share_links")


def share_links_view(request):
    qs = ShareLink.objects.filter(_owner_filter_for_links(request)).select_related("processed_pdf")
    last_token = request.session.pop("last_share_token", None)
    public_links = [
        {
            "link": link,
            "url": request.build_absolute_uri(reverse("public_share_download", args=[link.token])),
            "is_new": link.token == last_token,
        }
        for link in qs
    ]

    processed_qs = ProcessedPDF.objects.filter(owner_filter(request))[:50]

    return render(
        request,
        "pdfeditor/share_links.html",
        {
            "links": public_links,
            "processed_pdfs": processed_qs,
            "form": ShareLinkForm(),
        },
    )


@auth_aware_ratelimit(anon_rate="30/h", user_rate="200/h", method="POST")
@require_http_methods(["POST"])
def revoke_share_link_view(request, token: str):
    link = ShareLink.objects.filter(_owner_filter_for_links(request), token=token).first()
    if link is None:
        messages.error(request, _("Share link not found."))
    else:
        link.delete()
        messages.success(request, _("Share link revoked."))
    return redirect("share_links")


def public_share_download_view(request, token: str):
    """Public, no-auth endpoint that serves the PDF behind a share token."""
    link = get_object_or_404(ShareLink, token=token)
    if link.is_expired():
        return render(request, "pdfeditor/share_unavailable.html", {"reason": "expired"}, status=410)
    if link.is_exhausted():
        return render(request, "pdfeditor/share_unavailable.html", {"reason": "exhausted"}, status=410)

    pdf = link.processed_pdf
    if not pdf or not pdf.exists_on_disk():
        return render(request, "pdfeditor/share_unavailable.html", {"reason": "missing"}, status=410)

    # Atomically claim a download slot. The is_exhausted() check above is a
    # fast path; this conditional UPDATE is what actually enforces the cap
    # under concurrency — read-modify-write (count = python_value + 1) could
    # let parallel requests blow past max_downloads.
    if link.max_downloads:
        claimed = ShareLink.objects.filter(
            pk=link.pk, download_count__lt=link.max_downloads
        ).update(download_count=F("download_count") + 1)
        if not claimed:
            return render(request, "pdfeditor/share_unavailable.html", {"reason": "exhausted"}, status=410)
    else:
        ShareLink.objects.filter(pk=link.pk).update(download_count=F("download_count") + 1)

    try:
        return FileResponse(
            open(pdf.path, "rb"),
            as_attachment=True,
            filename=os.path.basename(pdf.path),
        )
    except FileNotFoundError as exc:
        raise Http404("PDF missing") from exc


def _owner_filter_for_links(request):
    from django.db.models import Q

    if request.user.is_authenticated:
        return Q(creator=request.user)
    return Q(creator__isnull=True, session_key=request.session.session_key or "")
