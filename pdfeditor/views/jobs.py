"""Web views for async job status + listing."""

from __future__ import annotations

import os

from django.contrib import messages
from django.http import Http404, JsonResponse
from django.shortcuts import render
from django.utils.translation import gettext as _

from ..models import Job
from ._common import attachment_response, owner_filter


def _user_job(request, job_id) -> Job:
    job = Job.objects.filter(owner_filter(request), id=job_id).first()
    if job is None:
        raise Http404("Job not found")
    return job


def jobs_list_view(request):
    jobs = (
        Job.objects.filter(owner_filter(request))
        .select_related("source", "output")
        .order_by("-created_at")[:100]
    )
    return render(request, "pdfeditor/jobs_list.html", {"jobs": jobs})


def job_detail_view(request, job_id):
    job = _user_job(request, job_id)
    return render(
        request,
        "pdfeditor/job_detail.html",
        {"job": job},
    )


def job_status_view(request, job_id):
    """Lightweight JSON polled by the status page every couple of seconds."""
    job = _user_job(request, job_id)
    payload = {
        "id": str(job.id),
        "kind": job.kind,
        "status": job.status,
        "progress": job.progress,
        "is_terminal": job.is_terminal(),
        "error_message": job.error_message,
        "output_id": str(job.output_id) if job.output_id else None,
        "output_name": job.output.name if job.output_id else None,
        "output_size": job.output.size if job.output_id else None,
    }
    if job.kind == "compare" and (job.params or {}).get("stats"):
        payload["stats"] = job.params["stats"]
    return JsonResponse(payload)


def job_download_view(request, job_id):
    job = _user_job(request, job_id)
    if not job.output or not job.output.exists_on_disk():
        messages.error(request, _("Job output not found."))
        raise Http404
    return attachment_response(job.output.path)


_UNUSED = os
