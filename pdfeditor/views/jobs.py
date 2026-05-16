"""Web views for async job status + listing."""

from __future__ import annotations

import asyncio
import json
import logging
import os

from django.conf import settings
from django.contrib import messages
from django.http import Http404, JsonResponse, StreamingHttpResponse
from django.shortcuts import redirect, render
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from ..models import Job
from ._common import attachment_response, owner_filter

logger = logging.getLogger(__name__)


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
    from django.urls import reverse

    from ..models import ProcessedPDF

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
    # chat_index has no ProcessedPDF output — instead point the user back at
    # the chat page for the source PDF once indexing is done.
    if job.kind == ProcessedPDF.KIND_CHAT_INDEX and job.status == Job.STATUS_DONE and job.source_id:
        payload["follow_up_url"] = reverse("chat", args=[job.source_id])
    return JsonResponse(payload)


def job_download_view(request, job_id):
    job = _user_job(request, job_id)
    if not job.output or not job.output.exists_on_disk():
        messages.error(request, _("Job output not found."))
        raise Http404
    return attachment_response(job.output.path)


@require_POST
def job_cancel_view(request, job_id):
    """POST endpoint hit by the cancel button on the job detail page."""
    from django.urls import reverse

    from ..tasks import cancel_job

    job = _user_job(request, job_id)
    if cancel_job(job):
        messages.success(request, _("Job cancelled."))
    else:
        messages.info(request, _("Job already finished."))
    # If the caller is an HTMX/fetch request expecting JSON, oblige; the
    # default is a redirect back to the detail page so the form-POST flow
    # works without JS.
    if request.headers.get("Accept", "").startswith("application/json"):
        return JsonResponse({"status": job.status, "error_message": job.error_message})
    return redirect(reverse("job_detail", args=[job.id]))


# ---------- Server-Sent Events ----------
#
# The poll-based status endpoint (job_status_view) is still served for
# clients that can't speak SSE (or for tests). job_events_view is the
# live alternative: subscribe to a Redis pub/sub channel that the Celery
# task publishes to on every state change, and forward each message as
# an SSE frame. Heartbeat every 15s so proxies don't kill an idle
# connection, and close as soon as the job hits a terminal state.

_HEARTBEAT_SECONDS = 15
_SSE_HEADERS = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    # Nginx buffers responses by default; this header tells it not to so
    # frames reach the browser as soon as we yield them.
    "X-Accel-Buffering": "no",
}


def _redis_url() -> str | None:
    return (
        getattr(settings, "REDIS_PUBSUB_URL", None)
        or os.environ.get("REDIS_CACHE_URL")
        or os.environ.get("REDIS_URL")
    )


def _job_payload(job: Job) -> dict:
    return {
        "id": str(job.id),
        "status": job.status,
        "progress": job.progress,
        "is_terminal": job.is_terminal(),
        "error_message": job.error_message,
        "output_id": str(job.output_id) if job.output_id else None,
        "output_name": job.output.name if job.output_id else None,
        "output_size": job.output.size if job.output_id else None,
    }


def _sse_frame(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


async def _stream_events(job_id: str, initial: dict):
    """Yield SSE-formatted bytes until the job hits a terminal state or
    the client disconnects."""
    yield _sse_frame(initial)
    if initial.get("is_terminal"):
        return

    redis_url = _redis_url()
    if not redis_url:
        # No Redis configured — the client will fall back to polling.
        return

    try:
        import redis.asyncio as redis_async
    except ImportError:
        return

    client = redis_async.from_url(redis_url)
    pubsub = client.pubsub()
    try:
        await pubsub.subscribe(f"job:{job_id}")
        while True:
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=_HEARTBEAT_SECONDS)
            if msg is None:
                # No data this window — emit a comment frame to keep
                # the connection from being reaped by an idle timeout.
                yield b": heartbeat\n\n"
                continue
            try:
                data = json.loads(msg["data"])
            except (TypeError, ValueError):
                continue
            yield _sse_frame(data)
            if data.get("is_terminal"):
                return
    except asyncio.CancelledError:
        # Client disconnected — let it bubble so Django closes cleanly.
        raise
    except Exception as exc:  # noqa: BLE001 — never crash the connection
        logger.warning("SSE stream error for job %s: %s", job_id, exc)
    finally:
        try:
            await pubsub.unsubscribe(f"job:{job_id}")
            await pubsub.aclose()
            await client.aclose()
        except Exception:  # noqa: BLE001
            pass


async def job_events_view(request, job_id):
    """SSE endpoint: streams job state changes to the browser in real time."""
    user = await request.auser()
    if user.is_authenticated:
        job = await Job.objects.filter(user=user, id=job_id).afirst()
    else:
        session_key = request.session.session_key
        if not session_key:
            raise Http404("Job not found")
        job = await Job.objects.filter(user__isnull=True, session_key=session_key, id=job_id).afirst()
    if job is None:
        raise Http404("Job not found")
    initial = _job_payload(job)
    response = StreamingHttpResponse(_stream_events(str(job_id), initial))
    for header, value in _SSE_HEADERS.items():
        response[header] = value
    return response


_UNUSED = os
