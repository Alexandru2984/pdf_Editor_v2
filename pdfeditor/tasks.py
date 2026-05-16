"""Celery tasks for long-running PDF operations.

Each task takes a ``job_id`` (UUID string of a ``Job`` row), loads the job,
marks it running, executes the underlying op, and links the resulting
``ProcessedPDF`` row on success — or stores a friendly error on failure.
"""

from __future__ import annotations

import json
import logging
import os

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from django.conf import settings
from django.utils import timezone

from .metrics import EMBEDDINGS_CREATED, JOB_TOTAL, OP_DURATION_SECONDS, OP_TOTAL
from .models import Job, ProcessedPDF, UploadedPDF

logger = logging.getLogger(__name__)

# Whitelist-style validation for params that arrive via job.params (a JSONField
# populated from forms or the API). Forms already restrict these to known
# choices, but the API path lets clients send raw JSON — re-clamping here
# prevents a hostile payload from coaxing the worker into a high-DPI render
# (OOM) or an invalid tesseract language string.
_ALLOWED_OCR_LANGS = {"eng", "ron", "eng+ron"}
_ALLOWED_PDFA_VERSIONS = {"1b", "2b", "3b"}
_ALLOWED_IMAGE_FORMATS = {"png", "jpg"}
_MIN_DPI = 72
_MAX_DPI = 600
# Hard ceiling on chunk count for chat indexing. A 500-page PDF of dense text
# yields ~5k chunks; anything above this is almost certainly a "word-soup" PDF
# crafted to exhaust the embedding pool.
_MAX_CHAT_CHUNKS = 5000


def _clamp_dpi(raw: object, default: int) -> int:
    if isinstance(raw, (int, float, str)) and not isinstance(raw, bool):
        try:
            dpi = int(raw)
        except (TypeError, ValueError):
            dpi = default
    else:
        dpi = default
    return max(_MIN_DPI, min(dpi, _MAX_DPI))


def _check_choice(raw: object, allowed: set[str], param_label: str) -> str:
    """Strict whitelist check. Raises ValueError so _safe_run records a
    helpful error_message — silent fallback would mask client misuse."""
    if not isinstance(raw, str) or raw not in allowed:
        raise ValueError(f"Invalid {param_label}: {raw!r}. Allowed: {sorted(allowed)}")
    return raw


def _job_channel(job_id) -> str:
    return f"job:{job_id}"


def _publish_job_event(job: Job) -> None:
    """Broadcast the job's current state to anyone subscribed via SSE.

    Best-effort: if Redis is unavailable or this Django install isn't
    configured with a Redis cache backend, we just skip — the browser
    will fall back to HTTP polling.
    """
    redis_url = (
        getattr(settings, "REDIS_PUBSUB_URL", None)
        or os.environ.get("REDIS_CACHE_URL")
        or os.environ.get("REDIS_URL")
    )
    if not redis_url:
        return
    payload = {
        "id": str(job.id),
        "status": job.status,
        "progress": job.progress,
        "is_terminal": job.is_terminal(),
        "error_message": job.error_message,
        "output_id": str(job.output_id) if job.output_id else None,
        "output_name": job.output.name if job.output_id else None,
        "output_size": job.output.size if job.output_id else None,
    }
    try:
        import redis as _redis

        client = _redis.Redis.from_url(redis_url, socket_timeout=2)
        client.publish(_job_channel(job.id), json.dumps(payload))
    except Exception as exc:  # noqa: BLE001 — pub/sub is best-effort
        logger.debug("Job event publish failed for %s: %s", job.id, exc)


def _start(job: Job) -> None:
    job.status = Job.STATUS_RUNNING
    job.started_at = timezone.now()
    job.progress = 5
    job.save(update_fields=["status", "started_at", "progress"])
    _publish_job_event(job)


def _record_output(job: Job, kind: str, out_path: str) -> ProcessedPDF:
    output = ProcessedPDF.objects.create(
        user=job.user,
        session_key=job.session_key,
        kind=kind,
        source=job.source,
        name=os.path.basename(out_path),
        path=out_path,
        size=os.path.getsize(out_path) if os.path.exists(out_path) else 0,
    )
    job.output = output
    job.status = Job.STATUS_DONE
    job.progress = 100
    job.finished_at = timezone.now()
    job.save(update_fields=["output", "status", "progress", "finished_at"])
    JOB_TOTAL.labels(kind=kind, status="done").inc()
    if job.started_at:
        OP_DURATION_SECONDS.labels(kind=kind).observe((job.finished_at - job.started_at).total_seconds())
    OP_TOTAL.labels(kind=kind, outcome="success").inc()
    _publish_job_event(job)
    return output


def _fail(job: Job, error: str) -> None:
    job.status = Job.STATUS_FAILED
    job.error_message = error[:500]
    job.finished_at = timezone.now()
    job.save(update_fields=["status", "error_message", "finished_at"])
    JOB_TOTAL.labels(kind=job.kind, status="failed").inc()
    OP_TOTAL.labels(kind=job.kind, outcome="failure").inc()
    _publish_job_event(job)


def _safe_run(job_id: str, kind: str, run):
    """Common wrapper: load job, mark running, call run(job) → out_path,
    record output, swallow errors into job.error_message."""
    try:
        job = Job.objects.select_related("source", "second_source").get(id=job_id)
    except Job.DoesNotExist:
        logger.warning("Celery task fired for missing job %s", job_id)
        return None

    if not job.source or not job.source.exists_on_disk():
        _fail(job, "Source PDF is missing on disk.")
        return None

    _start(job)
    try:
        out_path = run(job)
    except SoftTimeLimitExceeded:
        _fail(job, "Operation timed out.")
        return None
    except ValueError as exc:
        _fail(job, str(exc))
        return None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Async %s task failed for job %s", kind, job_id)
        _fail(job, f"Internal error: {exc.__class__.__name__}")
        return None

    output = _record_output(job, kind, out_path)
    return str(output.id)


@shared_task(name="pdfeditor.run_ocr")
def run_ocr_task(job_id: str) -> str | None:
    from .pdf_processor import make_pdf_searchable

    def run(job: Job) -> str:
        params = job.params or {}
        out, _ = make_pdf_searchable(
            job.source.path,
            language=_check_choice(params.get("language", "eng+ron"), _ALLOWED_OCR_LANGS, "OCR language"),
            dpi=_clamp_dpi(params.get("dpi"), 200),
        )
        return out

    return _safe_run(job_id, ProcessedPDF.KIND_OCR_LAYER, run)


@shared_task(name="pdfeditor.run_pdfa")
def run_pdfa_task(job_id: str) -> str | None:
    from .pdf_processor import convert_to_pdfa

    def run(job: Job) -> str:
        params = job.params or {}
        out, _ = convert_to_pdfa(
            job.source.path,
            version=_check_choice(params.get("version", "2b"), _ALLOWED_PDFA_VERSIONS, "PDF/A version"),
        )
        return out

    return _safe_run(job_id, ProcessedPDF.KIND_PDFA, run)


@shared_task(name="pdfeditor.run_compare")
def run_compare_task(job_id: str) -> str | None:
    from .pdf_processor import compare_pdfs

    def run(job: Job) -> str:
        if not job.second_source or not job.second_source.exists_on_disk():
            raise ValueError("Revised PDF is missing on disk.")
        out, stats = compare_pdfs(job.source.path, job.second_source.path)
        # Stash stats on the job so the result page can display them.
        merged = dict(job.params or {})
        merged["stats"] = stats
        Job.objects.filter(pk=job.pk).update(params=merged)
        return out

    return _safe_run(job_id, ProcessedPDF.KIND_COMPARE, run)


@shared_task(name="pdfeditor.run_convert_docx")
def run_convert_docx_task(job_id: str) -> str | None:
    from .pdf_processor import convert_pdf_to_docx

    def run(job: Job) -> str:
        out, _ = convert_pdf_to_docx(job.source.path)
        return out

    return _safe_run(job_id, ProcessedPDF.KIND_CONVERT, run)


@shared_task(name="pdfeditor.run_to_images")
def run_to_images_task(job_id: str) -> str | None:
    """Rasterize every PDF page to PNG/JPG and ZIP the result.

    Heaviest CPU op in the catalog at high DPI — load testing put p95 at
    ~3.8s sync, which monopolizes a gunicorn worker. Async dispatch keeps
    the request-serving pool free, and the progress_cb lets us push live
    per-page updates so the SSE stream actually has something to say.
    """
    from .pdf_processor import convert_pdf_to_images

    def run(job: Job) -> str:
        params = job.params or {}
        last_pct = {"v": 5}  # _start set progress=5, don't regress below it

        def on_page(rendered: int, total: int) -> None:
            # 90 % of the bar is rendering; the last 10 % covers zip-close +
            # ProcessedPDF row creation in _record_output (which jumps to 100).
            pct = 5 + int((rendered / max(total, 1)) * 90)
            if pct <= last_pct["v"]:
                return
            last_pct["v"] = pct
            job.progress = pct
            # update_fields=["progress"] keeps this write cheap. The save
            # call itself triggers _publish_job_event only if we add it —
            # we go through the same publish path so the SSE subscriber
            # sees the new percent.
            Job.objects.filter(pk=job.pk).update(progress=pct)
            _publish_job_event(job)

        out, page_count = convert_pdf_to_images(
            job.source.path,
            fmt=_check_choice(params.get("fmt", "png"), _ALLOWED_IMAGE_FORMATS, "image format"),
            dpi=_clamp_dpi(params.get("dpi"), 150),
            progress_cb=on_page,
        )
        # Stash page_count on the job so the result template can show it.
        merged = dict(job.params or {})
        merged["page_count"] = page_count
        Job.objects.filter(pk=job.pk).update(params=merged)
        return out

    return _safe_run(job_id, ProcessedPDF.KIND_TO_IMAGES, run)


@shared_task(name="pdfeditor.run_chat_index")
def run_chat_index_task(job_id: str) -> str | None:
    """Chunk + embed a PDF for RAG chat. Doesn't produce a ProcessedPDF —
    it populates ``Embedding`` rows linked to the source UploadedPDF."""
    from .models import Embedding
    from .pdf_processor.rag import chunk_pdf, embed_texts

    try:
        job = Job.objects.select_related("source").get(id=job_id)
    except Job.DoesNotExist:
        logger.warning("chat_index fired for missing job %s", job_id)
        return None
    if not job.source or not job.source.exists_on_disk():
        _fail(job, "Source PDF is missing on disk.")
        return None

    _start(job)
    try:
        chunks = list(chunk_pdf(job.source.path))
        if not chunks:
            _fail(job, "No extractable text — run OCR first to make this PDF searchable.")
            return None
        if len(chunks) > _MAX_CHAT_CHUNKS:
            _fail(
                job,
                f"PDF has too many text segments ({len(chunks)} > {_MAX_CHAT_CHUNKS}). "
                "Try a shorter document.",
            )
            return None

        # Wipe any prior embeddings for this PDF so reindex is idempotent.
        Embedding.objects.filter(uploaded_pdf=job.source).delete()

        texts = [c[2] for c in chunks]
        vectors = embed_texts(texts)
        Embedding.objects.bulk_create(
            [
                Embedding(
                    uploaded_pdf=job.source,
                    chunk_index=idx,
                    page_number=page,
                    chunk_text=text,
                    embedding=vec,
                )
                for (idx, page, text), vec in zip(chunks, vectors, strict=True)
            ],
            batch_size=200,
        )
        EMBEDDINGS_CREATED.inc(len(chunks))
    except SoftTimeLimitExceeded:
        _fail(job, "Indexing timed out.")
        return None
    except ValueError as exc:
        _fail(job, str(exc))
        return None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat index failed for job %s", job_id)
        _fail(job, f"Internal error: {exc.__class__.__name__}")
        return None

    # No ProcessedPDF for indexing — just mark done with progress=100.
    merged = dict(job.params or {})
    merged["chunk_count"] = len(chunks)
    job.params = merged
    job.status = Job.STATUS_DONE
    job.progress = 100
    from django.utils import timezone as _tz

    job.finished_at = _tz.now()
    job.save(update_fields=["params", "status", "progress", "finished_at"])
    return str(job.id)


# Map kind → task callable. Helps views enqueue without importing each task.
KIND_TO_TASK = {
    ProcessedPDF.KIND_OCR_LAYER: run_ocr_task,
    ProcessedPDF.KIND_PDFA: run_pdfa_task,
    ProcessedPDF.KIND_COMPARE: run_compare_task,
    ProcessedPDF.KIND_CONVERT: run_convert_docx_task,
    ProcessedPDF.KIND_CHAT_INDEX: run_chat_index_task,
    ProcessedPDF.KIND_TO_IMAGES: run_to_images_task,
}


def enqueue_job(job: Job) -> None:
    """Look up the task for the job's kind and dispatch it. Raises KeyError
    if the kind isn't async-enabled. Persists the Celery task id on the
    row so the cancel endpoint can revoke it later."""
    task = KIND_TO_TASK[job.kind]
    result = task.delay(str(job.id))
    if getattr(result, "id", None):
        Job.objects.filter(pk=job.pk).update(celery_task_id=result.id)
        job.celery_task_id = result.id


def cancel_job(job: Job) -> bool:
    """Cancel an in-flight Job. Idempotent — calling on a terminal job
    is a no-op (returns False). Returns True iff this call moved the job
    into a terminal state."""
    if job.is_terminal():
        return False

    if job.celery_task_id:
        # Revoke broadcasts to all workers; ``terminate=True`` SIGTERMs the
        # process if the task already started. Imported lazily so the
        # module still works in environments where the broker is absent
        # (e.g. unit tests with CELERY_TASK_ALWAYS_EAGER).
        try:
            from pdf_project.celery import app as celery_app

            celery_app.control.revoke(job.celery_task_id, terminate=True, signal="SIGTERM")
        except Exception as exc:
            logger.warning("Job %s revoke failed: %s — marking failed anyway", job.id, exc)

    job.status = Job.STATUS_FAILED
    job.error_message = "Cancelled by user"
    job.finished_at = timezone.now()
    job.save(update_fields=["status", "error_message", "finished_at"])
    JOB_TOTAL.labels(kind=job.kind, status="failed").inc()
    _publish_job_event(job)
    return True


# Use UploadedPDF in a no-op import-time reference so static analysers see
# the symbol — Django apps may load this module before models are ready and
# we want the import to fail loudly if the schema diverges.
_UPLOAD_GUARD = UploadedPDF
