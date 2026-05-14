"""Celery tasks for long-running PDF operations.

Each task takes a ``job_id`` (UUID string of a ``Job`` row), loads the job,
marks it running, executes the underlying op, and links the resulting
``ProcessedPDF`` row on success — or stores a friendly error on failure.
"""

from __future__ import annotations

import logging
import os

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from django.utils import timezone

from .metrics import EMBEDDINGS_CREATED, JOB_TOTAL, OP_DURATION_SECONDS, OP_TOTAL
from .models import Job, ProcessedPDF, UploadedPDF

logger = logging.getLogger(__name__)


def _start(job: Job) -> None:
    job.status = Job.STATUS_RUNNING
    job.started_at = timezone.now()
    job.progress = 5
    job.save(update_fields=["status", "started_at", "progress"])


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
    return output


def _fail(job: Job, error: str) -> None:
    job.status = Job.STATUS_FAILED
    job.error_message = error[:500]
    job.finished_at = timezone.now()
    job.save(update_fields=["status", "error_message", "finished_at"])
    JOB_TOTAL.labels(kind=job.kind, status="failed").inc()
    OP_TOTAL.labels(kind=job.kind, outcome="failure").inc()


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
            language=params.get("language", "eng+ron"),
            dpi=int(params.get("dpi", 200)),
        )
        return out

    return _safe_run(job_id, ProcessedPDF.KIND_OCR_LAYER, run)


@shared_task(name="pdfeditor.run_pdfa")
def run_pdfa_task(job_id: str) -> str | None:
    from .pdf_processor import convert_to_pdfa

    def run(job: Job) -> str:
        params = job.params or {}
        out, _ = convert_to_pdfa(job.source.path, version=params.get("version", "2b"))
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
    the request-serving pool free.
    """
    from .pdf_processor import convert_pdf_to_images

    def run(job: Job) -> str:
        params = job.params or {}
        out, page_count = convert_pdf_to_images(
            job.source.path,
            fmt=params.get("fmt", "png"),
            dpi=int(params.get("dpi", 150)),
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
    if the kind isn't async-enabled."""
    task = KIND_TO_TASK[job.kind]
    task.delay(str(job.id))


# Use UploadedPDF in a no-op import-time reference so static analysers see
# the symbol — Django apps may load this module before models are ready and
# we want the import to fail loudly if the schema diverges.
_UPLOAD_GUARD = UploadedPDF
