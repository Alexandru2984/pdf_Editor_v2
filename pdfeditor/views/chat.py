"""Chat-with-PDF (RAG) views.

The flow:
  1. User clicks "Chat" on an uploaded PDF.
  2. If the PDF has no embeddings yet, kick off an indexing Job and route
     the user to its status page; they come back once it's done.
  3. Chat page renders. The browser POSTs each user message to
     ``chat_message_view``, which embeds the query, retrieves the top-K
     similar chunks via pgvector cosine distance, sends them to Groq with
     a RAG system prompt, and returns the answer + citations.
"""

from __future__ import annotations

import json
import logging
import os

import requests
from django.conf import settings
from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.utils.translation import gettext as _
from django.views.decorators.http import require_http_methods
from pgvector.django import CosineDistance

from ..models import Embedding, Job, ProcessedPDF, UploadedPDF
from ..ratelimiting import auth_aware_ratelimit
from ._common import owner_filter
from .basic_ops import _queue_async_job

logger = logging.getLogger(__name__)

TOP_K = 5
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = os.environ.get("CHAT_GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = (
    "You are an assistant that answers questions strictly from the user's PDF. "
    "You will receive numbered excerpts from the document. Cite each fact you "
    "use by its excerpt number in square brackets, like [2]. If the answer is "
    "not in the excerpts, say so clearly — do not invent. Answer in the same "
    "language as the user's question."
)


def _resolve_pdf(request, pdf_id) -> UploadedPDF | None:
    return UploadedPDF.objects.filter(owner_filter(request), id=pdf_id).first()


# Curated list of Groq models — limited to text-LLMs that handle multilingual
# input well and aren't prompt-guard / vision variants. (id, friendly_label)
AVAILABLE_LLMS = [
    ("llama-3.3-70b-versatile", "Llama 3.3 70B (balanced, default)"),
    ("llama-3.1-8b-instant", "Llama 3.1 8B (fastest, less accurate)"),
    ("meta-llama/llama-4-scout-17b-16e-instruct", "Llama 4 Scout 17B (modern MoE)"),
    ("openai/gpt-oss-120b", "GPT-OSS 120B (largest, slowest)"),
]
_VALID_MODEL_IDS = {m[0] for m in AVAILABLE_LLMS}


def chat_view(request, pdf_id):
    """Chat over one or many PDFs.

    The path param ``pdf_id`` is the "anchor" — if no embeddings exist for
    it, we queue an indexing job and redirect there. Extra PDFs can join
    the context via ``?pdf=<id>&pdf=<id>...`` query params; each must be
    owned by the requester AND already indexed (we silently drop the rest).
    """
    anchor = _resolve_pdf(request, pdf_id)
    if anchor is None:
        messages.error(request, _("PDF not found."))
        return redirect("dashboard")

    has_index = Embedding.objects.filter(uploaded_pdf=anchor).exists()
    if not has_index:
        # Already a pending index job? Don't spam new ones — link to it.
        pending = Job.objects.filter(
            owner_filter(request),
            kind=ProcessedPDF.KIND_CHAT_INDEX,
            source=anchor,
            status__in=[Job.STATUS_QUEUED, Job.STATUS_RUNNING],
        ).first()
        if pending:
            return redirect("job_detail", job_id=pending.id)
        job = _queue_async_job(request, kind=ProcessedPDF.KIND_CHAT_INDEX, source=anchor)
        messages.info(request, _("Indexing this PDF for chat. Takes a few seconds — we'll bring you back."))
        return redirect("job_detail", job_id=job.id)

    # Build the list of additional PDFs the user has + which are indexed.
    user_pdfs = list(UploadedPDF.objects.filter(owner_filter(request)).order_by("-uploaded_at"))
    indexed_ids = set(
        Embedding.objects.filter(uploaded_pdf__in=user_pdfs)
        .values_list("uploaded_pdf_id", flat=True)
        .distinct()
    )

    # Selected extras from query string — sanitize to owned + indexed.
    requested_extras = request.GET.getlist("pdf")
    extra_ids: list = []
    for raw in requested_extras:
        try:
            from uuid import UUID

            uid = UUID(raw)
        except (ValueError, TypeError):
            continue
        if uid != anchor.id and uid in indexed_ids:
            extra_ids.append(uid)

    selected_ids = [anchor.id, *extra_ids]
    chunk_count = Embedding.objects.filter(uploaded_pdf_id__in=selected_ids).count()

    pdf_options = [
        {
            "id": str(p.id),
            "name": p.name,
            "indexed": p.id in indexed_ids,
            "is_anchor": p.id == anchor.id,
            "selected": p.id in selected_ids,
        }
        for p in user_pdfs
    ]

    return render(
        request,
        "pdfeditor/chat.html",
        {
            "pdf": anchor,
            "selected_ids": [str(i) for i in selected_ids],
            "selected_count": len(selected_ids),
            "chunk_count": chunk_count,
            "pdf_options": pdf_options,
            "available_llms": AVAILABLE_LLMS,
            "default_llm": request.session.get("chat_llm", AVAILABLE_LLMS[0][0]),
        },
    )


def _retrieve(pdf_ids, query_vector: list[float], k: int = TOP_K):
    """Top-K most-similar chunks across ANY of the given PDF ids.

    Accepts either a single UploadedPDF or a list/iterable of PDF ids.
    Cosine distance — lower is better."""
    if hasattr(pdf_ids, "id"):  # single UploadedPDF object
        pdf_ids = [pdf_ids.id]
    return list(
        Embedding.objects.filter(uploaded_pdf_id__in=list(pdf_ids))
        .select_related("uploaded_pdf")
        .annotate(distance=CosineDistance("embedding", query_vector))
        .order_by("distance")[:k]
    )


def _build_rag_prompt(question: str, chunks, multi_doc: bool = False) -> list[dict]:
    if multi_doc:
        excerpts = "\n\n".join(
            f"[{i + 1}] (document: {c.uploaded_pdf.name}, page {c.page_number})\n{c.chunk_text}"
            for i, c in enumerate(chunks)
        )
    else:
        excerpts = "\n\n".join(
            f"[{i + 1}] (page {c.page_number})\n{c.chunk_text}" for i, c in enumerate(chunks)
        )
    user_msg = f"Document excerpts:\n\n{excerpts}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def _call_groq(messages_payload: list[dict], model: str | None = None) -> tuple[str, str | None]:
    """Returns (answer, error). On success error is None."""
    api_key = getattr(settings, "GROQ_API_KEY", "") or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return "", "GROQ_API_KEY is not set on the server."
    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model or DEFAULT_GROQ_MODEL,
                "messages": messages_payload,
                "temperature": 0.2,
                "max_tokens": 800,
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        return "", f"Network error talking to Groq: {exc}"

    if resp.status_code != 200:
        logger.warning("Groq returned %s: %s", resp.status_code, resp.text[:200])
        return "", f"Groq error {resp.status_code}"
    try:
        return resp.json()["choices"][0]["message"]["content"].strip(), None
    except (KeyError, ValueError, IndexError) as exc:
        return "", f"Malformed Groq response: {exc}"


@auth_aware_ratelimit(anon_rate="10/h", user_rate="60/h", method="POST")
@require_http_methods(["POST"])
def start_index_view(request, pdf_id):
    """AJAX endpoint — start indexing a PDF without navigating away from
    the chat page. Returns the job_id; the frontend polls
    ``/jobs/<id>/status/`` and refreshes the checkbox row when done.

    Rate-limited because each call enqueues a Celery job that loads the
    embedding model and processes the full PDF — expensive enough that
    a tight cap on anonymous traffic is worth it.
    """
    pdf = _resolve_pdf(request, pdf_id)
    if pdf is None:
        return JsonResponse({"error": "PDF not found."}, status=404)

    if Embedding.objects.filter(uploaded_pdf=pdf).exists():
        return JsonResponse({"already_indexed": True, "pdf_id": str(pdf.id)})

    # Reuse a pending/running job if one exists.
    existing = Job.objects.filter(
        owner_filter(request),
        kind=ProcessedPDF.KIND_CHAT_INDEX,
        source=pdf,
        status__in=[Job.STATUS_QUEUED, Job.STATUS_RUNNING],
    ).first()
    if existing:
        return JsonResponse({"job_id": str(existing.id), "status": existing.status})

    job = _queue_async_job(request, kind=ProcessedPDF.KIND_CHAT_INDEX, source=pdf)
    return JsonResponse({"job_id": str(job.id), "status": job.status})


@auth_aware_ratelimit(anon_rate="10/h", user_rate="100/h", method="POST")
@require_http_methods(["POST"])
def chat_message_view(request, pdf_id):
    anchor = _resolve_pdf(request, pdf_id)
    if anchor is None:
        return JsonResponse({"error": "PDF not found."}, status=404)

    try:
        payload = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON."}, status=400)

    question = (payload.get("message") or "").strip()
    if not question:
        return JsonResponse({"error": "Message is required."}, status=400)
    if len(question) > 2000:
        return JsonResponse({"error": "Message too long (2000 chars max)."}, status=400)

    # Validate the LLM choice (or fall back to the default). Persisted in
    # session so the user's pick survives across messages.
    model = (payload.get("model") or "").strip()
    if model and model not in _VALID_MODEL_IDS:
        return JsonResponse({"error": "Unknown model."}, status=400)
    if not model:
        model = request.session.get("chat_llm", DEFAULT_GROQ_MODEL)
    request.session["chat_llm"] = model

    # Resolve extra PDFs from the request — same scoping rules as the page.
    extra_raw = payload.get("extra_pdf_ids") or []
    if not isinstance(extra_raw, list):
        return JsonResponse({"error": "extra_pdf_ids must be a list."}, status=400)
    indexed_ids = set(
        Embedding.objects.filter(uploaded_pdf__in=UploadedPDF.objects.filter(owner_filter(request)))
        .values_list("uploaded_pdf_id", flat=True)
        .distinct()
    )
    selected_ids = [anchor.id]
    for raw in extra_raw:
        try:
            from uuid import UUID

            uid = UUID(str(raw))
        except (ValueError, TypeError):
            continue
        if uid != anchor.id and uid in indexed_ids and uid not in selected_ids:
            selected_ids.append(uid)

    if anchor.id not in indexed_ids:
        return JsonResponse({"error": "PDF not indexed yet — open it from the chat page first."}, status=409)

    # Embed query + retrieve top-K chunks across all selected PDFs.
    try:
        from ..pdf_processor.rag import embed_query

        qvec = embed_query(question)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query embedding failed")
        return JsonResponse({"error": f"Embedding failed: {exc.__class__.__name__}"}, status=500)

    chunks = _retrieve(selected_ids, qvec)
    multi = len(selected_ids) > 1
    answer, error = _call_groq(_build_rag_prompt(question, chunks, multi_doc=multi), model=model)
    if error:
        return JsonResponse({"error": error}, status=502)

    citations = [
        {
            "index": i + 1,
            "page": c.page_number,
            "document": c.uploaded_pdf.name if multi else None,
            "excerpt": c.chunk_text[:300] + ("…" if len(c.chunk_text) > 300 else ""),
        }
        for i, c in enumerate(chunks)
    ]
    return JsonResponse({"answer": answer, "citations": citations, "model": model})
