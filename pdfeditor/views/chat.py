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


def chat_view(request, pdf_id):
    pdf = _resolve_pdf(request, pdf_id)
    if pdf is None:
        messages.error(request, _("PDF not found."))
        return redirect("dashboard")

    has_index = Embedding.objects.filter(uploaded_pdf=pdf).exists()
    if not has_index:
        # Already a pending index job? Don't spam new ones — link to it.
        pending = Job.objects.filter(
            owner_filter(request),
            kind=ProcessedPDF.KIND_CHAT_INDEX,
            source=pdf,
            status__in=[Job.STATUS_QUEUED, Job.STATUS_RUNNING],
        ).first()
        if pending:
            return redirect("job_detail", job_id=pending.id)
        job = _queue_async_job(request, kind=ProcessedPDF.KIND_CHAT_INDEX, source=pdf)
        messages.info(request, _("Indexing this PDF for chat. Takes a few seconds — we'll bring you back."))
        return redirect("job_detail", job_id=job.id)

    return render(
        request,
        "pdfeditor/chat.html",
        {
            "pdf": pdf,
            "chunk_count": Embedding.objects.filter(uploaded_pdf=pdf).count(),
        },
    )


def _retrieve(pdf: UploadedPDF, query_vector: list[float], k: int = TOP_K):
    """Top-K most-similar chunks by cosine distance (lower = better)."""
    return list(
        Embedding.objects.filter(uploaded_pdf=pdf)
        .annotate(distance=CosineDistance("embedding", query_vector))
        .order_by("distance")[:k]
    )


def _build_rag_prompt(question: str, chunks) -> list[dict]:
    excerpts = "\n\n".join(f"[{i + 1}] (page {c.page_number})\n{c.chunk_text}" for i, c in enumerate(chunks))
    user_msg = f"Document excerpts:\n\n{excerpts}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def _call_groq(messages_payload: list[dict]) -> tuple[str, str | None]:
    """Returns (answer, error). On success error is None."""
    api_key = getattr(settings, "GROQ_API_KEY", "") or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return "", "GROQ_API_KEY is not set on the server."
    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": DEFAULT_GROQ_MODEL,
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


@auth_aware_ratelimit(anon_rate="10/h", user_rate="100/h", method="POST")
@require_http_methods(["POST"])
def chat_message_view(request, pdf_id):
    pdf = _resolve_pdf(request, pdf_id)
    if pdf is None:
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

    if not Embedding.objects.filter(uploaded_pdf=pdf).exists():
        return JsonResponse({"error": "PDF not indexed yet — open it from the chat page first."}, status=409)

    # Embed query + retrieve top-K chunks.
    try:
        from ..pdf_processor.rag import embed_query

        qvec = embed_query(question)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query embedding failed")
        return JsonResponse({"error": f"Embedding failed: {exc.__class__.__name__}"}, status=500)

    chunks = _retrieve(pdf, qvec)
    answer, error = _call_groq(_build_rag_prompt(question, chunks))
    if error:
        return JsonResponse({"error": error}, status=502)

    citations = [
        {
            "index": i + 1,
            "page": c.page_number,
            "excerpt": c.chunk_text[:300] + ("…" if len(c.chunk_text) > 300 else ""),
        }
        for i, c in enumerate(chunks)
    ]
    return JsonResponse({"answer": answer, "citations": citations})
