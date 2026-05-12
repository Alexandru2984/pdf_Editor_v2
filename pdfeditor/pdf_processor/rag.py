"""RAG (Retrieval-Augmented Generation) primitives: chunking + embeddings.

The model is lazy-loaded the first time we encode anything — fastembed
fetches the ONNX file (~120 MB) on first use and caches it under
``~/.cache/fastembed/``. Subsequent processes share the cache.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Iterable

import fitz

logger = logging.getLogger(__name__)

# Multilingual MiniLM — 384 dims, ~120 MB ONNX, supports 50+ languages
# including Romanian. Good quality/size tradeoff for RAG over user PDFs.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

CHUNK_SIZE = 600  # characters
CHUNK_OVERLAP = 80  # characters

_model_lock = threading.Lock()
_model = None


def _get_model():  # type: ignore[no-untyped-def]
    """Load (once) and return the fastembed text embedding model.

    First call downloads the ONNX weights — subsequent calls return the
    cached instance. Thread-safe via a module-level lock so concurrent
    Celery workers don't race during cold start.
    """
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            from fastembed import TextEmbedding

            cache_dir = os.environ.get("FASTEMBED_CACHE_DIR") or os.path.expanduser("~/.cache/fastembed")
            os.makedirs(cache_dir, exist_ok=True)
            _model = TextEmbedding(model_name=EMBEDDING_MODEL, cache_dir=cache_dir)
            logger.info("Loaded embedding model %s", EMBEDDING_MODEL)
    return _model


_PARA_SPLIT = re.compile(r"\n\s*\n+")


def _split_paragraphs(text: str) -> list[str]:
    """Naive paragraph split on blank lines, stripping whitespace."""
    return [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]


def chunk_pdf(pdf_path: str) -> Iterable[tuple[int, int, str]]:
    """Yield ``(chunk_index, page_number, text)`` tuples from a PDF.

    Each page's text is split on blank lines, then chunked to ~CHUNK_SIZE
    chars with CHUNK_OVERLAP overlap so phrases on chunk boundaries stay
    retrievable. Empty pages (image-only) are skipped — run OCR first to
    make them searchable.
    """
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    idx = 0
    with fitz.open(pdf_path) as doc:
        if doc.is_encrypted:
            raise ValueError("Cannot index an encrypted PDF — remove the password first")

        for page_no, page in enumerate(doc, start=1):
            raw = page.get_text("text") or ""
            if not raw.strip():
                continue
            paragraphs = _split_paragraphs(raw)
            buf = ""
            for para in paragraphs:
                if len(buf) + len(para) + 1 <= CHUNK_SIZE:
                    buf = (buf + "\n\n" + para).strip()
                    continue
                if buf:
                    yield idx, page_no, buf
                    idx += 1
                # Para longer than chunk size — slice with overlap.
                if len(para) > CHUNK_SIZE:
                    start = 0
                    while start < len(para):
                        piece = para[start : start + CHUNK_SIZE]
                        yield idx, page_no, piece
                        idx += 1
                        start += CHUNK_SIZE - CHUNK_OVERLAP
                    buf = ""
                else:
                    # Start a new buffer with this paragraph; carry over
                    # the tail of the previous buf as overlap.
                    tail = buf[-CHUNK_OVERLAP:] if buf else ""
                    buf = (tail + "\n\n" + para).strip()
            if buf:
                yield idx, page_no, buf
                idx += 1


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return one embedding per input text. Texts are processed in batch
    by fastembed for throughput."""
    if not texts:
        return []
    model = _get_model()
    return [vec.tolist() for vec in model.embed(texts)]


def embed_query(text: str) -> list[float]:
    """Single-vector embedding for a query string."""
    return embed_texts([text])[0]
