"""Registry of sync-only PDF ops that the batch endpoint can dispatch.

Each entry maps a public op name to:

* the ``ProcessedPDF.kind`` to write on successful outputs, and
* a runner ``(pdf_path, params) -> out_path``.

Async-only ops (OCR layer, PDF/A, compare, convert-docx, to-images) are
intentionally excluded — they already run as standalone jobs, batching
them would just be a job spawning jobs. Multi-input ops (merge) are
excluded too since the batch contract is "one op applied to each PDF".
Password-bearing ops are also excluded because batch params are stored on
the async Job row.
"""

from __future__ import annotations

from collections.abc import Callable

from ..models import ProcessedPDF
from ..pdf_processor import (
    add_page_numbers,
    add_watermark,
    compress_pdf,
    edit_pdf_metadata,
    flatten_pdf,
    redact_text,
    rotate_pages,
)

OpRunner = Callable[[str, dict], str]


def _compress(pdf_path: str, params: dict) -> str:
    out, *_ = compress_pdf(pdf_path, quality=params.get("quality", "medium"))
    return out


def _rotate(pdf_path: str, params: dict) -> str:
    return rotate_pages(
        pdf_path,
        rotation_angle=params["rotation_angle"],
        page_range=params.get("page_range") or None,
    )


def _watermark(pdf_path: str, params: dict) -> str:
    options = {k: params[k] for k in ("position", "opacity", "rotation", "font_size") if k in params}
    return add_watermark(
        pdf_path,
        watermark_type=params.get("watermark_type", "text"),
        watermark_content=params["text"],
        options=options or None,
    )


def _flatten(pdf_path: str, _params: dict) -> str:
    return flatten_pdf(pdf_path)


def _page_numbers(pdf_path: str, params: dict) -> str:
    options = {k: params[k] for k in ("position", "format", "font_size", "start_page") if k in params}
    return add_page_numbers(pdf_path, options=options or None)


def _metadata(pdf_path: str, params: dict) -> str:
    return edit_pdf_metadata(pdf_path, params)


def _redact(pdf_path: str, params: dict) -> str:
    out, _ = redact_text(
        pdf_path,
        params["search_terms"],
        page_range=params.get("page_range") or None,
    )
    return out


BATCH_OPS: dict[str, tuple[str, OpRunner]] = {
    "compress": (ProcessedPDF.KIND_COMPRESS, _compress),
    "rotate": (ProcessedPDF.KIND_ROTATE, _rotate),
    "watermark": (ProcessedPDF.KIND_WATERMARK, _watermark),
    "flatten": (ProcessedPDF.KIND_FLATTEN, _flatten),
    "page-numbers": (ProcessedPDF.KIND_PAGE_NUMBERS, _page_numbers),
    "metadata": (ProcessedPDF.KIND_METADATA, _metadata),
    "redact": (ProcessedPDF.KIND_REDACT, _redact),
}

#: Cap so a single user can't fan out an unbounded amount of work in one
#: request. Picked to match the upload quota for typical users while
#: still letting bulk archives through.
MAX_BATCH_SIZE = 50
