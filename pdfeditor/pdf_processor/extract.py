"""Text extraction: direct text layer + OCR fallback.

OCR is parallelized across pages with a process pool because tesseract is
CPU-bound — a 20-page scan that took ~30s sequentially completes in ~7s
with 4 workers. Pages that already have an extractable text layer skip
the rasterise+OCR entirely (huge win on mixed-content PDFs).
"""

from __future__ import annotations

import io
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import fitz

logger = logging.getLogger(__name__)

# 200dpi is enough for body text at A4 — readable by tesseract while ~2.25x
# faster to render than 300dpi. Override per environment if needed.
OCR_DPI = int(os.environ.get("OCR_DPI", "200"))

# tesseract loads each language model once; "eng+ron" handles the bulk of
# this app's audience. Override with OCR_LANG if you only need one.
OCR_LANG = os.environ.get("OCR_LANG", "eng+ron")

# 0 = pick automatically, capped at 4 so we don't starve a shared VPS.
OCR_MAX_WORKERS = int(os.environ.get("OCR_MAX_WORKERS", "0"))


def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    with fitz.open(pdf_path) as doc:
        chunks: list[str] = []
        for i, page in enumerate(doc, 1):
            t = page.get_text()
            if t.strip():
                chunks.append(f"=== Page {i} ===\n{t}\n")

    if not chunks:
        return "No text found in PDF. This might be a scanned document - try OCR instead."

    return "\n".join(chunks)


def _ocr_one_page(args: tuple[str, int, int, str]) -> tuple[int, str]:
    """Worker entry point. Runs in a separate process — must be picklable.

    Re-opens the PDF in the child rather than passing pixmap bytes, since
    PyMuPDF rasterisation is itself CPU-bound and benefits from running in
    parallel with the OCR call.
    """
    pdf_path, page_index, dpi, lang = args
    import pytesseract
    from PIL import Image

    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        layer_text = page.get_text()
        if layer_text.strip():
            return page_index, layer_text
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return page_index, pytesseract.image_to_string(img, lang=lang)


def _resolve_worker_count(page_count: int) -> int:
    if OCR_MAX_WORKERS > 0:
        target = OCR_MAX_WORKERS
    else:
        target = min(4, max(1, os.cpu_count() or 1))
    return max(1, min(target, page_count))


def ocr_pdf_to_text(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    try:
        import pytesseract  # noqa: F401
    except ImportError as e:
        raise Exception("pytesseract not installed. Run: pip install pytesseract") from e

    with fitz.open(pdf_path) as doc:
        page_count = len(doc)

    if page_count == 0:
        return "No text could be extracted via OCR. The document might be blank or poor quality."

    workers = _resolve_worker_count(page_count)
    args_list = [(pdf_path, i, OCR_DPI, OCR_LANG) for i in range(page_count)]

    if workers == 1:
        results = [_ocr_one_page(a) for a in args_list]
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_ocr_one_page, args_list))
        except Exception as exc:
            # Fall back to sequential if the pool can't spin up (rare —
            # mostly seen in restricted sandboxes / containerised tests).
            logger.warning("OCR process pool failed (%s); falling back to sequential", exc)
            results = [_ocr_one_page(a) for a in args_list]

    out: list[str] = []
    for page_index, text in sorted(results, key=lambda r: r[0]):
        if text.strip():
            out.append(f"=== Page {page_index + 1} ===\n{text}\n")

    if not out:
        return "No text could be extracted via OCR. The document might be blank or poor quality."
    return "\n".join(out)
