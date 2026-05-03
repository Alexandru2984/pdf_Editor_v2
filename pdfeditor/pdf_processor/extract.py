"""Text extraction: direct text layer + OCR fallback.

The big win is skipping any page that already has an extractable text
layer — on mixed-content PDFs, only the truly scanned pages pay the
rasterise+OCR cost. OCR runs sequentially by default: a process pool
inherits the gunicorn worker's memory and OOM-kills the host, while a
thread pool runs slower than sequential on this hardware (tesseract
subprocesses contend for CPU and bloat per-page latency). Set
``OCR_MAX_WORKERS`` to opt into thread-pool parallelism.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import fitz

logger = logging.getLogger(__name__)

# 200dpi is enough for body text at A4 — readable by tesseract while ~2.25x
# faster to render than 300dpi. Override per environment if needed.
OCR_DPI = int(os.environ.get("OCR_DPI", "200"))

# tesseract loads each language model once; "eng+ron" handles the bulk of
# this app's audience. Override with OCR_LANG if you only need one.
OCR_LANG = os.environ.get("OCR_LANG", "eng+ron")

# Default to 1 (sequential) — safest in a gunicorn worker. Set higher to
# experiment with thread-pool parallelism on faster hardware.
OCR_MAX_WORKERS = int(os.environ.get("OCR_MAX_WORKERS", "1"))

# systemd often runs services with a minimal PATH that excludes /usr/bin,
# so pytesseract's default subprocess('tesseract') lookup fails. Resolve
# the binary up-front and tell pytesseract exactly where it is.
TESSERACT_CMD = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract") or "/usr/bin/tesseract"


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
    """Render one page and OCR it. Safe to call from multiple threads."""
    pdf_path, page_index, dpi, lang = args
    import pytesseract
    from PIL import Image

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        layer_text = page.get_text()
        if layer_text.strip():
            return page_index, layer_text
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return page_index, pytesseract.image_to_string(img, lang=lang)


def _resolve_worker_count(page_count: int) -> int:
    return max(1, min(OCR_MAX_WORKERS, page_count))


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
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_ocr_one_page, args_list))

    out: list[str] = []
    for page_index, text in sorted(results, key=lambda r: r[0]):
        if text.strip():
            out.append(f"=== Page {page_index + 1} ===\n{text}\n")

    if not out:
        return "No text could be extracted via OCR. The document might be blank or poor quality."
    return "\n".join(out)
