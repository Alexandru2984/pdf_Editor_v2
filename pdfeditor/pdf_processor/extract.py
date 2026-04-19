"""Text extraction: direct text layer + OCR fallback."""
from __future__ import annotations

import io
import os
from typing import List

import fitz


def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    with fitz.open(pdf_path) as doc:
        chunks: List[str] = []
        for i, page in enumerate(doc, 1):
            t = page.get_text()
            if t.strip():
                chunks.append(f"=== Page {i} ===\n{t}\n")

    if not chunks:
        return "No text found in PDF. This might be a scanned document - try OCR instead."

    return "\n".join(chunks)


def ocr_pdf_to_text(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:
        raise Exception("pytesseract not installed. Run: pip install pytesseract") from e

    with fitz.open(pdf_path) as doc:
        out: List[str] = []
        for i, page in enumerate(doc, 1):
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)
            if text.strip():
                out.append(f"=== Page {i} ===\n{text}\n")

    if not out:
        return "No text could be extracted via OCR. The document might be blank or poor quality."

    return "\n".join(out)
