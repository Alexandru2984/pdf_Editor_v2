"""Thin Python client for the PDF Editor REST API.

Quick start:

    from pdf_editor import PdfEditorClient

    c = PdfEditorClient("https://pdf.micutu.com", api_key="ghp_...")
    pdf = c.upload("/path/to/file.pdf")
    out = c.compress(pdf["id"], quality="medium")
    c.download(out["id"], "compressed.pdf")

Async ops (OCR, PDF/A, compare, …) return a Job; ``wait_for`` polls
until terminal.

    job = c.ocr(pdf["id"], language="eng+ron")
    job = c.wait_for(job)
    c.download(job["output_id"], "ocr.pdf")
"""

from .client import ApiError, PdfEditorClient

__all__ = ["PdfEditorClient", "ApiError"]
__version__ = "0.1.0"
