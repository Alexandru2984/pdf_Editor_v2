"""PDF processing package.

Public API preserved for backward compatibility with the previous
monolithic ``pdf_processor.py`` module.
"""

from ._common import check_pdf_has_text, parse_page_range
from .edit import (
    find_and_replace_text,
    rephrase_text_in_pdf,
    rephrase_with_coordinates,
    replace_in_rect_safe,
    replace_with_flow,
)
from .extract import extract_text_from_pdf, ocr_pdf_to_text
from .ops import (
    add_page_numbers,
    add_watermark,
    compress_pdf,
    merge_pdfs,
    rotate_pages,
    split_pdf,
)

__all__ = [
    "check_pdf_has_text",
    "parse_page_range",
    "split_pdf",
    "merge_pdfs",
    "compress_pdf",
    "add_watermark",
    "rotate_pages",
    "add_page_numbers",
    "extract_text_from_pdf",
    "ocr_pdf_to_text",
    "find_and_replace_text",
    "rephrase_text_in_pdf",
    "rephrase_with_coordinates",
    "replace_in_rect_safe",
    "replace_with_flow",
]
