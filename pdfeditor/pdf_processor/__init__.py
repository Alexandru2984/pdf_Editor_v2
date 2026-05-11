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
from .extract import extract_text_from_pdf, make_pdf_searchable, ocr_pdf_to_text
from .forms import FormField, extract_form_fields, fill_form_fields, has_form_fields
from .ops import (
    add_page_numbers,
    add_watermark,
    compare_pdfs,
    compress_pdf,
    convert_images_to_pdf,
    convert_pdf_to_docx,
    convert_pdf_to_images,
    convert_to_pdfa,
    crop_pages,
    edit_pdf_metadata,
    flatten_pdf,
    merge_pdfs,
    protect_pdf,
    read_pdf_metadata,
    read_pdf_outline,
    redact_text,
    remove_pdf_password,
    render_page_thumbnail,
    reorder_pages,
    rotate_pages,
    set_pdf_outline,
    sign_pdf,
    split_pdf,
    verify_pdf_signatures,
)

__all__ = [
    "check_pdf_has_text",
    "parse_page_range",
    "split_pdf",
    "merge_pdfs",
    "compress_pdf",
    "convert_pdf_to_docx",
    "convert_pdf_to_images",
    "compare_pdfs",
    "convert_images_to_pdf",
    "convert_to_pdfa",
    "crop_pages",
    "flatten_pdf",
    "read_pdf_metadata",
    "read_pdf_outline",
    "set_pdf_outline",
    "edit_pdf_metadata",
    "reorder_pages",
    "render_page_thumbnail",
    "protect_pdf",
    "redact_text",
    "remove_pdf_password",
    "sign_pdf",
    "verify_pdf_signatures",
    "add_watermark",
    "rotate_pages",
    "add_page_numbers",
    "extract_text_from_pdf",
    "make_pdf_searchable",
    "ocr_pdf_to_text",
    "find_and_replace_text",
    "rephrase_text_in_pdf",
    "rephrase_with_coordinates",
    "replace_in_rect_safe",
    "replace_with_flow",
    "FormField",
    "extract_form_fields",
    "fill_form_fields",
    "has_form_fields",
]
