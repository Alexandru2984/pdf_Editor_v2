"""Views package.

Re-exports every view function at the package level so existing URL
conf and tests can keep using ``from pdfeditor import views``.
"""

from ._common import serve_media_view
from .auth import (
    change_email_view,
    confirm_email_change_view,
    confirm_email_view,
    register_view,
    resend_confirmation_view,
)
from .basic_ops import (
    compress_result_view,
    compress_view,
    download_compressed_view,
    download_merged_view,
    download_split_file_view,
    merge_result_view,
    merge_view,
    split_result_view,
    split_view,
)
from .edit import (
    download_view,
    edit_view,
    preview_view,
    result_view,
)
from .extract import (
    download_text_view,
    extract_text_ajax,
    more_tools_view,
    ocr_text_ajax,
)
from .form_fill import (
    download_filled_view,
    form_fill_result_view,
    form_fill_view,
)
from .health import healthz, readyz
from .history import history_delete_view, history_download_view, history_view
from .layout_ops import (
    download_numbered_view,
    download_rotated_view,
    download_watermarked_view,
    page_numbers_result_view,
    page_numbers_view,
    rotate_result_view,
    rotate_view,
    watermark_result_view,
    watermark_view,
)
from .profile import delete_account_view, export_data_view, profile_view
from .rephrase import (
    download_rephrased_view,
    rephrase_preview_ajax,
    rephrase_result_view,
    rephrase_view,
)
from .upload import dashboard_view, delete_pdf_view, upload_view

__all__ = [
    "serve_media_view",
    "dashboard_view",
    "upload_view",
    "delete_pdf_view",
    "edit_view",
    "result_view",
    "download_view",
    "preview_view",
    "split_view",
    "split_result_view",
    "download_split_file_view",
    "merge_view",
    "merge_result_view",
    "download_merged_view",
    "compress_view",
    "compress_result_view",
    "download_compressed_view",
    "watermark_view",
    "watermark_result_view",
    "download_watermarked_view",
    "rotate_view",
    "rotate_result_view",
    "download_rotated_view",
    "page_numbers_view",
    "page_numbers_result_view",
    "download_numbered_view",
    "more_tools_view",
    "extract_text_ajax",
    "ocr_text_ajax",
    "download_text_view",
    "rephrase_view",
    "rephrase_preview_ajax",
    "rephrase_result_view",
    "download_rephrased_view",
    "form_fill_view",
    "form_fill_result_view",
    "download_filled_view",
    "history_view",
    "history_download_view",
    "history_delete_view",
    "healthz",
    "readyz",
    "register_view",
    "confirm_email_view",
    "resend_confirmation_view",
    "change_email_view",
    "confirm_email_change_view",
    "profile_view",
    "export_data_view",
    "delete_account_view",
]
