from django.urls import path

from . import views

urlpatterns = [
    path("", views.dashboard_view, name="dashboard"),
    path("upload/", views.upload_view, name="upload"),
    path("edit/", views.edit_view, name="edit"),
    path("result/", views.result_view, name="result"),
    path("download/", views.download_view, name="download"),
    path("preview/", views.preview_view, name="preview"),
    path("split/", views.split_view, name="split"),
    path("split/result/", views.split_result_view, name="split_result"),
    path("download_split/", views.download_split_file_view, name="download_split"),
    path("merge/", views.merge_view, name="merge"),
    path("merge/result/", views.merge_result_view, name="merge_result"),
    path("download_merged/", views.download_merged_view, name="download_merged"),
    path("compress/", views.compress_view, name="compress"),
    path("compress/result/", views.compress_result_view, name="compress_result"),
    path("download_compressed/", views.download_compressed_view, name="download_compressed"),
    path("watermark/", views.watermark_view, name="watermark"),
    path("watermark/result/", views.watermark_result_view, name="watermark_result"),
    path("download_watermarked/", views.download_watermarked_view, name="download_watermarked"),
    path("rotate/", views.rotate_view, name="rotate"),
    path("rotate/result/", views.rotate_result_view, name="rotate_result"),
    path("download_rotated/", views.download_rotated_view, name="download_rotated"),
    path("page-numbers/", views.page_numbers_view, name="page_numbers"),
    path("page-numbers/result/", views.page_numbers_result_view, name="page_numbers_result"),
    path("download_numbered/", views.download_numbered_view, name="download_numbered"),
    path("more-tools/", views.more_tools_view, name="more_tools"),
    path("extract-text/<str:pdf_id>/", views.extract_text_ajax, name="extract_text"),
    path("ocr-text/<str:pdf_id>/", views.ocr_text_ajax, name="ocr_text"),
    path("download-text/", views.download_text_view, name="download_text"),
    path("delete-pdf/<str:pdf_id>/", views.delete_pdf_view, name="delete_pdf"),
    # AI Rephrase
    path("rephrase/", views.rephrase_view, name="rephrase"),
    path("rephrase/preview/", views.rephrase_preview_ajax, name="rephrase_preview"),
    path("rephrase/result/", views.rephrase_result_view, name="rephrase_result"),
    path("download_rephrased/", views.download_rephrased_view, name="download_rephrased"),
    # Form fill
    path("form-fill/", views.form_fill_view, name="form_fill"),
    path("form-fill/result/", views.form_fill_result_view, name="form_fill_result"),
    path("download_filled/", views.download_filled_view, name="download_filled"),
    # History
    path("history/", views.history_view, name="history"),
    path("history/download/<str:output_id>/", views.history_download_view, name="history_download"),
    path("history/delete/<str:output_id>/", views.history_delete_view, name="history_delete"),
    # Health checks (skipped from rate-limit + axes; safe for LBs)
    path("healthz", views.healthz, name="healthz"),
    path("readyz", views.readyz, name="readyz"),
]
