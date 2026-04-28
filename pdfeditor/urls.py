from django.contrib.auth import views as auth_views
from django.urls import path, reverse_lazy

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
    # Auth — login/logout/register; password change/reset added in phase 4
    path(
        "accounts/login/",
        auth_views.LoginView.as_view(template_name="registration/login.html"),
        name="login",
    ),
    path(
        "accounts/logout/",
        auth_views.LogoutView.as_view(next_page=reverse_lazy("dashboard")),
        name="logout",
    ),
    path("accounts/register/", views.register_view, name="register"),
    path(
        "accounts/confirm/<str:uidb64>/<str:token>/",
        views.confirm_email_view,
        name="confirm_email",
    ),
    path(
        "accounts/resend-confirmation/",
        views.resend_confirmation_view,
        name="resend_confirmation",
    ),
    # Password change (for logged-in users).
    path(
        "accounts/password/change/",
        auth_views.PasswordChangeView.as_view(
            template_name="registration/password_change_form.html",
            success_url=reverse_lazy("password_change_done"),
        ),
        name="password_change",
    ),
    path(
        "accounts/password/change/done/",
        auth_views.PasswordChangeDoneView.as_view(
            template_name="registration/password_change_done.html",
        ),
        name="password_change_done",
    ),
    # Password reset (forgot-password flow, for users who can't log in).
    path(
        "accounts/password/reset/",
        auth_views.PasswordResetView.as_view(
            template_name="registration/password_reset_form.html",
            email_template_name="registration/password_reset_email.txt",
            html_email_template_name="registration/password_reset_email.html",
            subject_template_name="registration/password_reset_subject.txt",
            success_url=reverse_lazy("password_reset_done"),
        ),
        name="password_reset",
    ),
    path(
        "accounts/password/reset/done/",
        auth_views.PasswordResetDoneView.as_view(
            template_name="registration/password_reset_done.html",
        ),
        name="password_reset_done",
    ),
    path(
        "accounts/password/reset/<uidb64>/<token>/",
        auth_views.PasswordResetConfirmView.as_view(
            template_name="registration/password_reset_confirm.html",
            success_url=reverse_lazy("password_reset_complete"),
        ),
        name="password_reset_confirm",
    ),
    path(
        "accounts/password/reset/complete/",
        auth_views.PasswordResetCompleteView.as_view(
            template_name="registration/password_reset_complete.html",
        ),
        name="password_reset_complete",
    ),
]
