from django.contrib.auth import views as auth_views
from django.urls import path, reverse_lazy

from . import views
from .ratelimiting import auth_aware_ratelimit

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
    path("convert/", views.convert_view, name="convert"),
    path("convert/result/", views.convert_result_view, name="convert_result"),
    path("download_converted/", views.download_converted_view, name="download_converted"),
    path("to-images/", views.to_images_view, name="to_images"),
    path("to-images/result/", views.to_images_result_view, name="to_images_result"),
    path("download_images/", views.download_images_view, name="download_images"),
    path("images-to-pdf/", views.images_to_pdf_view, name="images_to_pdf"),
    path("images-to-pdf/result/", views.images_to_pdf_result_view, name="images_to_pdf_result"),
    path("download_images_to_pdf/", views.download_images_to_pdf_view, name="download_images_to_pdf"),
    path("metadata/", views.metadata_view, name="metadata"),
    path("metadata/result/", views.metadata_result_view, name="metadata_result"),
    path("download_metadata/", views.download_metadata_view, name="download_metadata"),
    path("protect/", views.protect_view, name="protect"),
    path("protect/result/", views.protect_result_view, name="protect_result"),
    path("download_protected/", views.download_protected_view, name="download_protected"),
    path("unprotect/", views.unprotect_view, name="unprotect"),
    path("unprotect/result/", views.unprotect_result_view, name="unprotect_result"),
    path("download_unprotected/", views.download_unprotected_view, name="download_unprotected"),
    path("flatten/", views.flatten_view, name="flatten"),
    path("flatten/result/", views.flatten_result_view, name="flatten_result"),
    path("download_flattened/", views.download_flattened_view, name="download_flattened"),
    path("redact/", views.redact_view, name="redact"),
    path("redact/result/", views.redact_result_view, name="redact_result"),
    path("download_redacted/", views.download_redacted_view, name="download_redacted"),
    path("searchable/", views.searchable_view, name="searchable"),
    path("searchable/result/", views.searchable_result_view, name="searchable_result"),
    path("download_searchable/", views.download_searchable_view, name="download_searchable"),
    path("pdfa/", views.pdfa_view, name="pdfa"),
    path("pdfa/result/", views.pdfa_result_view, name="pdfa_result"),
    path("download_pdfa/", views.download_pdfa_view, name="download_pdfa"),
    path("sign/", views.sign_view, name="sign"),
    path("sign/result/", views.sign_result_view, name="sign_result"),
    path("sign/generate-cert/", views.generate_cert_view, name="generate_cert"),
    path("sign/verify/", views.verify_signature_view, name="verify_signature"),
    path("download_signed/", views.download_signed_view, name="download_signed"),
    path("watermark/", views.watermark_view, name="watermark"),
    path("watermark/result/", views.watermark_result_view, name="watermark_result"),
    path("download_watermarked/", views.download_watermarked_view, name="download_watermarked"),
    path("rotate/", views.rotate_view, name="rotate"),
    path("rotate/result/", views.rotate_result_view, name="rotate_result"),
    path("download_rotated/", views.download_rotated_view, name="download_rotated"),
    path("page-numbers/", views.page_numbers_view, name="page_numbers"),
    path("page-numbers/result/", views.page_numbers_result_view, name="page_numbers_result"),
    path("download_numbered/", views.download_numbered_view, name="download_numbered"),
    path("reorder/", views.reorder_view, name="reorder"),
    path("reorder/result/", views.reorder_result_view, name="reorder_result"),
    path("download_reordered/", views.download_reordered_view, name="download_reordered"),
    path("crop/", views.crop_view, name="crop"),
    path("crop/result/", views.crop_result_view, name="crop_result"),
    path("download_cropped/", views.download_cropped_view, name="download_cropped"),
    path(
        "pdf/<str:pdf_id>/page/<int:page_number>/thumbnail/",
        views.page_thumbnail_view,
        name="page_thumbnail",
    ),
    path("more-tools/", views.more_tools_view, name="more_tools"),
    path("extract-text/<str:pdf_id>/", views.extract_text_ajax, name="extract_text"),
    path("ocr-text/<str:pdf_id>/", views.ocr_text_ajax, name="ocr_text"),
    path("download-text/", views.download_text_view, name="download_text"),
    path("delete-pdf/<str:pdf_id>/", views.delete_pdf_view, name="delete_pdf"),
    path("pdf/<str:pdf_id>/thumbnail/", views.thumbnail_view, name="pdf_thumbnail"),
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
    path("accounts/profile/", views.profile_view, name="profile"),
    path("accounts/profile/export/", views.export_data_view, name="export_data"),
    path("accounts/profile/delete/", views.delete_account_view, name="delete_account"),
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
    # Email change (for logged-in users — proves password ownership and
    # confirms via a signed-token link sent to the new address).
    path("accounts/email/change/", views.change_email_view, name="change_email"),
    path(
        "accounts/email/change/confirm/<str:token>/",
        views.confirm_email_change_view,
        name="confirm_email_change",
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
    # Rate-limited per IP/user since each POST sends an email — without a
    # cap, an attacker can flood arbitrary inboxes via our SMTP relay.
    path(
        "accounts/password/reset/",
        auth_aware_ratelimit(anon_rate="5/h", user_rate="5/h", method="POST")(
            auth_views.PasswordResetView.as_view(
                template_name="registration/password_reset_form.html",
                email_template_name="registration/password_reset_email.txt",
                html_email_template_name="registration/password_reset_email.html",
                subject_template_name="registration/password_reset_subject.txt",
                success_url=reverse_lazy("password_reset_done"),
            )
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
