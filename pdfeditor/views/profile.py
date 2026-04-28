"""Profile view + GDPR account-management actions (export, delete)."""

from __future__ import annotations

import logging
import os
from typing import Any

from django import forms
from django.contrib import messages
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render

from ..models import ProcessedPDF, UploadedPDF

logger = logging.getLogger(__name__)


@login_required
def profile_view(request: HttpRequest) -> HttpResponse:
    user = request.user
    uploaded_count = UploadedPDF.objects.filter(user=user).count()
    processed_count = ProcessedPDF.objects.filter(user=user).count()

    return render(
        request,
        "pdfeditor/profile.html",
        {
            "uploaded_count": uploaded_count,
            "processed_count": processed_count,
        },
    )


@login_required
def export_data_view(request: HttpRequest) -> JsonResponse:
    """Return a JSON dump of the user's account data + PDF metadata (GDPR Art. 20)."""
    user = request.user
    uploaded = [
        {
            "id": str(pdf.id),
            "name": pdf.name,
            "size_bytes": pdf.size,
            "uploaded_at": pdf.uploaded_at.isoformat(),
        }
        for pdf in UploadedPDF.objects.filter(user=user).order_by("-uploaded_at")
    ]
    processed = [
        {
            "id": str(p.id),
            "kind": p.kind,
            "kind_display": p.get_kind_display(),
            "name": p.name,
            "size_bytes": p.size,
            "created_at": p.created_at.isoformat(),
            "source_id": str(p.source_id) if p.source_id else None,
        }
        for p in ProcessedPDF.objects.filter(user=user).order_by("-created_at")
    ]

    payload: dict[str, Any] = {
        "account": {
            "username": user.username,
            "email": user.email,
            "date_joined": user.date_joined.isoformat(),
            "last_login": user.last_login.isoformat() if user.last_login else None,
            "is_active": user.is_active,
        },
        "uploaded_pdfs": uploaded,
        "processed_pdfs": processed,
    }
    response = JsonResponse(payload, json_dumps_params={"indent": 2})
    response["Content-Disposition"] = f'attachment; filename="pdfeditor-data-{user.username}.json"'
    return response


class DeleteAccountForm(forms.Form):
    """Two-factor confirmation: password + literal-string typed in."""

    password = forms.CharField(
        required=True,
        label="Current password",
        widget=forms.PasswordInput(attrs={"class": "form-input", "autocomplete": "current-password"}),
    )
    confirmation = forms.CharField(
        required=True,
        label='Type "DELETE" to confirm',
        widget=forms.TextInput(attrs={"class": "form-input", "autocomplete": "off"}),
    )

    def __init__(self, *args: Any, user: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.user = user

    def clean_password(self) -> str:
        pw = self.cleaned_data.get("password") or ""
        if not (self.user and self.user.check_password(pw)):
            raise forms.ValidationError("That password doesn't match your account.")
        return pw

    def clean_confirmation(self) -> str:
        value = (self.cleaned_data.get("confirmation") or "").strip()
        if value != "DELETE":
            raise forms.ValidationError('You must type "DELETE" exactly.')
        return value


def _delete_user_files(user: Any) -> None:
    """Best-effort removal of files owned by ``user`` from disk.

    DB rows go via the FK CASCADE on User delete; we just have to clean up
    the actual blobs. Errors are logged, not raised — a missing file
    shouldn't block account deletion.
    """
    paths: list[str] = []
    paths.extend(UploadedPDF.objects.filter(user=user).values_list("path", flat=True))
    paths.extend(ProcessedPDF.objects.filter(user=user).values_list("path", flat=True))
    for path in paths:
        if not path:
            continue
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as exc:
            logger.warning("Failed to remove %s while deleting user %s: %s", path, user.pk, exc)


@login_required
def delete_account_view(request: HttpRequest) -> HttpResponse:
    """Permanent account deletion. Wipes user, FK-cascaded PDF rows, and on-disk files."""
    if request.method == "POST":
        form = DeleteAccountForm(request.POST, user=request.user)
        if form.is_valid():
            user = request.user
            username = user.username
            _delete_user_files(user)
            logout(request)
            user.delete()
            logger.info("Account deleted: %s", username)
            messages.success(request, "Your account and all associated data have been deleted.")
            return redirect("dashboard")
    else:
        form = DeleteAccountForm(user=request.user)

    return render(request, "pdfeditor/delete_account.html", {"form": form})


__all__ = [
    "profile_view",
    "export_data_view",
    "delete_account_view",
]
