"""Profile view + GDPR account-management actions (export, delete)."""

from __future__ import annotations

import logging
from typing import Any

from django import forms
from django.contrib import messages
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.utils.translation import gettext as _
from django.views.decorators.http import require_http_methods

from ..models import ApiKey, ProcessedPDF, UploadedPDF

logger = logging.getLogger(__name__)


@login_required
def profile_view(request: HttpRequest) -> HttpResponse:
    user = request.user
    uploaded_count = UploadedPDF.objects.filter(user=user).count()
    processed_count = ProcessedPDF.objects.filter(user=user).count()
    api_keys = ApiKey.objects.filter(user=user)
    new_token = request.session.pop("new_api_key_token", None)

    return render(
        request,
        "pdfeditor/profile.html",
        {
            "uploaded_count": uploaded_count,
            "processed_count": processed_count,
            "api_keys": api_keys,
            "new_api_key_token": new_token,
        },
    )


@login_required
@require_http_methods(["POST"])
def create_api_key_view(request: HttpRequest) -> HttpResponse:
    label = (request.POST.get("label") or "").strip()[:80]
    _, token = ApiKey.create_for_user(request.user, label=label)
    request.session["new_api_key_token"] = token
    messages.success(request, _("API key created. Copy the token now — it won't be shown again."))
    return redirect("profile")


@login_required
@require_http_methods(["POST"])
def revoke_api_key_view(request: HttpRequest, key_id: str) -> HttpResponse:
    key = get_object_or_404(ApiKey, user=request.user, id=key_id)
    if key.revoked_at is None:
        key.revoked_at = timezone.now()
        key.save(update_fields=["revoked_at"])
        messages.success(request, _("API key revoked."))
    return redirect("profile")


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


@login_required
def delete_account_view(request: HttpRequest) -> HttpResponse:
    """Permanent account deletion. Wipes user, FK-cascaded PDF rows, and on-disk files.

    The post_delete signal in pdfeditor/signals.py fires for each cascaded
    UploadedPDF/ProcessedPDF row and removes the matching file + thumbnail.
    """
    if request.method == "POST":
        form = DeleteAccountForm(request.POST, user=request.user)
        if form.is_valid():
            user = request.user
            username = user.username
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
