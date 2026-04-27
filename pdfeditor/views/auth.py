"""Authentication views: register + a thin login wrapper.

Uses Django's built-in ``LoginView`` for sign-in and ``LogoutView`` for
sign-out (wired in urls.py). Registration is hand-rolled because the
default ``UserCreationForm`` doesn't ask for email — and we need it for the
confirmation flow added in phase 4.
"""

from __future__ import annotations

from typing import Any

from django import forms
from django.contrib import messages
from django.contrib.auth import get_user_model, login
from django.contrib.auth.forms import UserCreationForm
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render

from ..email_utils import decode_uid, is_token_valid, send_confirmation_email

User = get_user_model()


class RegisterForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        label="Email",
        help_text="We'll use this for password resets and confirmations.",
        widget=forms.EmailInput(attrs={"class": "form-input", "autocomplete": "email"}),
    )

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username", "email")

    def clean_email(self) -> str:
        email = (self.cleaned_data.get("email") or "").strip().lower()
        if email and User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("An account with this email already exists.")
        return email

    def save(self, commit: bool = True) -> Any:
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]
        if commit:
            user.save()
        return user


def register_view(request: HttpRequest) -> HttpResponse:
    if request.user.is_authenticated:
        return redirect("dashboard")

    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_active = False  # activated only after email confirmation
            user.save()

            sent = send_confirmation_email(user)
            if sent:
                messages.success(
                    request,
                    "Account created! Check your email to confirm your address before signing in.",
                )
            else:
                messages.warning(
                    request,
                    "Account created, but we couldn't send the confirmation email. "
                    "Contact support to activate your account.",
                )
            return redirect("login")
    else:
        form = RegisterForm()

    return render(request, "registration/register.html", {"form": form})


def confirm_email_view(request: HttpRequest, uidb64: str, token: str) -> HttpResponse:
    """Activate a user account from the link sent in the confirmation email."""
    user = decode_uid(uidb64)
    if user is None or not is_token_valid(user, token):
        return render(request, "registration/confirm_email_invalid.html", status=400)

    if not user.is_active:
        user.is_active = True
        user.save(update_fields=["is_active"])

    # Auto-login on confirmation — same UX as most SaaS apps.
    login(request, user, backend="django.contrib.auth.backends.ModelBackend")
    messages.success(request, "Email confirmed! Welcome aboard.")
    return redirect("dashboard")
