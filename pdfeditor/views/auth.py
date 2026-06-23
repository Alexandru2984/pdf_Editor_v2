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
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy as _lazy

from ..email_utils import (
    decode_uid,
    is_token_valid,
    send_account_exists_notice,
    send_confirmation_email,
    send_email_change_confirmation,
    verify_email_change_token,
)
from ..ratelimiting import auth_aware_ratelimit

User = get_user_model()


class RegisterForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        label=_lazy("Email"),
        help_text=_lazy("We'll use this for password resets and confirmations."),
        widget=forms.EmailInput(attrs={"class": "form-input", "autocomplete": "email"}),
    )

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username", "email")

    def clean_email(self) -> str:
        # Normalise only. Uniqueness is NOT enforced here on purpose: raising
        # "this email already exists" would turn registration into an
        # account-existence oracle. The view handles the duplicate case
        # without revealing it (anti-enumeration) — see register_view.
        return (self.cleaned_data.get("email") or "").strip().lower()

    def save(self, commit: bool = True) -> Any:
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]
        if commit:
            user.save()
        return user


@auth_aware_ratelimit(anon_rate="5/h", user_rate="5/h", method="POST")
def register_view(request: HttpRequest) -> HttpResponse:
    if request.user.is_authenticated:
        return redirect("dashboard")

    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data["email"]
            existing = User.objects.filter(email__iexact=email).first() if email else None
            if existing is not None:
                # Address already in use. Don't tell the requester (that would
                # be an account-existence oracle) — notify the real owner and
                # fall through to the SAME generic message below. No duplicate
                # account is created, preserving email uniqueness.
                send_account_exists_notice(existing)
            else:
                user = form.save(commit=False)
                user.is_active = False  # activated only after email confirmation
                user.save()
                send_confirmation_email(user)

            # Identical response whether or not the email was already taken, so
            # response content/timing can't be used to enumerate accounts.
            messages.success(
                request,
                _("Almost there — check your email to confirm your address and finish signing up."),
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

    # Auto-login on confirmation — same UX as most SaaS apps. Must name a
    # backend that's actually in AUTHENTICATION_BACKENDS, otherwise Django's
    # get_user() drops the session to AnonymousUser on the next request and
    # the "login" silently doesn't stick.
    login(request, user, backend="pdfeditor.auth_backends.CaseInsensitiveModelBackend")
    messages.success(request, _("Email confirmed! Welcome aboard."))
    return redirect("dashboard")


class ResendConfirmationForm(forms.Form):
    email = forms.EmailField(
        required=True,
        label=_lazy("Email address"),
        widget=forms.EmailInput(
            attrs={"class": "form-input", "autocomplete": "email", "autofocus": True},
        ),
    )


class EmailChangeForm(forms.Form):
    new_email = forms.EmailField(
        required=True,
        label=_lazy("New email address"),
        widget=forms.EmailInput(attrs={"class": "form-input", "autocomplete": "email"}),
    )
    current_password = forms.CharField(
        required=True,
        label=_lazy("Current password"),
        widget=forms.PasswordInput(attrs={"class": "form-input", "autocomplete": "current-password"}),
    )

    def __init__(self, *args: Any, user: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.user = user

    def clean_new_email(self) -> str:
        email = (self.cleaned_data.get("new_email") or "").strip().lower()
        if self.user is not None and email == (self.user.email or "").strip().lower():
            raise forms.ValidationError(_("That's already your current email."))
        if email and User.objects.filter(email__iexact=email).exclude(pk=self.user.pk).exists():
            raise forms.ValidationError(_("An account with this email already exists."))
        return email

    def clean_current_password(self) -> str:
        pw = self.cleaned_data.get("current_password") or ""
        if not (self.user and self.user.check_password(pw)):
            raise forms.ValidationError(_("That password doesn't match your account."))
        return pw


@login_required
@auth_aware_ratelimit(anon_rate="5/h", user_rate="5/h", method="POST")
def change_email_view(request: HttpRequest) -> HttpResponse:
    """Authenticated form: prove password ownership + request new-email confirmation.

    The actual swap only happens after the user clicks the link sent to the
    *new* address — until then their old email remains the source of truth.
    """
    if request.method == "POST":
        form = EmailChangeForm(request.POST, user=request.user)
        if form.is_valid():
            new_email = form.cleaned_data["new_email"]
            sent = send_email_change_confirmation(request.user, new_email)
            if sent:
                messages.success(
                    request,
                    f"We sent a confirmation link to {new_email}. "
                    "Click it to finish the change — your current email stays active until then.",
                )
            else:
                messages.warning(
                    request,
                    "We couldn't send the confirmation email. Please try again later.",
                )
            return redirect("change_email")
    else:
        form = EmailChangeForm(user=request.user)

    return render(request, "registration/email_change_form.html", {"form": form})


def confirm_email_change_view(request: HttpRequest, token: str) -> HttpResponse:
    """Apply the email change requested via the signed token."""
    result = verify_email_change_token(token)
    if result is None:
        return render(request, "registration/email_change_invalid.html", status=400)

    user, new_email = result
    # Re-check uniqueness at apply-time: someone else may have claimed this
    # address between request and confirmation.
    if User.objects.filter(email__iexact=new_email).exclude(pk=user.pk).exists():
        return render(request, "registration/email_change_invalid.html", status=400)

    user.email = new_email
    user.save(update_fields=["email"])
    messages.success(request, f"Email updated to {new_email}.")
    return redirect("dashboard" if request.user.is_authenticated else "login")


@auth_aware_ratelimit(anon_rate="3/h", user_rate="3/h", method="POST")
def resend_confirmation_view(request: HttpRequest) -> HttpResponse:
    """Allow inactive users to request a fresh confirmation email.

    Always renders the same "we sent it if the address exists" page on POST,
    so this endpoint can't be used as an account-existence oracle.
    """
    if request.user.is_authenticated:
        return redirect("dashboard")

    if request.method == "POST":
        form = ResendConfirmationForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data["email"].strip().lower()
            user = User.objects.filter(email__iexact=email, is_active=False).first()
            if user is not None:
                send_confirmation_email(user)
            return render(request, "registration/resend_confirmation_done.html")
    else:
        form = ResendConfirmationForm()

    return render(request, "registration/resend_confirmation_form.html", {"form": form})
