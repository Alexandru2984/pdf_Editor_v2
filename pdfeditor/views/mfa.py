"""Optional TOTP multi-factor auth: enrolment, login second step, recovery.

The login flow is two-step only for users who enabled MFA: ``MfaLoginView``
verifies the password (still through axes), then — instead of logging in —
parks the user id in the session and routes to ``mfa_verify_view`` for the
code. Everyone else logs in in one step as before.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth import login as auth_login
from django.contrib.auth import views as auth_views
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.utils.translation import gettext as _
from django.views.decorators.http import require_http_methods

from ..mfa import (
    consume_backup_code,
    generate_backup_codes,
    provisioning_uri,
    qr_data_uri,
    user_has_mfa,
    verify_totp,
)
from ..models import MfaDevice
from ..ratelimiting import auth_aware_ratelimit

# Session keys for the half-authenticated "password ok, code pending" state.
_PENDING_USER = "mfa_pending_user_id"
_PENDING_TS = "mfa_pending_ts"
_PENDING_NEXT = "mfa_pending_next"
_PENDING_MAX_AGE = 300  # seconds — re-login required if the code step stalls

# The user reaches mfa_verify from session, not authenticate(), so login()
# needs an explicit backend. Use the project's real (non-axes) backend.
_LOGIN_BACKEND = "pdfeditor.auth_backends.CaseInsensitiveModelBackend"


class MfaLoginView(auth_views.LoginView):
    """Password step. Defers to the TOTP step when the user has MFA on."""

    template_name = "registration/login.html"

    def form_valid(self, form):
        user = form.get_user()
        if user_has_mfa(user):
            self.request.session[_PENDING_USER] = user.pk
            self.request.session[_PENDING_TS] = timezone.now().isoformat()
            self.request.session[_PENDING_NEXT] = self.get_success_url()
            return redirect("mfa_verify")
        return super().form_valid(form)


def _pending_user(request: HttpRequest):
    uid = request.session.get(_PENDING_USER)
    ts = request.session.get(_PENDING_TS)
    if not uid or not ts:
        return None
    try:
        started = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    if timezone.now() - started > timedelta(seconds=_PENDING_MAX_AGE):
        return None
    return get_user_model().objects.filter(pk=uid, is_active=True).first()


def _clear_pending(request: HttpRequest) -> None:
    for key in (_PENDING_USER, _PENDING_TS, _PENDING_NEXT):
        request.session.pop(key, None)


@auth_aware_ratelimit(anon_rate="10/h", user_rate="20/h", method="POST")
def mfa_verify_view(request: HttpRequest) -> HttpResponse:
    """Second login step: accept a TOTP code or a single-use backup code."""
    user = _pending_user(request)
    if user is None:
        _clear_pending(request)
        messages.error(request, _("Your sign-in attempt expired. Please log in again."))
        return redirect("login")

    if request.method == "POST":
        code = request.POST.get("code") or ""
        device = MfaDevice.objects.filter(user=user, confirmed=True).first()
        if device and (verify_totp(device, code) or consume_backup_code(device, code)):
            next_url = request.session.get(_PENDING_NEXT) or settings.LOGIN_REDIRECT_URL
            _clear_pending(request)
            auth_login(request, user, backend=_LOGIN_BACKEND)
            return redirect(next_url)
        messages.error(request, _("Invalid code. Try again or use a backup code."))

    return render(request, "registration/mfa_verify.html")


@login_required
@auth_aware_ratelimit(anon_rate="10/h", user_rate="10/h", method="POST")
def mfa_setup_view(request: HttpRequest) -> HttpResponse:
    """Enrol an authenticator: show the QR, confirm one code, reveal backups."""
    device, _created = MfaDevice.objects.get_or_create(user=request.user)
    if device.confirmed:
        messages.info(request, _("Two-factor authentication is already enabled."))
        return redirect("profile")

    if request.method == "POST":
        if verify_totp(device, request.POST.get("code") or ""):
            device.confirmed = True
            device.confirmed_at = timezone.now()
            device.save(update_fields=["confirmed", "confirmed_at"])
            codes = generate_backup_codes(device)
            return render(request, "registration/mfa_backup_codes.html", {"codes": codes, "fresh": True})
        messages.error(
            request,
            _("That code didn't match. Check your device's clock and try again."),
        )

    uri = provisioning_uri(device, request.user.get_username())
    return render(
        request,
        "registration/mfa_setup.html",
        {"qr_data_uri": qr_data_uri(uri), "secret": device.secret},
    )


@login_required
@auth_aware_ratelimit(anon_rate="5/h", user_rate="5/h", method="POST")
@require_http_methods(["POST"])
def mfa_disable_view(request: HttpRequest) -> HttpResponse:
    """Turn MFA off — re-auth with the account password first."""
    if not request.user.check_password(request.POST.get("password") or ""):
        messages.error(request, _("Password incorrect — two-factor authentication was not disabled."))
        return redirect("profile")
    MfaDevice.objects.filter(user=request.user).delete()
    messages.success(request, _("Two-factor authentication disabled."))
    return redirect("profile")


@login_required
@auth_aware_ratelimit(anon_rate="5/h", user_rate="5/h", method="POST")
@require_http_methods(["POST"])
def mfa_backup_codes_view(request: HttpRequest) -> HttpResponse:
    """Regenerate backup codes (invalidates the old set)."""
    device = MfaDevice.objects.filter(user=request.user, confirmed=True).first()
    if device is None:
        messages.error(request, _("Enable two-factor authentication first."))
        return redirect("profile")
    codes = generate_backup_codes(device)
    return render(request, "registration/mfa_backup_codes.html", {"codes": codes, "fresh": False})
