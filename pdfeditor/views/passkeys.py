"""WebAuthn / passkey endpoints: registration, password-less login, removal.

Ceremony validation is delegated to ``py_webauthn``; these views own the
challenge lifecycle (one-shot, session-bound), the credential storage and
the Django login. Passkey sign-in is phishing-resistant possession+presence
auth, so it intentionally bypasses the TOTP second step — the authenticator
already provides two factors.

Wire format: the browser-side helper (static/js/passkeys.js) uses the
standard ``PublicKeyCredential.parseCreationOptionsFromJSON`` /
``credential.toJSON()`` pair, and py_webauthn accepts/emits exactly that
JSON shape — no manual base64url plumbing on either side.
"""

from __future__ import annotations

import json
import logging

from django.conf import settings
from django.contrib.auth import login
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_http_methods
from webauthn import (
    generate_authentication_options,
    generate_registration_options,
    options_to_json,
    verify_authentication_response,
    verify_registration_response,
)
from webauthn.helpers import base64url_to_bytes, bytes_to_base64url
from webauthn.helpers.exceptions import InvalidAuthenticationResponse, InvalidRegistrationResponse
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    PublicKeyCredentialDescriptor,
    ResidentKeyRequirement,
    UserVerificationRequirement,
)

from ..models import WebAuthnCredential
from ..ratelimiting import auth_aware_ratelimit

logger = logging.getLogger(__name__)

_REG_CHALLENGE_KEY = "webauthn_reg_challenge"
_AUTH_CHALLENGE_KEY = "webauthn_auth_challenge"
_MAX_BODY = 16 * 1024
_MAX_PASSKEYS_PER_USER = 10


def _rp_id() -> str:
    return settings.WEBAUTHN_RP_ID


def _origin() -> str:
    return settings.WEBAUTHN_ORIGIN


def _json_body(request: HttpRequest) -> dict:
    try:
        body = json.loads(request.body[:_MAX_BODY])
    except (ValueError, UnicodeDecodeError):
        return {}
    return body if isinstance(body, dict) else {}


@auth_aware_ratelimit(anon_rate="30/h", user_rate="60/h", method="POST")
@require_http_methods(["POST"])
def passkey_register_options_view(request: HttpRequest) -> HttpResponse:
    """Step 1 of enrolment: creation options + one-shot session challenge."""
    if not request.user.is_authenticated:
        return JsonResponse({"error": "auth required"}, status=403)
    if WebAuthnCredential.objects.filter(user=request.user).count() >= _MAX_PASSKEYS_PER_USER:
        return JsonResponse({"error": "passkey limit reached"}, status=400)

    options = generate_registration_options(
        rp_id=_rp_id(),
        rp_name=settings.WEBAUTHN_RP_NAME,
        user_id=str(request.user.pk).encode(),
        user_name=request.user.get_username(),
        # Discoverable credential so the passkey works for username-less
        # sign-in; user verification required (PIN/biometric) — the passkey
        # then carries both possession and knowledge/inherence factors.
        authenticator_selection=AuthenticatorSelectionCriteria(
            resident_key=ResidentKeyRequirement.REQUIRED,
            user_verification=UserVerificationRequirement.REQUIRED,
        ),
        exclude_credentials=[
            PublicKeyCredentialDescriptor(id=base64url_to_bytes(c.credential_id))
            for c in WebAuthnCredential.objects.filter(user=request.user)
        ],
    )
    request.session[_REG_CHALLENGE_KEY] = bytes_to_base64url(options.challenge)
    return HttpResponse(options_to_json(options), content_type="application/json")


@auth_aware_ratelimit(anon_rate="30/h", user_rate="60/h", method="POST")
@require_http_methods(["POST"])
def passkey_register_view(request: HttpRequest) -> HttpResponse:
    """Step 2 of enrolment: verify the attestation and store the credential."""
    if not request.user.is_authenticated:
        return JsonResponse({"error": "auth required"}, status=403)
    challenge_b64 = request.session.pop(_REG_CHALLENGE_KEY, None)  # one-shot
    if not challenge_b64:
        return JsonResponse({"error": "no pending registration"}, status=400)

    body = _json_body(request)
    credential = body.get("credential")
    if not credential:
        return JsonResponse({"error": "missing credential"}, status=400)

    try:
        verified = verify_registration_response(
            credential=credential,
            expected_challenge=base64url_to_bytes(challenge_b64),
            expected_rp_id=_rp_id(),
            expected_origin=_origin(),
            require_user_verification=True,
        )
    except InvalidRegistrationResponse as exc:
        logger.warning("Passkey registration rejected for user %s: %s", request.user.pk, exc)
        return JsonResponse({"error": "registration could not be verified"}, status=400)

    WebAuthnCredential.objects.create(
        user=request.user,
        credential_id=bytes_to_base64url(verified.credential_id),
        public_key=bytes_to_base64url(verified.credential_public_key),
        sign_count=verified.sign_count,
        transports=",".join(credential.get("response", {}).get("transports", []) or [])[:255],
        label=str(body.get("label") or "")[:80],
    )
    return JsonResponse({"ok": True})


@auth_aware_ratelimit(anon_rate="30/h", user_rate="60/h", method="POST")
@require_http_methods(["POST"])
def passkey_auth_options_view(request: HttpRequest) -> HttpResponse:
    """Step 1 of sign-in: request options with an empty allow-list, so the
    browser offers whatever discoverable credentials it holds for this RP."""
    options = generate_authentication_options(
        rp_id=_rp_id(),
        user_verification=UserVerificationRequirement.REQUIRED,
    )
    request.session[_AUTH_CHALLENGE_KEY] = bytes_to_base64url(options.challenge)
    return HttpResponse(options_to_json(options), content_type="application/json")


@auth_aware_ratelimit(anon_rate="30/h", user_rate="60/h", method="POST")
@require_http_methods(["POST"])
def passkey_login_view(request: HttpRequest) -> HttpResponse:
    """Step 2 of sign-in: verify the assertion and create the session."""
    challenge_b64 = request.session.pop(_AUTH_CHALLENGE_KEY, None)  # one-shot
    if not challenge_b64:
        return JsonResponse({"error": "no pending authentication"}, status=400)

    credential = _json_body(request).get("credential")
    if not isinstance(credential, dict) or not credential.get("id"):
        return JsonResponse({"error": "missing credential"}, status=400)

    stored = WebAuthnCredential.objects.select_related("user").filter(credential_id=credential["id"]).first()
    # Same error for unknown credential and failed verification — this
    # endpoint must not be an enumeration oracle.
    if stored is None or not stored.user.is_active:
        return JsonResponse({"error": "sign-in could not be verified"}, status=400)

    try:
        verified = verify_authentication_response(
            credential=credential,
            expected_challenge=base64url_to_bytes(challenge_b64),
            expected_rp_id=_rp_id(),
            expected_origin=_origin(),
            credential_public_key=base64url_to_bytes(stored.public_key),
            credential_current_sign_count=stored.sign_count,
            require_user_verification=True,
        )
    except InvalidAuthenticationResponse as exc:
        logger.warning("Passkey sign-in rejected (credential %s): %s", stored.pk, exc)
        return JsonResponse({"error": "sign-in could not be verified"}, status=400)

    WebAuthnCredential.objects.filter(pk=stored.pk).update(
        sign_count=verified.new_sign_count, last_used_at=timezone.now()
    )
    # Phishing-resistant possession+verification auth → no TOTP second step.
    login(request, stored.user, backend="pdfeditor.auth_backends.CaseInsensitiveModelBackend")
    return JsonResponse({"ok": True, "redirect": reverse("dashboard")})


@auth_aware_ratelimit(anon_rate="30/h", user_rate="120/h", method="POST")
@require_http_methods(["POST"])
def passkey_delete_view(request: HttpRequest, passkey_id) -> HttpResponse:
    """Plain form POST from the security page — redirects back with a flash."""
    from django.contrib import messages
    from django.shortcuts import redirect
    from django.utils.translation import gettext as _

    if not request.user.is_authenticated:
        return JsonResponse({"error": "auth required"}, status=403)
    deleted, _rows = WebAuthnCredential.objects.filter(user=request.user, id=passkey_id).delete()
    if deleted:
        messages.success(request, _("Passkey removed."))
    else:
        messages.error(request, _("Passkey not found."))
    return redirect("security_sessions")
