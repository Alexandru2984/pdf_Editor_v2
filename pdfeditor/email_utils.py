"""Email helpers for account confirmation and email-change.

The activation flow wraps Django's ``default_token_generator`` so the same
logic generates *and* validates the token. The token implicitly invalidates
once the user becomes active (token_generator hashes ``user.is_active`` into
the seed), so a single-use guarantee comes for free.

The email-change flow uses ``django.core.signing`` with a payload that
includes the user's *current* email — meaning any subsequent change (or
revert) invalidates pending tokens.
"""

from __future__ import annotations

import logging
from typing import Any

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AbstractBaseUser
from django.contrib.auth.tokens import default_token_generator
from django.core.mail import EmailMultiAlternatives
from django.core.signing import BadSignature, SignatureExpired, dumps, loads
from django.template.loader import render_to_string
from django.urls import reverse
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode

logger = logging.getLogger(__name__)
User = get_user_model()

EMAIL_CHANGE_SALT = "pdfeditor.email_change.v1"
EMAIL_CHANGE_MAX_AGE_SECONDS = 24 * 3600


def _build_confirmation_link(user: AbstractBaseUser) -> str:
    uid = urlsafe_base64_encode(force_bytes(user.pk))
    token = default_token_generator.make_token(user)
    path = reverse("confirm_email", kwargs={"uidb64": uid, "token": token})
    return f"{settings.SITE_URL.rstrip('/')}{path}"


def send_confirmation_email(user: AbstractBaseUser) -> bool:
    """Send the activation email; returns True on success, False on SMTP failure.

    We swallow SMTP errors so a misconfigured mail server can't take the
    registration page down. The caller should warn the user when this returns
    False.
    """
    email = getattr(user, "email", "") or ""
    if not email:
        return False

    link = _build_confirmation_link(user)
    context = {"username": getattr(user, "username", "there"), "link": link}
    subject = "Confirm your PDF Editor account"
    text_body = render_to_string("registration/confirmation_email.txt", context)
    html_body = render_to_string("registration/confirmation_email.html", context)

    try:
        message = EmailMultiAlternatives(
            subject=subject,
            body=text_body,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[email],
        )
        message.attach_alternative(html_body, "text/html")
        message.send(fail_silently=False)
        return True
    except Exception as exc:  # noqa: BLE001 — broad catch is intentional, see docstring
        logger.warning("Failed to send confirmation email to %s: %s", email, exc)
        return False


def send_account_exists_notice(user: AbstractBaseUser) -> bool:
    """Tell an existing account that someone tried to register with its email.

    Used by the registration view's anti-enumeration path: instead of telling
    the *requester* that the address is taken (an account-existence oracle),
    we notify the real owner and show the requester the same generic
    "check your email" message as a genuine sign-up. Best-effort.
    """
    email = getattr(user, "email", "") or ""
    if not email:
        return False

    from django.core.mail import send_mail

    login_url = f"{settings.SITE_URL.rstrip('/')}{reverse('login')}"
    reset_url = f"{settings.SITE_URL.rstrip('/')}{reverse('password_reset')}"
    try:
        send_mail(
            subject="You already have a PDF Editor account",
            message=(
                f"Hi {getattr(user, 'username', 'there')},\n\n"
                "Someone just tried to create a PDF Editor account using this "
                "email address, but you already have one.\n\n"
                f"If this was you, just sign in: {login_url}\n"
                f"Forgot your password? Reset it here: {reset_url}\n\n"
                "If it wasn't you, you can safely ignore this email — no "
                "account was created and nothing changed."
            ),
            from_email=None,  # DEFAULT_FROM_EMAIL
            recipient_list=[email],
            fail_silently=False,
        )
        return True
    except Exception as exc:  # noqa: BLE001 — same broad-catch policy as the others
        logger.warning("Failed to send account-exists notice to %s: %s", email, exc)
        return False


def decode_uid(uidb64: str) -> AbstractBaseUser | None:
    """Recover the User from a urlsafe-base64 PK, or None if it's invalid."""
    try:
        pk = urlsafe_base64_decode(uidb64).decode()
        return User.objects.filter(pk=pk).first()
    except (TypeError, ValueError, OverflowError, UnicodeDecodeError):
        return None


def is_token_valid(user: AbstractBaseUser, token: str) -> bool:
    return default_token_generator.check_token(user, token)


def make_email_change_token(user: AbstractBaseUser, new_email: str) -> str:
    """Return a signed token authorising ``user`` to switch to ``new_email``.

    Embedding the *current* email means any later change to ``user.email``
    invalidates this token — so revoking by changing back-and-forth still
    works, and a stolen old token is unusable after any other email change.
    """
    payload = {
        "user_pk": user.pk,
        "new_email": new_email.strip().lower(),
        "old_email": (getattr(user, "email", "") or "").strip().lower(),
    }
    return dumps(payload, salt=EMAIL_CHANGE_SALT)


def verify_email_change_token(token: str) -> tuple[AbstractBaseUser, str] | None:
    """Return (user, new_email) if the token is valid and current, else None."""
    try:
        payload: dict[str, Any] = loads(token, salt=EMAIL_CHANGE_SALT, max_age=EMAIL_CHANGE_MAX_AGE_SECONDS)
    except SignatureExpired:
        logger.info("Email-change token expired")
        return None
    except BadSignature:
        logger.warning("Email-change token failed signature check")
        return None

    user = User.objects.filter(pk=payload.get("user_pk")).first()
    if user is None or not user.is_active:
        return None

    current_email = (getattr(user, "email", "") or "").strip().lower()
    if current_email != (payload.get("old_email") or ""):
        # User's email has changed since the token was issued — reject.
        return None

    new_email = (payload.get("new_email") or "").strip().lower()
    if not new_email:
        return None
    return user, new_email


def send_email_change_confirmation(user: AbstractBaseUser, new_email: str) -> bool:
    """Send a confirmation link to the *new* address. Returns True on success."""
    if not new_email:
        return False

    token = make_email_change_token(user, new_email)
    path = reverse("confirm_email_change", kwargs={"token": token})
    link = f"{settings.SITE_URL.rstrip('/')}{path}"
    context = {
        "username": getattr(user, "username", "there"),
        "link": link,
        "new_email": new_email,
    }
    subject = "Confirm your new PDF Editor email address"
    text_body = render_to_string("registration/email_change_email.txt", context)
    html_body = render_to_string("registration/email_change_email.html", context)

    try:
        message = EmailMultiAlternatives(
            subject=subject,
            body=text_body,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[new_email],
        )
        message.attach_alternative(html_body, "text/html")
        message.send(fail_silently=False)
        return True
    except Exception as exc:  # noqa: BLE001 — same broad-catch policy as send_confirmation_email
        logger.warning("Failed to send email-change confirmation to %s: %s", new_email, exc)
        return False
