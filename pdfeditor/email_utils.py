"""Email helpers for account confirmation.

Wraps Django's ``default_token_generator`` so the same logic generates *and*
validates the activation token. The token implicitly invalidates once the
user becomes active (token_generator hashes ``user.is_active`` into the
seed), so a single-use guarantee comes for free.
"""

from __future__ import annotations

import logging

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AbstractBaseUser
from django.contrib.auth.tokens import default_token_generator
from django.core.mail import send_mail
from django.urls import reverse
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode

logger = logging.getLogger(__name__)
User = get_user_model()


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
    link = _build_confirmation_link(user)
    subject = "Confirm your PDF Editor account"
    body = (
        f"Hi {getattr(user, 'username', 'there')},\n\n"
        "Click the link below to confirm your account and activate sign-in:\n\n"
        f"{link}\n\n"
        "If you didn't sign up, just ignore this email.\n"
    )

    email = getattr(user, "email", "") or ""
    if not email:
        return False

    try:
        send_mail(
            subject,
            body,
            settings.DEFAULT_FROM_EMAIL,
            [email],
            fail_silently=False,
        )
        return True
    except Exception as exc:  # noqa: BLE001 — broad catch is intentional, see docstring
        logger.warning("Failed to send confirmation email to %s: %s", email, exc)
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
