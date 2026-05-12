"""Have-I-Been-Pwned password validator.

Uses the HIBP range API (k-anonymity model — only the first 5 chars of the
SHA1 hash leave the box, never the full password). Blocks passwords that
appear in the public breach corpus.

Fail-open by design: network or upstream errors must not break sign-up.
The tradeoff is conscious — degraded availability of HIBP can otherwise
DoS our own registration flow, which is worse than letting one weak
password slip through during an outage.
"""

from __future__ import annotations

import hashlib
import logging

import httpx
from django.conf import settings
from django.core.exceptions import ValidationError
from django.utils.translation import gettext as _

logger = logging.getLogger(__name__)

_HIBP_RANGE_URL = "https://api.pwnedpasswords.com/range/{prefix}"
_TIMEOUT_SECONDS = 2.5


class PwnedPasswordValidator:
    """Reject passwords that appear in HIBP's breach corpus."""

    def __init__(self, min_breach_count: int = 1) -> None:
        # Setting this above 1 lets you tolerate passwords seen rarely;
        # default 1 = "any breach hit is a fail".
        self.min_breach_count = min_breach_count

    def validate(self, password: str, user: object | None = None) -> None:
        if getattr(settings, "TESTING", False):
            return  # tests run offline; don't hit external APIs

        # SHA1 is mandated by the HaveIBeenPwned "range" API, which uses
        # k-anonymity (we send the first 5 hex chars, never the password).
        # Not a password-storage hash — usedforsecurity=False silences bandit.
        sha1 = hashlib.sha1(password.encode("utf-8"), usedforsecurity=False).hexdigest().upper()
        prefix, suffix = sha1[:5], sha1[5:]

        try:
            response = httpx.get(
                _HIBP_RANGE_URL.format(prefix=prefix),
                timeout=_TIMEOUT_SECONDS,
                headers={"Add-Padding": "true", "User-Agent": "pdf-editor-signup"},
            )
            response.raise_for_status()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            logger.warning("HIBP lookup failed (%s); allowing password by fail-open policy", exc)
            return

        for line in response.text.splitlines():
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            hash_suffix, count_str = parts
            if hash_suffix.strip().upper() != suffix:
                continue
            try:
                count = int(count_str.strip())
            except ValueError:
                continue
            if count >= self.min_breach_count:
                raise ValidationError(
                    _(
                        "This password has appeared in a public data breach. "
                        "Choose a different one — using it puts your account at risk."
                    ),
                    code="password_pwned",
                )
            return

    def get_help_text(self) -> str:
        return _("Your password must not appear in known public data breaches.")
