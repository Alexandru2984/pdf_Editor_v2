"""TOTP multi-factor auth helpers (pyotp-backed).

Optional, per-user. A user enrols an authenticator app, confirms one code,
and gets single-use backup codes. From then on login requires a code. Secrets
live on :class:`~pdfeditor.models.MfaDevice`; backup codes are stored only as
SHA-256 hashes.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import secrets
import time

import pyotp
import qrcode
from django.conf import settings

from .models import MfaBackupCode, MfaDevice

BACKUP_CODE_COUNT = 10
TOTP_STEP_SECONDS = 30


def user_has_mfa(user) -> bool:
    """True if the user has a *confirmed* authenticator (i.e. login is gated)."""
    return bool(
        user
        and getattr(user, "is_authenticated", False)
        and MfaDevice.objects.filter(user=user, confirmed=True).exists()
    )


def provisioning_uri(device: MfaDevice, username: str) -> str:
    """otpauth:// URI to encode in the enrolment QR code."""
    issuer = getattr(settings, "MFA_ISSUER", "PDF Editor")
    return pyotp.TOTP(device.secret).provisioning_uri(name=username, issuer_name=issuer)


def qr_data_uri(otpauth_uri: str) -> str:
    """Render ``otpauth_uri`` to a PNG data: URI (img-src allows data:)."""
    img = qrcode.make(otpauth_uri)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def verify_totp(device: MfaDevice, code: str) -> bool:
    """Check a 6-digit TOTP code (±1 step tolerates clock skew), replay-safe.

    The timestep that produced an accepted code is recorded on the device via
    a conditional UPDATE; any code at or below that step is rejected, so an
    intercepted code cannot be replayed within its validity window — even by
    two concurrent requests racing each other.
    """
    code = (code or "").strip().replace(" ", "")
    if not code.isdigit():
        return False
    totp = pyotp.TOTP(device.secret)
    current_step = int(time.time()) // TOTP_STEP_SECONDS
    for offset in (1, 0, -1):
        step = current_step + offset
        if hmac.compare_digest(totp.at(step * TOTP_STEP_SECONDS), code):
            claimed = MfaDevice.objects.filter(pk=device.pk, last_verified_step__lt=step).update(
                last_verified_step=step
            )
            return bool(claimed)
    return False


def _hash_backup_code(code: str) -> str:
    return hashlib.sha256(code.strip().lower().encode("utf-8")).hexdigest()


def generate_backup_codes(device: MfaDevice, count: int = BACKUP_CODE_COUNT) -> list[str]:
    """Replace the device's backup codes; return the new plaintext set ONCE."""
    device.backup_codes.all().delete()
    # 2 × token_hex(5) = 80 bits of entropy — enough that the unsalted
    # SHA-256 hashes can't be brute-forced offline from a leaked DB dump.
    codes = ["-".join(secrets.token_hex(5) for _ in range(2)) for _ in range(count)]
    MfaBackupCode.objects.bulk_create(
        [MfaBackupCode(device=device, code_hash=_hash_backup_code(c)) for c in codes]
    )
    return codes


def consume_backup_code(device: MfaDevice, code: str) -> bool:
    """Spend one unused backup code matching ``code``. True if one was burned.

    Single conditional UPDATE so two concurrent attempts can't both spend the
    same code (read-then-write would let the second racer through).
    """
    from django.utils import timezone

    return bool(
        device.backup_codes.filter(code_hash=_hash_backup_code(code), used_at__isnull=True).update(
            used_at=timezone.now()
        )
    )
