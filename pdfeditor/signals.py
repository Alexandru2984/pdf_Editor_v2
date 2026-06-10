"""Filesystem cleanup + session-tracking signals.

`cleanup_old_pdfs` is the catch-all sweep that runs on a cron and removes
files older than the retention window. These signals are the *fast path* —
when a row is deleted in real time (user action, cascade from User delete,
admin delete), we delete the matching file/thumbnail immediately so the
filesystem never drifts ahead of the database.

The auth signals at the bottom maintain :class:`~pdfeditor.models.UserSession`
rows (one per active login) for the profile "Sessions & security" page, and
send a best-effort alert email when a sign-in comes from a device the account
has never used before.
"""

from __future__ import annotations

import logging
import os

from django.contrib.auth.signals import user_logged_in, user_logged_out
from django.db.models.signals import post_delete
from django.dispatch import receiver

from .models import ProcessedPDF, UploadedPDF, UserSession

logger = logging.getLogger(__name__)


def _unlink(path: str | None, label: str) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Failed to remove %s %s: %s", label, path, exc)


@receiver(post_delete, sender=UploadedPDF)
def _on_uploaded_pdf_delete(sender, instance: UploadedPDF, **_):
    _unlink(instance.path, "uploaded PDF")
    # Local import avoids circular: views.upload imports from models.
    from .views.upload import _thumbnail_path

    _unlink(_thumbnail_path(instance.id), "thumbnail")


@receiver(post_delete, sender=ProcessedPDF)
def _on_processed_pdf_delete(sender, instance: ProcessedPDF, **_):
    _unlink(instance.path, "processed PDF")
    if instance.r2_key:
        from . import objectstore

        objectstore.delete_object(instance.r2_key)


# --------------------------------------------------------------------------
# Session tracking (profile "Sessions & security" page)
# --------------------------------------------------------------------------


def _send_new_device_alert(user, ip: str | None, user_agent: str) -> None:
    """Best-effort 'new sign-in' email. Failures are logged, never raised."""
    if not user.email:
        return
    from django.conf import settings
    from django.core.mail import send_mail
    from django.utils import timezone
    from django.utils.translation import gettext as _

    try:
        send_mail(
            subject=_("New sign-in to your PDF Editor account"),
            message=_(
                "Your account %(username)s just signed in from a device we "
                "haven't seen before.\n\n"
                "  Time: %(time)s (UTC)\n"
                "  IP address: %(ip)s\n"
                "  Browser: %(agent)s\n\n"
                "If this was you, no action is needed. If not, change your "
                "password immediately and review your active sessions at "
                "%(site)s/accounts/security/sessions/."
            )
            % {
                "username": user.get_username(),
                "time": timezone.now().strftime("%Y-%m-%d %H:%M"),
                "ip": ip or _("unknown"),
                "agent": user_agent[:200] or _("unknown"),
                "site": getattr(settings, "SITE_URL", ""),
            },
            from_email=None,  # DEFAULT_FROM_EMAIL
            recipient_list=[user.email],
            fail_silently=True,
        )
    except Exception as exc:  # noqa: BLE001 — alerting must never block login
        logger.warning("New-device alert email failed for %s: %s", user.pk, exc)


@receiver(user_logged_in)
def _on_user_logged_in(sender, request, user, **_):
    from .netutils import client_ip

    # login() cycles the session key but doesn't persist it yet; force a
    # save so there is a key to record.
    if not request.session.session_key:
        request.session.save()
    session_key = request.session.session_key
    if not session_key:
        return

    ip = client_ip(request)
    user_agent = request.META.get("HTTP_USER_AGENT", "")[:300]

    # "New device" = this account has prior sessions, none from this browser.
    known = UserSession.objects.filter(user=user)
    is_new_device = known.exists() and not known.filter(user_agent=user_agent).exists()

    UserSession.objects.update_or_create(
        session_key=session_key,
        defaults={"user": user, "ip_address": ip, "user_agent": user_agent},
    )
    if is_new_device:
        _send_new_device_alert(user, ip, user_agent)


@receiver(user_logged_out)
def _on_user_logged_out(sender, request, user, **_):
    session_key = getattr(request.session, "session_key", None)
    if session_key:
        UserSession.objects.filter(session_key=session_key).delete()
