"""Profile "Sessions & security" page: list, revoke, log out everywhere.

Backed by :class:`~pdfeditor.models.UserSession` rows that the auth signals
in ``signals.py`` maintain at login/logout time. Revocation deletes the
backing Django session through the configured SESSION_ENGINE (so it works
for db and cached_db alike) — the revoked browser is logged out on its very
next request.
"""

from __future__ import annotations

from importlib import import_module

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.sessions.models import Session
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.utils.translation import gettext as _
from django.views.decorators.http import require_http_methods

from ..models import UserSession


def _kill_django_session(session_key: str) -> None:
    """Delete the backing session via the configured engine (cache + DB)."""
    engine = import_module(settings.SESSION_ENGINE)
    engine.SessionStore(session_key).delete()


def _device_label(user_agent: str) -> str:
    """Tiny best-effort UA → 'Browser on OS' label. No parsing dependency —
    this is cosmetic, the raw UA is shown as the tooltip."""
    ua = user_agent.lower()
    if "edg/" in ua or "edge" in ua:
        browser = "Edge"
    elif "firefox" in ua:
        browser = "Firefox"
    elif "chrome" in ua or "crios" in ua:
        browser = "Chrome"
    elif "safari" in ua:
        browser = "Safari"
    else:
        browser = _("Unknown browser")
    if "android" in ua:
        os_name = "Android"
    elif "iphone" in ua or "ipad" in ua or "ios" in ua:
        os_name = "iOS"
    elif "windows" in ua:
        os_name = "Windows"
    elif "mac os" in ua or "macintosh" in ua:
        os_name = "macOS"
    elif "linux" in ua:
        os_name = "Linux"
    else:
        os_name = _("unknown OS")
    return f"{browser} · {os_name}"


def _prune_dead_sessions(user) -> None:
    """Drop UserSession rows whose backing session expired or vanished."""
    keys = list(UserSession.objects.filter(user=user).values_list("session_key", flat=True))
    if not keys:
        return
    live = set(
        Session.objects.filter(session_key__in=keys, expire_date__gt=timezone.now()).values_list(
            "session_key", flat=True
        )
    )
    dead = [k for k in keys if k not in live]
    if dead:
        UserSession.objects.filter(user=user, session_key__in=dead).delete()


@login_required
def security_sessions_view(request: HttpRequest) -> HttpResponse:
    _prune_dead_sessions(request.user)
    current_key = request.session.session_key or ""
    UserSession.objects.filter(session_key=current_key).update(last_seen=timezone.now())

    sessions = [
        {
            "row": row,
            "device": _device_label(row.user_agent),
            "is_current": row.session_key == current_key,
        }
        for row in UserSession.objects.filter(user=request.user)
    ]
    # Current session first, then most recently seen.
    sessions.sort(key=lambda s: (not s["is_current"],))
    return render(request, "pdfeditor/security_sessions.html", {"sessions": sessions})


@login_required
@require_http_methods(["POST"])
def revoke_session_view(request: HttpRequest, session_id) -> HttpResponse:
    row = UserSession.objects.filter(user=request.user, id=session_id).first()
    if row is None:
        messages.error(request, _("Session not found."))
        return redirect("security_sessions")
    if row.session_key == request.session.session_key:
        messages.error(request, _("Use the logout button to end your current session."))
        return redirect("security_sessions")
    _kill_django_session(row.session_key)
    row.delete()
    messages.success(request, _("Session revoked — that device is now signed out."))
    return redirect("security_sessions")


@login_required
@require_http_methods(["POST"])
def revoke_other_sessions_view(request: HttpRequest) -> HttpResponse:
    current_key = request.session.session_key or ""
    others = UserSession.objects.filter(user=request.user).exclude(session_key=current_key)
    count = 0
    for row in others:
        _kill_django_session(row.session_key)
        count += 1
    others.delete()
    messages.success(request, _("Signed out everywhere else (%(count)d session(s)).") % {"count": count})
    return redirect("security_sessions")
