"""Liveness + readiness endpoints + staff health dashboard."""

from __future__ import annotations

from datetime import timedelta

from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.db import connection
from django.db.models import Count, Sum
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.utils import timezone

from ..models import ProcessedPDF, UploadedPDF


def healthz(request: HttpRequest) -> JsonResponse:
    """Lightweight liveness check — always 200 if the process is alive."""
    return JsonResponse({"status": "ok"})


def readyz(request: HttpRequest) -> JsonResponse:
    """Readiness — verifies this instance can reach the stateful backends it
    needs to actually serve: Postgres (via pgbouncer) and Redis (which backs
    the cache, sessions, the Celery broker, and the rate-limit store).

    Returns 503 if either is unreachable. NOTE: the internal nginx LB is
    stock (not Plus), so it does *passive* failover only — it never polls
    /readyz — so a 503 here can't yank a live node out of rotation on its
    own. This endpoint is consumed by the post-deploy smoke test and external
    uptime monitors, for which "degraded when a backend is down" is the
    honest, actionable answer.
    """
    checks: dict[str, str] = {}
    overall_ok = True

    try:
        with connection.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001 — readiness must never itself 500
        checks["database"] = f"error: {exc}"
        overall_ok = False

    # Redis reachability through Django's cache — the same Redis server also
    # backs the Celery broker and session store, so a round-trip here proves
    # the whole Redis dependency. (In tests/dev with LocMemCache this is a
    # local no-op that always passes, which is correct — there's no Redis to
    # be down.)
    try:
        canary = f"readyz-{id(request)}"
        cache.set(canary, "1", 5)
        if cache.get(canary) != "1":
            raise RuntimeError("cache round-trip returned unexpected value")
        cache.delete(canary)
        checks["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["redis"] = f"error: {exc}"
        overall_ok = False

    if not overall_ok:
        return JsonResponse({"status": "degraded", "checks": checks}, status=503)
    return JsonResponse({"status": "ok", "checks": checks})


@staff_member_required
def admin_health_view(request: HttpRequest) -> HttpResponse:
    """Staff-only dashboard with aggregate platform stats."""
    User = get_user_model()
    now = timezone.now()
    week_ago = now - timedelta(days=7)

    db_ok = True
    db_error = ""
    try:
        with connection.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    except Exception as exc:
        db_ok = False
        db_error = str(exc)

    total_users = User.objects.count()
    active_users = User.objects.filter(is_active=True).count()
    new_users_week = User.objects.filter(date_joined__gte=week_ago).count()

    uploaded_total = UploadedPDF.objects.count()
    uploaded_week = UploadedPDF.objects.filter(uploaded_at__gte=week_ago).count()
    uploaded_size = UploadedPDF.objects.aggregate(s=Sum("size"))["s"] or 0

    processed_total = ProcessedPDF.objects.count()
    processed_week = ProcessedPDF.objects.filter(created_at__gte=week_ago).count()
    processed_size = ProcessedPDF.objects.aggregate(s=Sum("size"))["s"] or 0

    kind_breakdown = list(ProcessedPDF.objects.values("kind").annotate(count=Count("id")).order_by("-count"))
    kind_label = dict(ProcessedPDF.KIND_CHOICES)
    for row in kind_breakdown:
        row["label"] = kind_label.get(row["kind"], row["kind"])

    return render(
        request,
        "pdfeditor/admin_health.html",
        {
            "db_ok": db_ok,
            "db_error": db_error,
            "total_users": total_users,
            "active_users": active_users,
            "new_users_week": new_users_week,
            "uploaded_total": uploaded_total,
            "uploaded_week": uploaded_week,
            "uploaded_size": uploaded_size,
            "processed_total": processed_total,
            "processed_week": processed_week,
            "processed_size": processed_size,
            "total_size": uploaded_size + processed_size,
            "kind_breakdown": kind_breakdown,
        },
    )
