"""Liveness + readiness endpoints for load balancers and orchestrators."""

from __future__ import annotations

from django.db import OperationalError, connection
from django.http import HttpRequest, JsonResponse


def healthz(request: HttpRequest) -> JsonResponse:
    """Lightweight liveness check — always 200 if the process is alive."""
    return JsonResponse({"status": "ok"})


def readyz(request: HttpRequest) -> JsonResponse:
    """Readiness check — verifies that we can talk to the primary database."""
    checks: dict[str, str] = {}
    overall_ok = True

    try:
        with connection.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        checks["database"] = "ok"
    except OperationalError as exc:
        checks["database"] = f"error: {exc}"
        overall_ok = False
    except Exception as exc:
        checks["database"] = f"error: {exc}"
        overall_ok = False

    if not overall_ok:
        return JsonResponse({"status": "degraded", "checks": checks}, status=503)
    return JsonResponse({"status": "ok", "checks": checks})
