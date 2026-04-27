"""Liveness + readiness endpoints for load balancers and orchestrators."""

from __future__ import annotations

from django.db import OperationalError, connection
from django.http import HttpRequest, JsonResponse


def healthz(request: HttpRequest) -> JsonResponse:
    """Lightweight liveness check — always 200 if the process is alive."""
    return JsonResponse({"status": "ok"})


def readyz(request: HttpRequest) -> JsonResponse:
    """Readiness check — verifies that we can talk to the primary database."""
    payload: dict[str, object] = {"status": "ok", "checks": {}}
    overall_ok = True

    try:
        with connection.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        payload["checks"]["database"] = "ok"
    except OperationalError as exc:
        payload["checks"]["database"] = f"error: {exc}"
        overall_ok = False
    except Exception as exc:
        payload["checks"]["database"] = f"error: {exc}"
        overall_ok = False

    if not overall_ok:
        payload["status"] = "degraded"
        return JsonResponse(payload, status=503)

    return JsonResponse(payload)
