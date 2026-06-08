"""Prometheus scrape endpoint, gated by an IP allowlist.

The endpoint is intentionally NOT under ``/admin/`` so a misconfigured
auth layer can't accidentally expose it — only IPs in
``PROMETHEUS_METRICS_ALLOW`` see real metrics. Everyone else gets 404.

Allowlist entries may be single IPs (``127.0.0.1``) or CIDR ranges
(``172.16.0.0/12``). CIDR is what makes scraping work inside docker
compose, where the Prometheus container's IP is dynamic.
"""

from __future__ import annotations

import ipaddress

from django.conf import settings
from django.http import Http404, HttpResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..metrics import refresh_queue_depth
from ..netutils import client_ip


def _ip_allowed(ip: str, allowlist: set[str]) -> bool:
    if not ip:
        return False
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for entry in allowlist:
        try:
            if "/" in entry:
                if addr in ipaddress.ip_network(entry, strict=False):
                    return True
            elif addr == ipaddress.ip_address(entry):
                return True
        except ValueError:
            continue
    return False


def metrics_view(request):
    """Return Prometheus text-format metrics. 404 to non-allowlisted IPs."""
    allowlist: set[str] = getattr(settings, "PROMETHEUS_METRICS_ALLOW", set())
    if allowlist:
        # client_ip() reads X-Forwarded-For from the RIGHT by trusted-proxy
        # count, so a forged `X-Forwarded-For: 127.0.0.1` can't satisfy the
        # allowlist. Prometheus scrapes web:8000 directly (no XFF), so it
        # falls back to REMOTE_ADDR — its container IP, covered by the CIDR.
        if not _ip_allowed(client_ip(request) or "", allowlist):
            raise Http404("not found")
    elif not settings.DEBUG:
        # No allowlist configured AND we're in prod — refuse rather than
        # accidentally expose. Operator must explicitly opt in.
        raise Http404("not found")

    # Refresh DB-derived gauges before serialising.
    refresh_queue_depth()
    return HttpResponse(generate_latest(), content_type=CONTENT_TYPE_LATEST)
