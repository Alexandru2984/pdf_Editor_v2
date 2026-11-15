"""Receiver for browser Content-Security-Policy violation reports.

The strict CSP set by ``SecurityHeadersMiddleware`` carries a
``report-uri /csp-report/`` directive; browsers POST a JSON document here
every time the policy blocks something. Reports become a structured log line
(for drill-down) and a Prometheus counter labelled by violated directive
(for alerting) — a policy regression after a deploy shows up as a counter
spike instead of silent breakage on users' machines.

Browsers send two wire formats, both handled:

* legacy ``application/csp-report``: ``{"csp-report": {...}}``
* Reporting API v1: ``[{"type": "csp-violation", "body": {...}}, ...]``
"""

from __future__ import annotations

import json
import logging
from urllib.parse import urlsplit

from django.http import HttpRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from ..metrics import CSP_VIOLATION_TOTAL
from ..ratelimiting import auth_aware_ratelimit

logger = logging.getLogger(__name__)

# A genuine report is well under 2 KB; cap hard so the endpoint can't be
# used to firehose the log pipeline.
_MAX_REPORT_BYTES = 16 * 1024
_MAX_REPORTS_PER_POST = 10
_DIRECTIVES = frozenset(
    {
        "base-uri",
        "child-src",
        "connect-src",
        "default-src",
        "font-src",
        "form-action",
        "frame-ancestors",
        "frame-src",
        "img-src",
        "manifest-src",
        "media-src",
        "object-src",
        "report-to",
        "report-uri",
        "require-trusted-types-for",
        "sandbox",
        "script-src",
        "script-src-attr",
        "script-src-elem",
        "style-src",
        "style-src-attr",
        "style-src-elem",
        "trusted-types",
        "upgrade-insecure-requests",
        "worker-src",
    }
)


def _report_origin(value: str) -> str:
    """Keep diagnostics without persisting tokens, filenames or URL credentials."""
    if value in {"inline", "eval", "wasm-eval", "self"}:
        return value
    try:
        parsed = urlsplit(value)
        if parsed.scheme in {"data", "blob"}:
            return parsed.scheme
        if parsed.scheme in {"https", "http"} and parsed.hostname:
            host = parsed.hostname
            if ":" in host:
                host = f"[{host}]"
            return f"{parsed.scheme}://{host}"[:200]
    except ValueError:
        pass
    return "unknown"


def _normalise(payload: object) -> list[dict]:
    """Extract the violation dicts from either wire format."""
    if isinstance(payload, dict):
        body = payload.get("csp-report")
        return [body] if isinstance(body, dict) else []
    if isinstance(payload, list):
        out = []
        for item in payload[:_MAX_REPORTS_PER_POST]:
            if isinstance(item, dict) and item.get("type") == "csp-violation":
                body = item.get("body")
                if isinstance(body, dict):
                    out.append(body)
        return out
    return []


def _field(report: dict, *names: str) -> str:
    """First present key (legacy and Reporting API use different spellings)."""
    for name in names:
        value = report.get(name)
        if isinstance(value, str) and value:
            return value
    return "unknown"


@csrf_exempt
@auth_aware_ratelimit(anon_rate="60/h", user_rate="60/h", method="POST")
@require_http_methods(["POST"])
def csp_report_view(request: HttpRequest) -> HttpResponse:
    """Record CSP violation reports. Always 204 — never an oracle."""
    raw = request.body[: _MAX_REPORT_BYTES + 1]
    if len(raw) > _MAX_REPORT_BYTES:
        return HttpResponse(status=204)
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return HttpResponse(status=204)

    for report in _normalise(payload):
        directive = _field(report, "effective-directive", "effectiveDirective", "violated-directive")
        # Length limits do not bound the number of attacker-chosen series.
        # Collapse all unrecognised labels to a single fixed bucket.
        directive = directive.partition(" ")[0]
        if directive not in _DIRECTIVES:
            directive = "unknown"
        CSP_VIOLATION_TOTAL.labels(directive=directive).inc()
        logger.warning(
            "CSP violation: directive=%r blocked=%r page=%r source=%r",
            directive,
            _report_origin(_field(report, "blocked-uri", "blockedURL")),
            _report_origin(_field(report, "document-uri", "documentURL")),
            _report_origin(_field(report, "source-file", "sourceFile")),
        )
    return HttpResponse(status=204)
