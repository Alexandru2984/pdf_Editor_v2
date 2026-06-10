"""Custom Prometheus metrics for PDF Editor.

These complement what ``django_prometheus`` already exports (HTTP request
counts + latency, DB query timings, etc.) with business-level signals:
how many PDFs we processed, how many jobs are queued vs done, RAG chat
volume, etc.

Counters never decrement — they reset to zero on worker restart and
Prometheus computes rate() over the time series. Histograms record
observed values for percentile queries.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ---- Operations ----------------------------------------------------------

OP_TOTAL = Counter(
    "pdfeditor_op_total",
    "PDF operations executed, labelled by kind (compress, merge, …) and outcome.",
    ["kind", "outcome"],
)

OP_DURATION_SECONDS = Histogram(
    "pdfeditor_op_duration_seconds",
    "Wall-clock time spent inside a PDF op, labelled by kind.",
    ["kind"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0),
)

# ---- Async jobs ----------------------------------------------------------

JOB_TOTAL = Counter(
    "pdfeditor_job_total",
    "Async jobs by kind and final status (done/failed).",
    ["kind", "status"],
)

JOB_QUEUE_DEPTH = Gauge(
    "pdfeditor_job_queue_depth",
    "Jobs sitting in queued or running state right now (sampled per scrape).",
    ["status"],
)

# ---- RAG / Chat ----------------------------------------------------------

EMBEDDINGS_CREATED = Counter(
    "pdfeditor_embeddings_created_total",
    "Embedding rows inserted (one per chunk).",
)

CHAT_MESSAGE_TOTAL = Counter(
    "pdfeditor_chat_message_total",
    "Chat messages sent to the LLM, labelled by model id and outcome.",
    ["model", "outcome"],
)

CHAT_LATENCY_SECONDS = Histogram(
    "pdfeditor_chat_latency_seconds",
    "End-to-end chat round trip (embed query + retrieve + LLM call).",
    ["model"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0),
)

# ---- Auth / uploads ------------------------------------------------------

UPLOAD_TOTAL = Counter(
    "pdfeditor_upload_total",
    "PDF uploads accepted vs rejected.",
    ["outcome"],
)

API_KEY_AUTH_TOTAL = Counter(
    "pdfeditor_api_key_auth_total",
    "API key auth attempts by outcome (success / revoked / invalid).",
    ["outcome"],
)

CSP_VIOLATION_TOTAL = Counter(
    "pdfeditor_csp_violation_total",
    "Content-Security-Policy violation reports received from browsers, "
    "labelled by the violated directive (script-src, style-src, …).",
    ["directive"],
)


def refresh_queue_depth() -> None:
    """Recompute the JOB_QUEUE_DEPTH gauge from the database. Called by
    the metrics endpoint on every scrape so the gauge stays fresh."""
    from .models import Job

    counts = {Job.STATUS_QUEUED: 0, Job.STATUS_RUNNING: 0}
    for row in Job.objects.filter(status__in=counts.keys()).values_list("status", flat=True):
        counts[row] = counts.get(row, 0) + 1
    for status, n in counts.items():
        JOB_QUEUE_DEPTH.labels(status=status).set(n)
