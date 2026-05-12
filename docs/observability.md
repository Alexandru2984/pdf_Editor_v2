# Observability

This project ships with three layers of operational visibility:

1. **Sentry** — error tracking + distributed tracing (10% sampled).
2. **Prometheus** — request/op/job metrics scraped from `/metrics`.
3. **Grafana** — pre-built dashboard import.

All three are **opt-in** — set the right env var(s) and the wiring activates;
leave them empty and the app runs unchanged.

## Sentry

The SDK is already wired in `pdf_project/settings.py`. To enable:

1. Sign up at <https://sentry.io> (free tier covers small projects).
2. Create a new **Django** project. Sentry shows you a DSN like
   `https://abc123@o0.ingest.us.sentry.io/12345`.
3. Add it to `.env`:
   ```env
   SENTRY_DSN=https://abc123@o0.ingest.us.sentry.io/12345
   SENTRY_ENVIRONMENT=production   # optional, defaults to "production"/"development"
   SENTRY_TRACES_SAMPLE_RATE=0.1   # optional, 10% by default
   ```
4. Restart the `web` and `worker` containers.

The integrations enabled:

- `DjangoIntegration` — captures unhandled exceptions, attaches request +
  user context (`send_default_pii=False` so emails/IPs stay off).
- `LoggingIntegration` — `INFO`+ logs become breadcrumbs, `ERROR`+ become
  events. Combined with the request-ID middleware, every event carries a
  `request_id` tag you can pivot on.

## Prometheus

The app exposes `/metrics` in the Prometheus text exposition format,
serving both the **standard `django_prometheus` series** (request counts +
latency by view + method + status, DB query timings, cache stats) and the
**custom business-level series** defined in `pdfeditor/metrics.py`:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `pdfeditor_op_total` | counter | `kind`, `outcome` | Every PDF op execution (success / failure). |
| `pdfeditor_op_duration_seconds` | histogram | `kind` | Wall-clock time spent inside the op. |
| `pdfeditor_job_total` | counter | `kind`, `status` | Async job terminal outcomes. |
| `pdfeditor_job_queue_depth` | gauge | `status` | Live queue depth (sampled per scrape). |
| `pdfeditor_embeddings_created_total` | counter | — | RAG chunks indexed. |
| `pdfeditor_chat_message_total` | counter | `model`, `outcome` | Chat queries by LLM model. |
| `pdfeditor_chat_latency_seconds` | histogram | `model` | End-to-end chat round-trip. |
| `pdfeditor_upload_total` | counter | `outcome` | Upload accept/reject counts. |
| `pdfeditor_api_key_auth_total` | counter | `outcome` | API key auth attempts. |

### Securing `/metrics`

The endpoint is gated by an **IP allowlist** — only requests from listed
IPs see metrics, everyone else gets `404`. Configure via env:

```env
PROMETHEUS_METRICS_ALLOW=127.0.0.1,10.0.0.5
```

Defaults to `127.0.0.1`. In production this means only a scraper running
**on the same host** (or via Grafana Cloud agent with a known egress IP)
can reach `/metrics`.

### Local Prometheus setup

If you want to run Prometheus alongside the app in compose, drop this into
your `docker-compose.yml`:

```yaml
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    ports:
      - "127.0.0.1:9090:9090"
```

And `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: pdfeditor
    scrape_interval: 30s
    metrics_path: /metrics
    static_configs:
      - targets: ["web:8000"]
```

(Add `web` to `PROMETHEUS_METRICS_ALLOW` because the scraper will appear
with the docker network's internal IP.)

## Grafana

A ready-made dashboard lives at `docs/grafana-dashboard.json`.

Import via:

1. Grafana → Dashboards → **New** → **Import**.
2. Paste the JSON or upload the file.
3. Select your Prometheus data source.

The dashboard has 4 stat tiles (req/s, p95 latency, 5xx ratio, jobs in
flight) plus 7 time-series panels covering ops, job outcomes, chat latency,
embedding throughput, and queue depth. Refresh defaults to 30s.

### Quick-start with Grafana Cloud (free tier)

Grafana Cloud's free tier handles ~10k metrics, more than enough for this
app. Setup:

1. Sign up at <https://grafana.com> → create a stack.
2. Open **Connections** → **Prometheus (remote_write)** → copy the push URL,
   user, and API key.
3. Run Grafana Agent in compose (alongside `web`/`worker`) pointed at
   `/metrics`, scraping every 30s and pushing to the Cloud URL.
4. Import the dashboard JSON into your Cloud Grafana.
