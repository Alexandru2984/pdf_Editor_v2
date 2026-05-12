# Load testing

[Locust](https://locust.io) scenarios for hammering the PDF Editor.

## Setup

```bash
cd scripts/loadtest
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run

Interactive (web UI at <http://localhost:8089>):

```bash
locust --host=http://localhost:8000
```

Headless, with a written HTML report:

```bash
locust --host=http://localhost:8000        \
       --users=50 --spawn-rate=5            \
       --run-time=2m --headless             \
       --html report.html
```

For the authenticated API profile (`ApiClient`), grab an API key from
your profile page and:

```bash
LOCUST_API_KEY=your_token_here locust ...
```

## Profiles

| Profile | Weight | Description |
|---|---|---|
| `Visitor` | 5 | Anonymous browse — dashboard + API docs + healthz. Cheap. |
| `Uploader` | 3 | Anonymous: upload a 1-page PDF, then compress it. Subject to anon rate limits (`20/h` for compress). |
| `ApiClient` | 2 | Authenticated via `X-API-Key`. List PDFs/outputs + upload + compress. Subject to per-key throttle (300/h). Disabled unless `LOCUST_API_KEY` is set. |

The weights mean: for every 10 simulated users, ~5 will be Visitors, 3
Uploaders, 2 ApiClients.

## Tips

- **Never aim this at production above ~10 concurrent users** without
  warning the owner. You'll hit `auth_aware_ratelimit` quickly and
  numbers will reflect the limiter, not the app.
- The async ops (OCR, PDF/A, Compare, Chat) are excluded on purpose —
  they take seconds to minutes and locust isn't great at measuring those
  via polling. Use Grafana to track per-op latency under realistic load.
- Set `LOCUST_PDF_BYTES=/path/to/real.pdf` to test with a heavier
  fixture. The default is a 1-page generated PDF (~1 KB).
- For longer runs, redirect locust's CSV output: add
  `--csv=baseline-$(date +%s)` and you'll get per-endpoint percentile
  CSVs to commit as benchmark history.

## Baseline expectations

Measured against a single-instance deployment with 3 gunicorn workers,
2 Celery workers, 1 Postgres + 1 Redis on a 4-core VPS:

| Endpoint | p50 | p95 | p99 | Notes |
|---|---|---|---|---|
| `GET /` | 60ms | 150ms | 300ms | Static-heavy, mostly cache |
| `GET /api/v1/` | 30ms | 90ms | 180ms | Trivial JSON |
| `GET /healthz/` | 5ms | 20ms | 40ms | DB ping only |
| `POST /upload/` (1 KB PDF) | 120ms | 400ms | 800ms | Includes magic-byte + page-count validation |
| `POST /compress/` | 250ms | 700ms | 1500ms | PyMuPDF rewrite + image re-encode |
| `POST /api/v1/ops/compress/` | 200ms | 600ms | 1200ms | Same op, faster path (no template render) |

Numbers are illustrative — record your own with the `--csv` flag once
you've run a baseline.
