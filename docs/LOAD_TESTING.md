# Load Testing — PDF Editor

`docs/SLOs.md` defines what "healthy" looks like (p95 < 1.0s, 99.9%
availability). This doc is how we *prove* the stack actually meets those
numbers, and how we catch a regression on a future deploy before users do.

## TL;DR

```bash
# 1. (Optional but recommended) Disable rate limiting so the test isn't
#    measuring the throttle. The stack already supports this flag.
RATELIMIT_ENABLE=0 docker compose up -d web nginx

# 2. Run the test.
docker run --rm --network=host \
    -v "$PWD/scripts:/scripts" \
    -e BASE_URL=http://localhost:8000 \
    grafana/k6:latest run /scripts/load_test.js

# 3. Re-enable rate limiting when done.
docker compose up -d web nginx
```

A run takes about 4 minutes. The output ends with a thresholds summary —
green checkmarks means the SLOs hold under the test load. Red means
something needs investigating before the next deploy.

## What the script does

Two parallel scenarios (`scripts/load_test.js`):

| Scenario | Endpoints | Peak VUs | What it proves |
|----------|-----------|----------|----------------|
| `anon_browse` | `/healthz`, `/`, `/api/v1/` | 50 | Anonymous traffic — heaviest by volume in real life. If the dashboard template or readiness check slows down, this catches it. |
| `auth_read`   | `/api/v1/outputs/`, `/api/v1/jobs/` | 20 | Authenticated DRF reads — exercises the heaviest serializer + the most-used indexes. Only runs when `API_KEY` is set. |

Each scenario ramps over 4 minutes:

```
30s →  warm up
1m  →  ramp to peak
2m  →  HOLD AT PEAK   ← the window the SLO thresholds are measured against
30s →  ramp down
```

The hold window is the SLO-validating one. Anything shorter than ~90s
won't produce statistically meaningful p95 numbers — that's why `SHORT=1`
mode is explicitly marked as smoke-only.

## Thresholds (mapped to SLOs)

| k6 threshold | Where it maps |
|--------------|---------------|
| `http_req_failed < 0.001` | SLO 1 (API availability ≥ 99.9%). Redefined via `setResponseCallback` to count only 5xx + network errors, ignoring 4xx — same as the SLO. |
| `http_req_duration p(95) < 1000` | SLO 2 (API latency p95 < 1.0s). |
| `http_req_duration p(99) < 2500` | Tail-latency guardrail. Not an SLO; pages start to feel unusable beyond ~2.5s. |
| `http_req_duration{scenario:anon_browse} p(95) < 500` | Public pages are cheaper than authed reads — hold them tighter. |

If any threshold fails, k6 exits non-zero. That's the regression signal.

## Baseline

Recorded against the dev stack on a quiet machine (`docker compose up`
with 2 `web` replicas, single Postgres, no co-located workload):

| Metric | Result |
|--------|--------|
| Peak concurrent VUs | 70 (50 anon + 20 authed) |
| Throughput | ~17 req/s sustained |
| p95 latency | 48 ms |
| p99 latency | 57 ms |
| Max latency | 70 ms |
| 5xx rate | 0.00% |

These are floor numbers — production traffic will look worse. Use them
as a "this is what fresh-and-quiet looks like" reference, not a target.
A run that returns p95 = 800ms isn't broken, but it's worth asking why
that delta exists (cold caches, contended host, slow DB I/O).

## When to run

- **Before every release that touches the request path.** New middleware,
  new DRF view, schema changes — anything that could change the latency
  shape. Run it locally; commit the new baseline if it shifted.
- **After every deploy** that touched infra (nginx, gunicorn workers,
  PgBouncer pool size). Capture a "post-deploy" baseline.
- **Quarterly** as a routine drift-check, alongside the DR drill
  (`docs/DR.md`). Calendar it.
- **Never** against prod during business hours without coordinating
  first — the run shows up in Grafana, and a hot SLI graph during a real
  incident is the last thing you want.

## Reading the output

The interesting block is `█ TOTAL RESULTS` near the end:

- `checks_total` / `checks_succeeded` — assertion-level signal. 100% is
  the expectation; anything less means a specific endpoint is unhappy
  under load.
- `http_req_failed` — actual 5xx + network errors, expressed as a rate.
  Compare directly to SLO 1 (≤ 0.1%).
- `http_req_duration` — the latency distribution. `p(95)` is the SLO
  number; `max` tells you about the worst single request, which usually
  reveals a slow query or a cold path.
- The per-scenario filters (`{scenario:anon_browse}`) split the numbers
  so a slow authed endpoint can't drag a fast public one down (and vice
  versa) into a single average.

If `http_req_failed` is non-zero, look at the per-status breakdown:

```bash
# Add --summary-export so you can pull rates by status code post-hoc.
k6 run --summary-export=/tmp/k6-summary.json scripts/load_test.js
jq '.metrics' /tmp/k6-summary.json
```

## Known limits

- **No write traffic.** The script doesn't upload PDFs or run ops. Those
  are job-based and not bound by the request-path SLO — exercising them
  here would just stress Celery and obscure the API numbers we care
  about. The smoke E2E suite (`pdfeditor/tests_e2e/`) covers the write
  path functionally; a future load test could add a separate scenario
  if we ever need throughput numbers for the job pipeline.
- **Localhost ≠ prod.** Numbers from `BASE_URL=http://localhost:8000`
  miss the host nginx layer, real TLS termination, and inter-host
  network latency. Treat them as a lower bound.
- **No coordinated omission correction.** k6 does not back-calculate
  latency for requests that were never sent because a prior one was
  slow. For latency-sensitive systems this matters; ours is web-app
  scale, so we're fine without it.

## Related

- `scripts/load_test.js` — the k6 script
- `docs/SLOs.md` — the SLOs the thresholds map to
- `docs/RUNBOOK.md` — what to do when an SLO breach pages someone
- `pdfeditor/tests_e2e/` — functional E2E (passes/fails, not how-fast)
