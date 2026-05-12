# Load test benchmarks

Captured against the production deployment at `https://pdf.micutu.com`
(bypassing Cloudflare via `/etc/hosts` override to nginx-localhost, so
the numbers measure the **app**, not the CDN). The VPS is a 12-core,
45 GB RAM OVH box running the docker-compose stack from the repo root.

All runs use `scripts/loadtest/realistic.py` or `stress.py`. Reproduce
with:

```bash
cd scripts/loadtest
LOCUST_API_KEY=<your-key> locust -f <file> --host=... --headless ...
```

## TL;DR

| | Default config | After tuning |
|---|---|---|
| Sustainable concurrent users | ~80 (DB conns exhausted) | **500+ before CPU saturates** |
| Peak throughput | 39 req/s (with 13% 500s) | **120 req/s (0.17% failure)** |
| DB connections at 500u | n/a (crashed at 80u) | **26 / 300** |
| Largest endpoint speedup | — | `/api/v1/outputs/` **60s → 1.2s** |

## What changed between "default" and "tuned"

1. **Postgres `max_connections`: 100 → 300** + `shared_buffers=512MB`
2. **PgBouncer transaction pooling** in front of Postgres (`POOL_MODE=transaction`, `DEFAULT_POOL_SIZE=25`, `MAX_CLIENT_CONN=1000`)
3. **Redis-backed Django cache** (was `LocMemCache` — fragmented across gunicorn workers)
4. **Cached-DB sessions** (`SESSION_ENGINE=cached_db`) — sessions read from Redis, write-through to DB
5. **DRF default pagination** (`PageNumberPagination`, page_size=50) — `/api/v1/outputs/` was returning 5713 rows uncpaged
6. **Schema endpoint cached** for 5 min — drf-spectacular's generator throws `AssertionError` under concurrent first-time generation
7. **gunicorn workers**: 3 → 8 (during stress test; back to 3 in prod since 12-core box reserves headroom for celery/db/redis containers)
8. **`RATELIMIT_ENABLE=0`** for load tests (kills both django-ratelimit and DRF throttles)

## Run history

### Baseline (script: `locustfile.py`) — 1-page PDF, compress-only

| Useri | Spawn rate | Duration | Total req | req/s | Real failures | DB conns peak | POST /upload p95 | POST /compress p95 |
|---|---|---|---|---|---|---|---|---|
| 20 | 2/s | 2 min | 1,014 | 8.5 | **0** | 60-80/100 | 120 ms | 48 ms |
| 100 (default config) | 10/s | 3 min | 7,104 | 39.5 | **1,505 × 500** | 100/100 (exhausted) | 160 ms | 80 ms |
| 100 + `max_connections=300` | 10/s | 3 min | 7,598 | 42.3 | **0** | 177/300 | 170 ms | 81 ms |
| 100 + pgbouncer | 10/s | 3 min | 7,569 | 42.1 | **0** | **10/300** | 240 ms | 130 ms |
| 200 + pgbouncer | 20/s | 3 min | 14,230 | 79.1 | **0** | 22/300 | 1,400 ms | 370 ms |

The "100 + default config" row is the failure-mode bookmark: Postgres
default `max_connections=100` exhausted at ~80 concurrent users and
returned `OperationalError: sorry, too many clients already`, which
Django serializes as HTTP 500.

### Realistic (script: `realistic.py`) — 3 PDF sizes, 7 ops mixed UI+API

| Useri | Spawn rate | Duration | Total req | req/s | Failures | Notes |
|---|---|---|---|---|---|---|
| 100 | 10/s | 5 min | 9,391 | 31.3 | **0** | Steady-state, 4 user profiles, 4-12s think time |

Per-op p95 (5-min sustained at 100u):

| Op | UI (sync, template) | API (JSON) |
|---|---|---|
| compress | 450 ms | 300 ms |
| rotate | 500 ms | 210 ms |
| flatten | 470 ms | 370 ms |
| watermark | 220 ms | 230 ms |
| page-numbers | 250 ms | 230 ms |
| metadata | — | 340 ms |
| **to-images (rasterize @150dpi)** | **3,800 ms** | **2,000 ms** |
| upload (avg of 3 sizes) | 470 ms | 380 ms |

The API path is consistently 30–50 % faster than the equivalent
template-rendered UI flow.

### Stress (script: `stress.py`) — 0-2s think time, biased toward medium/large PDFs

| Useri | Spawn rate | Duration | Total req | req/s | Failures | 500s | Notes |
|---|---|---|---|---|---|---|---|
| 500 (8 gunicorn workers, before bug fixes) | 25/s | 5 min | 28,406 | 94.7 | **8.21 %** | 2,003 | drf-spectacular race, no pagination on `/outputs/` |
| 500 (same setup, after bug fixes) | 25/s | 5 min | **36,149** | **120.5** | **0.17 %** | **3** | 27 % more throughput, 99.8 % fewer 500s |

#### Resource saturation at 500u (after fixes)

| Component | Peak | Steady | Notes |
|---|---|---|---|
| Host load avg | 18.63 / 12 | ~15 | 155 % CPU saturation — bottleneck |
| `web` container CPU | 725 % | ~650 % | 7.25 / 8 gunicorn workers maxed |
| `db` container CPU | 92 % | ~15 % | Real query load (vs 16 % wasted before fixes) |
| `pgbouncer` CPU | 77 % | ~55 % | Single-threaded, approaching limit |
| `redis` CPU | 6 % | ~3 % | Idle |
| **DB connections** | **27 / 300** | 26 | PgBouncer multiplexes 500 clients onto ~26 server conns |
| Redis clients | 629 | ~550 | Sessions + cache |

The bottleneck is **CPU on the Python/gunicorn process** at 500
concurrent users. Postgres, PgBouncer, and Redis all have substantial
headroom. To scale past this point a single-VPS deployment would need:
- More gunicorn workers (12+ to use all cores), or
- Horizontal scale (multiple web containers behind nginx upstream),
- Aggressive HTTP-level caching for read-heavy endpoints.

## Bugs found by load testing

1. **`/api/v1/schema/` race condition**
   drf-spectacular's schema generator threw
   `AssertionError: Schema generation REQUIRES a view instance` when
   many concurrent first-time requests raced through generation. Visible
   at 500 concurrent users, invisible below that.
   **Fix:** wrapped the view with `cache_page(60 * 5)` so generation
   happens once per 5-minute window, then is served from Redis.

2. **`/api/v1/outputs/` unbounded list**
   The viewset had no `pagination_class`, so it serialized every record
   the user owned. After a few minutes of stress traffic the locust user
   had 5,700 `ProcessedPDF` rows; each list request became a multi-second
   serialization that nginx then cut off at 60 s.
   **Fix:** set `DEFAULT_PAGINATION_CLASS = PageNumberPagination` and
   `PAGE_SIZE = 50` globally.

3. **Thumbnails not garbage-collected**
   Each upload generated a thumbnail in `media/thumbs/` that survived
   deletion of the source `UploadedPDF` record. The load test grew this
   directory to **922 MB** in a few hours. Not fixed in this round; needs
   a `post_delete` signal or a celery beat sweep.

4. **`POST /upload/` 504 timeouts under heavy load**
   At 500 concurrent users with biased-large fixtures, 1 % of uploads
   exceeded nginx's 60-second timeout because gunicorn workers were
   busy handling other requests in the queue. Symptom of CPU saturation,
   not an app bug. Mitigated by raising gunicorn worker count.

## Files

- `locustfile.py` — minimal baseline: upload + compress, 3 user profiles.
- `realistic.py` — 4 user profiles, 3 PDF sizes, 7 mixed ops with
  human-like think times.
- `stress.py` — same op mix as realistic but zero-think-time and biased
  toward medium/large PDFs. Use to find the breaking point.

Run artifacts (`*_stats.csv`, `*.html`) are gitignored — regenerate them
with `--csv=<name>` and `--html=<file>` on each run.
