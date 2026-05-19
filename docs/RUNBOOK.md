# Runbook — PDF Editor

What to do when a page wakes you up. Each section is one incident type:
how to confirm it's the right diagnosis, where to look, what to do
*right now* to stop the bleeding, and what to follow up on after.

Conventions:

- Commands assume you're SSH'd to the VPS in the project root
  (`cd /home/micu/pdf_Editor_v2`).
- "App" means the Django+Celery+Postgres+Redis stack in `docker-compose.yml`.
- Grafana: <https://pdf.micutu.com:3003> (replace with your bind).
- Sentry catches application exceptions; Grafana catches infra signals.

When in doubt, the first move is always the same:

```bash
docker compose ps                # what's up / not up
docker compose logs --tail=200   # recent log noise
```

---

## 1. 5xx burst — site is throwing errors

**Page source:** `pdfeditor_5xx_spike` (≥ 1 5xx/s for 5 min) or
`slo_api_availability_fast_burn` (≥ 1.44% 5xx rate over 1h).

### Confirm

1. Open Grafana → **PDF Editor** dashboard → "5xx rate" panel. Is the
   rate sustained, or did it already drop?
2. Check Sentry — most 5xx have an exception attached. If the top error
   has a clear traceback, jump straight to "Mitigate".
3. If Sentry is empty but 5xx is high: errors are happening *outside*
   Django (nginx, gunicorn boot loop, upstream DB unreachable). Check
   `docker compose logs --tail=200 nginx web`.

### Diagnose

```bash
# Are all replicas healthy?
docker compose ps web worker

# Per-replica 5xx breakdown — finds a single bad replica
docker compose exec prometheus wget -qO- \
  'http://localhost:9090/api/v1/query?query=sum%20by%20(replica)%20(rate(django_http_responses_total_by_status_view_method_total%7Bstatus%3D~%225..%22%7D%5B5m%5D))'

# Recent app logs filtered to errors
docker compose logs --since=10m web | grep -iE "error|exception|traceback"
```

### Mitigate

- **One bad replica:** restart it. `docker compose restart web` cycles
  all of them; safer is to find the bad replica by IP from the per-
  replica panel and `docker restart <container-id>`.
- **All replicas bad, recent deploy:** roll back.
  ```bash
  # deploy.sh tags the prior image as pdfeditor:rollback before pulling
  # the new one. Promote it back to :latest and restart the web stack.
  docker tag pdfeditor:rollback pdfeditor:latest
  docker compose up -d --no-build web worker
  ```
- **All replicas bad, no recent deploy:** likely a dependency went down
  (DB, Redis, Groq API). Skip ahead to section 2 or 3 depending on which
  one Prometheus says is unreachable (`up{job=...} == 0`).

### Follow-up

- Capture the Sentry issue link in the incident note.
- If a deploy was rolled back: open a PR with the reproduced bug + a
  test that catches it. Don't re-land the broken commit without one.

---

## 2. DB connection pool exhausted

**Page source:** Typically *no direct alert*; symptom is high 5xx + many
`OperationalError: too many clients already` exceptions in Sentry.

### Confirm

```bash
# pgbouncer is the choke point. Check its admin console for queue depth.
docker compose exec pgbouncer psql -h 127.0.0.1 -p 6432 -U $POSTGRES_USER pgbouncer -c "SHOW POOLS;"
# Look at sv_active vs sv_idle and cl_waiting. cl_waiting > 0 sustained
# means clients are stalled waiting for a server connection.
```

```bash
# Backend-side: how many sessions does Postgres actually have?
docker compose exec db psql -U $POSTGRES_USER -d $POSTGRES_DB -c \
  "SELECT state, count(*) FROM pg_stat_activity WHERE datname=current_database() GROUP BY state;"
# 'idle in transaction' rows are the killer — those hold a server slot
# while doing nothing.
```

### Mitigate

- **Burst load:** scale workers down momentarily to free up the pool.
  ```bash
  docker compose up -d --scale web=1 --scale worker=1
  ```
  Then scale back up once the queue drains.
- **`idle in transaction` rows piling up:** find the holders and kill
  them. Look at `query_start` to find old ones.
  ```sql
  SELECT pid, state, query_start, query
    FROM pg_stat_activity
   WHERE state = 'idle in transaction'
   ORDER BY query_start ASC
   LIMIT 20;

  SELECT pg_terminate_backend(<pid>);
  ```
- **Sustained legitimate traffic:** bump pgbouncer's `DEFAULT_POOL_SIZE`
  in `docker-compose.yml` (current: 25). Each +5 costs 5 Postgres
  backend slots; the DB is configured for `max_connections=100`. Don't
  go above ~80 without raising `max_connections` on the DB too.

### Follow-up

- Look for the code path that opened a transaction and didn't commit
  or roll back — usually a missing `@transaction.atomic` boundary or an
  exception escaping a `with transaction.atomic()` block.
- Consider an upper bound: a slow query or a Celery task can hold a
  connection for minutes. `statement_timeout` and `idle_in_transaction_
  session_timeout` are your friends.

---

## 3. Redis OOM / Celery broker unreachable

**Page source:** `pdfeditor_queue_depth` (≥ 50 jobs queued for 10 min)
or `slo_job_success_fast_burn`. Symptom: jobs sit queued, workers idle.

### Confirm

```bash
# Is Redis up?
docker compose ps redis
docker compose exec redis redis-cli PING                  # → PONG

# Memory pressure?
docker compose exec redis redis-cli INFO memory | grep -E "used_memory_human|maxmemory_human|maxmemory_policy"

# Celery broker queue depth — should be ≈ pdfeditor_job_queue_depth
docker compose exec redis redis-cli LLEN celery
```

```bash
# Are workers processing anything?
docker compose logs --tail=50 worker | grep -iE "received|succeeded|failed"
```

### Mitigate

- **Redis OOM (`used_memory` ≈ `maxmemory`):** the broker is full and
  refusing writes.
  - Drain non-critical keys first: `docker compose exec redis redis-cli
    --scan --pattern 'cache:*' | xargs -L 1000 docker compose exec -T redis redis-cli DEL` (cache
    keys are safe to drop; Celery queue keys are not).
  - If that doesn't help, bump `maxmemory` in `docker/redis.conf` and
    `docker compose up -d redis`. Restart drops in-flight broker state
    *but* tasks already in the DB Job table are picked up again by the
    worker on reconnect.
- **Worker pool deadlocked:** workers connected but not consuming.
  ```bash
  docker compose restart worker
  ```
  Tasks are idempotent (each one re-checks job state in the DB before
  doing work), so restart is safe.
- **Broker reachable but workers not running:** check `worker` service
  status and logs for an import error or migration mismatch.

### Follow-up

- If cache keys ate the broker memory: separate Redis instances for
  cache vs broker. They have different durability needs.
- Add a `redis_memory_used_bytes / redis_memory_max_bytes` alert at
  80% so we get warning *before* OOM, not after.

---

## 4. Celery workers stuck — jobs queued but not progressing

**Page source:** `pdfeditor_queue_depth` warning, but Redis is healthy.

### Confirm

```bash
# Workers are present?
docker compose exec worker celery -A pdf_project inspect active
docker compose exec worker celery -A pdf_project inspect stats

# A task that's been running > task_time_limit is the usual culprit —
# the worker is wedged in a C extension (PyMuPDF, tesseract) and Celery
# can't interrupt it.
docker compose exec worker celery -A pdf_project inspect active_queues
```

### Mitigate

- **Wedged worker:** restart it. `--max-tasks-per-child=50` is set so a
  worker recycles itself, but a single stuck task can outlast it.
  ```bash
  docker compose restart worker
  ```
  In-flight jobs are marked `failed` by the post-recovery code path
  in `tasks.py`; users will see a clear error and can retry.
- **Soft time limit firing repeatedly on one PDF:** an attacker (or a
  user with a pathological PDF) is wedging workers. Find the job and
  cancel it.
  ```bash
  docker compose exec web python manage.py shell -c "
  from pdfeditor.models import Job
  Job.objects.filter(status='running').values('id','kind','created_at')
  "
  # Then either kill the celery task or mark the job failed in the DB.
  ```

### Follow-up

- If the same op kind keeps wedging: add an explicit subprocess timeout
  inside the op, don't rely on Celery's soft limit alone.
- Capture the offending PDF as a regression test if you can extract it
  without exposing user data.

---

## 5. Disk full on media volume

**Page source:** Usually *no direct alert* (TODO: add one); symptom is
upload 5xx with "No space left on device" in Sentry, or jobs failing
right at the output-write step.

### Confirm

```bash
df -h /var/lib/docker
docker system df -v        # which volumes are biggest
du -sh ./media/* 2>/dev/null
```

### Mitigate

- **Quick win:** sweep orphaned files (created by the dedicated command).
  ```bash
  docker compose exec web python manage.py sweep_orphan_files --apply
  ```
- **Old processed PDFs piling up:** `cleanup_old_pdfs` is the retention
  sweep — runs on a schedule, but you can force one.
  ```bash
  docker compose exec web python manage.py cleanup_old_pdfs --apply
  ```
- **Still no room:** Docker logs are usually the next-biggest culprit.
  Find them with `du -sh /var/lib/docker/containers/*/*-json.log` and
  truncate the worst offenders (`: > <file>`). Log rotation should
  prevent this — check `daemon.json`.

### Follow-up

- If orphans appeared: there's a code path that creates files outside
  the DB. The `post_delete` signal in `pdfeditor/signals.py` should
  catch all deletes; the gap is usually an op that writes to disk
  before the DB row commits.
- Add a `node_filesystem_avail_bytes` alert at 15% — disk-full is one
  of the loudest, most-recoverable, most-preventable incidents.

---

## Drills

The runbook is only useful if you've run it once before being paged.
Twice a quarter, pick one section and execute it against a staging
copy of the stack, cold. Time yourself. If a step doesn't work as
written, fix the runbook *before* fixing anything else.
