# Changelog

All notable changes to PDF Editor are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); this project ships
continuously to <https://pdf.micutu.com> rather than tagging releases, so
entries are grouped by the period the work landed.

## [Unreleased] — production-finish sweep (2026-07)

### Security
- **Pillow 12.2.0 → 12.3.0** — patches five CVEs (PYSEC-2026-2253..2257)
  disclosed 2026-07-13; the CI `pip-audit` gate caught it. Lockfile regenerated
  with only Pillow changed.
- **gVisor sandbox for the worker** — the Celery worker (where untrusted PDFs
  meet ghostscript / PyMuPDF / tesseract / pdf2docx, a recurring C/C++ RCE
  surface) now runs under the `runsc` runtime, so a parser exploit is contained
  by gVisor's user-space kernel instead of reaching host syscalls. On top of
  the existing `cap_drop: ALL` + read-only rootfs + non-root + seccomp. Verified
  compatible with the full native stack, including onnxruntime/fastembed.
- **Hash-pinned dependency lockfile** — `requirements.lock` pins every dep +
  transitive to an exact version and SHA-256; the image installs with
  `pip install --require-hashes`, so builds are reproducible and a tampered
  or typosquatted PyPI wheel can't slip in. (pip-audit on the lock: clean.)
- **SSRF guard port allowlist** — certificate-revocation fetches (pyHanko
  signing path) are now restricted to ports 80/443 on top of the existing
  public-IP check, shrinking the internal-service target set even under a
  DNS-rebind.
- **Full security audit** (2026-07) across ~20 categories — injection,
  SSRF, path traversal, IDOR, XSS, deserialization, upload safety, auth
  rate-limiting, crypto, secrets handling: no new exploitable findings; the
  one residual (a low-severity DNS-rebind window on the authenticated
  signing path) is documented and partially mitigated by the port allowlist.

### Added
- **Webhooks** — logged-in users can register HTTPS endpoints that receive an
  HMAC-SHA256-signed POST when one of their async jobs finishes, instead of
  polling. Manageable from the web UI (`/accounts/profile/webhooks/`) **and the
  REST API** (`/api/v1/webhooks/`) — with a synchronous `…/test/` ping to verify
  an endpoint — plus **SDK** methods (`create_webhook`, `test_webhook`, …) and a
  `verify_signature` helper. Each endpoint keeps a **delivery history** (last
  25 terminal outcomes, at `…/deliveries/` and in the UI) for debugging.
  Delivery is SSRF-guarded (public https only, re-validated at send time),
  doesn't follow redirects, retries with exponential backoff, and auto-disables
  an endpoint after 15 consecutive failures.
- **`/readyz` now checks Redis too** — readiness round-trips through the
  cache (the same Redis that backs the Celery broker and sessions) and
  returns 503 if either Postgres or Redis is unreachable, so the deploy
  smoke test and uptime monitors see a truthful signal.
- **Legal pages** — `/privacy/` and `/terms/`, served as full per-language
  documents (Romanian/English, picked by the active language with English
  fallback). Linked from a new site-wide footer; the register form now
  states account-creation consent.
- **Modern CSP violation reporting** — the policy now advertises the
  Reporting API (`report-to` + `Reporting-Endpoints` header) alongside the
  legacy `report-uri`, so current Chrome actually delivers violation
  reports again.
- **Scheduled media backups** — `scripts/backup_media.sh` is now on a daily
  cron; every backup/cleanup cron alerts to Telegram on failure via
  `scripts/notify_fail.sh`.
- **Post-deploy smoke test** — `scripts/deploy.sh` polls the public
  `/readyz` after bring-up and fails the deploy (with logs + rollback hint)
  if the stack doesn't come back healthy.
- **Low-disk alert** — a `node-exporter` service now feeds host disk/CPU/
  memory into Prometheus; a Grafana rule emails when any host filesystem
  drops below 15% free. Closes the disk-full gap the runbook flagged.
- **Disaster-recovery drill verified** — the newest DB dump restores into
  an ephemeral Postgres in ~5s with all sanity checks green.

### Changed
- **Server-side cursors disabled** behind the transaction-pooling pgbouncer
  (`DISABLE_SERVER_SIDE_CURSORS=True`) — the Django-documented requirement
  for transaction poolers, preventing a class of prod-only cursor bugs.
- **Container resource + log limits** — every service now caps its json log
  at 10 MB × 3 (was unbounded — a disk-fill risk on the box we now alert on);
  the real memory consumers (web, worker, db, redis, clamav, prometheus,
  grafana, loki) get generous `mem_limit` ceilings so a runaway can't OOM
  neighbouring projects on the shared VPS. The CPU-heavy services (worker 6,
  web 3, db 4, clamav 2 of the 12 cores) also get `cpus` caps so a parallel
  OCR/ghostscript batch can't peg every core.
- **Retention now enforced** — `cleanup_old_pdfs` runs hourly from cron,
  matching the "files auto-delete after 24h" promise in the UI. (A backlog
  of ~20k files / 2.4 GB that had accumulated since the pre-Docker timer
  disappeared was purged.)
- **Self-hosted Inter font** — the Google Fonts stylesheet had been
  silently blocked by the strict CSP since it shipped; Inter now ships as a
  self-hosted variable `woff2`. No third-party browser requests remain.
- **Tighter CSP** — dropped the `https://*.cloudflare.com` script wildcard
  (it also allowed `cdnjs`, a public-CDN bypass); only exact hosts remain.
- **Romanian translation catalog completed** — 888/888 strings translated,
  zero fuzzy (was 846 translated / 282 untranslated). Catalogs regenerated
  without the framework-string pollution the old ones carried.
- **Docs synced with reality** — README test counts, feature list (MFA,
  passkeys, sessions, ClamAV, SBOM/cosign), and service inventory corrected;
  RUNBOOK gained scheduled-jobs and external-uptime sections.

## Security hardening (2026-06)

### Added
- **TOTP two-factor authentication** with single-use backup codes and
  replay protection (per-device last-verified timestep).
- **WebAuthn passkeys** — discoverable credentials, user verification
  required, passwordless login.
- **Session management** — per-device active-session list with instant
  revoke, "sign out everywhere else", and new-device login alert emails.
- **Cloudflare R2 output mirror** — processed files mirrored to object
  storage with presigned-URL downloads; purged in step with retention.
- **CSP violation endpoint** (`/csp-report/`) with a Prometheus counter, so
  a policy regression is an alert rather than silent breakage.
- **Supply-chain provenance** — SPDX SBOM published per build and cosign
  keyless image signatures, on top of the existing Trivy gate.
- **ClamAV upload scanning** (opt-in) for uploaded PDFs and images.
- **GDPR** self-service data export and account deletion.

### Changed
- Audit-log, rate-limit, and metrics IP handling unified on a
  correctly-counted `X-Forwarded-For` parser (spoof-resistant).
- pdf.js vendored locally with `isEvalSupported:false`; `unsafe-eval` and
  the unpkg origin removed from the CSP.
- Strict nonce-based CSP set per-request by middleware.

## REST API, AI, and scale (2026-06)

### Added
- **REST API** (DRF + OpenAPI 3) with per-user SHA-256-hashed API keys and
  a scope-aware 9-cell rate-limit matrix; Swagger + Redoc docs.
- **Python SDK + `pdf-edit` CLI** (`sdk/`).
- **Async job pipeline** — Celery worker for long ops (OCR, PDF/A, compare,
  convert, RAG-index, to-images) with live Server-Sent-Events progress and
  a polling fallback; jobs are cancellable mid-flight.
- **Chat with PDF (RAG)** — pgvector retrieval, local ONNX embeddings,
  Groq LLM with citations; plus a single-shot `/ops/summarize/`.
- **Batch operations** — one op across up to 50 PDFs in a single job.
- **Observability stack** — Prometheus (per-replica service discovery),
  Grafana (provisioned dashboard + alert rules), Loki + Promtail logs.
- **Horizontal scaling** — N `web` replicas behind an internal nginx load
  balancer; PgBouncer transaction pooling.

## PDF toolbox (2026-05)

### Added
- **25+ PDF operations** reachable from both the web UI and the API:
  split, merge, compress, convert (PDF↔DOCX, PDF↔images, images→PDF),
  metadata editor, crop, flatten, rotate, page numbers, watermark,
  reorder/delete pages, bookmark/outline editor, find & replace, AI
  rephrase, true text redaction, OCR text extraction + searchable-PDF
  layer, AcroForm detect/fill, compare two PDFs, PDF/A-1b/2b.
- **Digital signatures** — PKCS#7 / PAdES B-B, B-T, B-LT, B-LTA with TSA
  timestamps and LTV; a signature-verification endpoint; self-signed
  certificate generator; server-side trust anchors.
- **Password protect / remove** (AES-256 via PyMuPDF).
- **Share links** — public token downloads with TTL and download caps.
- **Accounts** — registration with email confirmation, password reset,
  persistent per-user history and storage quotas.
- **PWA** — installable, offline app shell, mobile-responsive UI with dark
  mode, browser notification when a backgrounded job finishes.
- **i18n** — full Romanian + English catalogs, including a JavaScript
  message catalog for static JS.
- **Custom error pages** (400/403/404/500) and an `/admin/health/`
  platform-stats dashboard.
