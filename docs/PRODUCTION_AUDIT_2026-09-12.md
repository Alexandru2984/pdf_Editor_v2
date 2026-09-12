# Production audit and implementation plan — 2026-09-12

Baseline: branch `main`, clean worktree, commit
`24f3d6d562fe554d59680af2fa368e0061a94771`. This is a continuing audit, not a
certificate of security or a claim that all existing features have been tested.
Evidence below distinguishes live observations, source findings and work still
requiring verification. No customer records, credentials or origin addresses
belong in this document.

## Production baseline

- Public HTTPS and the host-local `/healthz` probe returned 200. Public responses
  have enforced nonce CSP, HSTS, nosniff and frame protection.
- The loaded host vhost is `/etc/nginx/sites-enabled/pdf_editor`, targeting
  loopback port 8000. It verifies the original Cloudflare peer and replaces XFF.
  Docker Nginx appends the second proxy entry and resolves the two web replicas.
- Cloudflared runs on this host, but its local ingress configuration contains
  no PDF-domain entry. Do not infer that unrelated tunnels serve this app.
- Current path supported by configuration and probes:
  Cloudflare → host Nginx → Docker Nginx → two Gunicorn/Django replicas;
  application → PgBouncer → PostgreSQL/pgvector, Redis, ClamAV, Celery and R2.
  Provider-side routing/control-plane configuration has not been inspected.
- Both web replicas and the worker use `runsc-net`, non-root `app`, read-only
  roots and memory/CPU limits. Observed restart counts were zero. Web limits are
  1 GiB/3 CPUs each; worker 4 GiB/6 CPUs. No web/worker healthcheck is configured.
- Runtime has DEBUG off, secure SSL on, proxy depth 2, rate limiting on,
  ClamAV on/fail-closed, and R2 on. Live package versions include Django 5.2.16,
  requests 2.34.2, urllib3 2.7.0, PyMuPDF 1.28.0 and Pillow 12.3.0;
  Ghostscript 10.05.1 and Tesseract 5.5.0. These observations alone do not prove
  freedom from known vulnerabilities.
- Application image has no source revision label. Host Nginx differs from the
  repository's static-cache configuration. Deployment parity needs verification.
- Available host disk was approximately 193 GiB; no uncontrolled load test is
  justified. PostgreSQL is loopback-bound; cache/broker is not host-published.
- The existing host virtualenv is broken after a system Python change (launcher
  resolves Python 3.14 while installed packages are under Python 3.13).
  Verification will use isolated containers with the deployed application's
  dependency image, a new disposable database, temporary media and no production
  credentials, broker, cache or object-store access.

## Historical audit: what needs revalidation

Read `SECURITY_AUDIT_2026-08-09.md` and `PRODUCTION_ROADMAP.md` as historical
evidence, not as verified current state. gVisor, fail-closed scanning, proxy
normalization and CSP are observed live. Owner filters and encrypted-ingest
quarantine exist in source. The previous SSRF description understates the
redirect path: pyHanko's installed requests fetchers follow redirects after the
wrapper checks only the initial URL. Tests only classify initial destinations
and construct fetchers; they do not exercise this transport boundary.

Changes since that audit include localized public routes, marketing/tool pages,
SEO indexing policy, JSON-LD serialization, static asset delivery, Search Console
support, dependency floors and CI pin updates. These require route, cache and
security-policy regression checks in addition to the earlier feature coverage.

Prior claims of compromised credentials, backup encryption requirements or a
required reboot are **not reasserted as current incidents without new evidence**.
Provider revocation, whole-VPS backup changes and reboots require separately
verified scope and operational planning.

## Prioritized implementation roadmap

Each row records problem/evidence, failure scenario, treatment, regression risk,
verification and approximate scope/dependencies. Priority can change when a
reproducer establishes impact. Tests use synthetic objects and offline peers.

| ID / priority | Problem, evidence and scenario | Fix and affected components | Risk, tests, scope and dependencies |
| --- | --- | --- | --- |
| A01 / P0 candidate → P1 confirmed source flaw | `email_utils.py` uses Django's password-reset generator for confirmation; `confirm_email_view` logs in directly, including active users. A password-reset token may be redeemable at the confirmation route to bypass MFA or reactivate a disabled account. Existing comments/tests incorrectly claim the default generator binds `is_active`. | Separate activation token purpose and account state; refuse active-user activation; retain MFA at activation; serialize consumption. | Medium: pending activation links need resend; password reset must continue unchanged. Reproduce cross-purpose use, replay, inactive/active accounts and concurrent consumption. Small, first code stage. |
| A02 / P1 | `ssrf_guard.py` resolves then delegates to pyHanko requests GET/POST, which re-resolve and follow redirects. `_ip_is_public` also admits shared CGNAT space. Webhook validation accepts arbitrary HTTPS ports and malformed ports can raise outside the intended error handler. | Shared bounded outbound transport connecting to a validated IP with original TLS hostname; reject redirects, userinfo, unsupported ports and non-global IPs; integrate certificate fetches and webhooks. | Medium: unusual CA redirects/custom webhook ports become errors. Offline tests at socket/pool boundary for rebinding, redirect, TLS identity, IPv4/IPv6, response size, timeout and cleanup. Medium; no infrastructure change needed. |
| A03 / P1 | `webhooks.deliver_once` eagerly buffers an unrestricted response body despite only using status. A destination controlled by an account can consume web/worker RAM. | Stream headers-only delivery responses and always close them; bound certificate response bodies/decompression. | Low–medium; response bodies are not part of delivery contract. Oversized/chunked/compressed response and resource-close tests. Small; A02. |
| A04 / P1 | MFA backup regeneration and passkey/API-key creation lack recent-auth gates; password alone disables MFA. Stolen sessions can establish durable credentials. | Consistent recent password/passkey plus MFA challenge for sensitive actions, CSRF-safe continuations, bounded credential creation and security notifications where configured. | Medium–high UX/session impact. Route-by-route session/CSRF/replay tests; preserve passkey-only recovery. Medium; A01, design existing login/session flows first. |
| A05 / P1 | Upload web/API paths duplicate validation; `storage_usage` sums only originals without an atomic reservation. Derived files and parallel operations exceed quotas. | Shared ingest service, reservation/accounting for source/derived/temp storage, atomic owner quotas, bounded jobs and retention cleanup. | High: schema and filesystem lifecycle. Multi-connection concurrency tests, failure cleanup and backward-compatible migration review. Large; inspect table sizes/backups before any live migration. |
| A06 / P1 | Synchronous PDF/image/OCR paths coexist with Celery; container limits bound aggregate process damage but do not prove operation-level pixel/output/time bounds. | Inventory every parser and subprocess; apply page/pixel/output budgets before allocation, timeouts and safe temporary output cleanup. | Medium: oversized documents will receive actionable errors. Synthetic large-dimension PDF/image tests without huge allocation; web/API/worker parity. Medium; immediate non-schema bounds can precede A05. |
| A07 / P1 | `tasks.py` has start/finish/cancel/retry paths and late acknowledgements; state races and owner checks for secondary/batch sources need targeted review. | Atomic job state transitions, cancellation-safe output publication, current-owner/source validation, bounded queue admission and idempotency. | Medium–high: job lifecycle changes. Concurrent cancel/finish, duplicate delivery, deleted source and cross-owner batch tests. Medium; A05 for full accounting. |
| A08 / P1 | Owner-scoped web helpers and API querysets exist; media/download/job/chat/share paths have distinct implementations. Share owner fallback uses an empty session key. | Enumerate every object endpoint and prove foreign-user/session denial, including missing-session rows, encrypted quarantine, secondary inputs and R2 redirects. Fix any bypass at shared boundaries. | Low for focused authorization fixes. Synthetic two users/two anonymous sessions; GET/HEAD/POST/DELETE and identifier variants. Medium; independent. |
| A09 / P1 | Webhook secrets/TOTP seeds and share bearer tokens are plaintext DB columns. Public shares count GET and HEAD consumption and lack explicit private response policy. | Immediate method/cache/privacy safeguards; then hash bearer tokens and use a separate versioned keyring for reversible secrets, one-time disclosure and safe rotation. | Low for headers/methods; high for key deployment/migrations and old links. HEAD/replay/cap/concurrency, DB leak and dual-read migration tests. Large; external key custody and A04. |
| A10 / P1–P2 | Redis uses one instance with `allkeys-lru` for broker, results and cache; web/worker lack healthchecks, and source/image revision parity is opaque. | Verify queue-loss pressure, split durable broker policy, heartbeat/queue-age alerts, immutable revision labels and safe deployment health gates. | Medium/high for runtime changes. Isolated memory pressure and worker failure tests; canary/rollback before production. Medium; deployment validation. |
| A11 / P2 | Host config drifts from repository; proxy depth is trusted by count. Direct-origin denial and all alternate vhosts/routes need current proof. | Trace matching vhosts, compare trusted ranges, validate XFF/proto/Host across actual chain, static and SSE behavior, document exact drift. | High if changing shared Nginx/firewall; inspect and test first, back up exact vhost and `nginx -t` before reload. Small/medium; no unrelated infrastructure changes. |
| A12 / P2 | Mutable external images, old monitoring versions, Promtail and runtime model retrieval remain in Compose/processor source. | Current advisory/lock audit; immutable artifact provenance, maintained log collector and pinned offline embedding model. | Medium/high image/model compatibility. Scanner evidence, staging upgrade, embedding compatibility and log continuity tests. Large; vendor version verification and available deployment path. |
| A13 / P2 | `serve_media_view` enumerates all owned paths per download; export/listing/derived-file lifecycle need query and resource review. | Targeted ORM lookup, pagination, select-related and bounded exports; consolidate duplicated operations only after behavioral tests. | Low/medium. Query-count, missing-file, symlink/path traversal, close/cleanup tests. Medium; A08 and A05. |
| A14 / P2 | Backup/restore docs explicitly record PITR/media drill gaps; historical audit includes whole-VPS risks. Current backup freshness, encryption and recovery evidence remain unverified. | Inspect only application-relevant schedules/metadata; isolated database+media restore; record measured recovery and exact gaps. | High if scope reaches shared backup targets; no speculative credential rotation or destructive restore. Medium/large; backup target and encryption evidence. |
| A15 / P3 | Existing mobile/keyboard/CSP/SEO improvements are substantial; new localized routes and client code require behavior checks. | Browser audit of auth/upload/results/jobs/shares at mobile/desktop; correct focus/status/error feedback, metadata and cache behavior. Product changes follow proven broken flows. | Low/medium. Keyboard navigation, reduced motion, contrast, HTML validity, cross-language POST and browser smoke tests. Medium; stable backend fixes. |
| A16 / P4 | Existing editing/OCR/RAG/SDK/webhooks already cover many features; additional teams, billing, visual editor and collaboration are major product commitments. | Prioritize useful extensions from actual missing workflows; keep large platform features as separately scoped projects after security gates. | High scope/operational cost. Product acceptance criteria and architecture review required; no speculative implementation that obscures P0/P1. Large; prior stages. |

## Execution and release rules

1. Commit this reviewable baseline/plan, then establish isolated test tooling.
2. Prove and fix A01; commit after regression checks.
3. Prove and fix A02/A03 with a shared transport and offline adversarial tests;
   commit independently of account changes.
4. Continue A08/A06/A07 and sensitive account/share controls in coherent stages.
   Record coverage honestly; a green suite is not proof of an exhaustive audit.
5. Before every commit inspect status and intended diff, check for secrets, run
   relevant checks, stage exact paths and use the existing Git identity without
   attribution trailers. Do not push automatically.
6. Repository fixes do not change running application images. Release through
   the signed build/deploy workflow once available; review rollback, data impact
   and health evidence before any production mutation. No service restart,
   migration, credential rotation or customer-data modification is part of
   baseline discovery.

## Verification and implementation log

### A01 — activation/reset separation

- Six new isolated regressions failed before the fix, proving cross-purpose
  reset/activation tokens, activation bypass of MFA, invalid single-use
  assumptions and resend reactivation of previously used disabled accounts.
- Dedicated activation salt/state binding, inactive-and-never-logged-in gate,
  row locking during consumption and MFA-aware activation now close those paths.
- Verification: all 71 authentication tests passed, followed by a real
  two-connection PostgreSQL activation race test (one login, one rejection).
  Ruff lint/format and diff checks are required before commit.
- Release impact: old pending activation links are intentionally rejected;
  users must use resend. Password-reset token behavior is unchanged. No schema
  migration or production data update is needed. A dedicated account-verification
  timestamp remains preferable to `last_login` as a future account-state model.
- Changes are local until the signed deployment workflow publishes a new image.

## Reference material

- Python documents that shared CGNAT addresses have both `is_private=False` and
  `is_global=False`: <https://docs.python.org/3/library/ipaddress.html>.
- urllib3 documents explicit SNI/certificate hostname controls when connecting
  to a fixed address: <https://urllib3.readthedocs.io/en/stable/advanced-usage.html>.
- Installed pyHanko requests fetcher source was inspected directly; claims
  about redirect behavior are based on that implementation and requests call
  defaults, not solely on the historical audit.
