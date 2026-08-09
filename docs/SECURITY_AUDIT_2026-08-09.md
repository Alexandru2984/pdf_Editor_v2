# Security and production audit — 2026-08-09

Status: remediation in progress. This document is intentionally safe to keep in
the repository: it contains no credentials, private tokens, customer records,
origin addresses, or exploit payloads.

## Executive decision

**Verdict: MITIGATE URGENTLY.** The service can remain online with the live
containment controls described below, and the security-remediation build should
ship through the signed deployment pipeline. General-availability claims and
new feature releases remain gated until the P0 controls are closed.

The strongest parts of the current system are owner-scoped file access, a broad
automated test suite, hash-locked Python dependencies, malware scanning,
Cloudflare/origin separation, non-root read-only containers, gVisor isolation,
rate limiting, MFA/passkeys, CSP, signed image deployment, observability, and
documented SLOs/runbooks.

The largest remaining risks are operational rather than a single trivial web
bug:

1. Production credentials need a coordinated rotation and revocation cycle.
2. The whole-VPS off-site backup path must use client-side encryption and be
   proven through a restore exercise.
3. Derived files, temporary work, embeddings, object-store mirrors, and repeated
   CPU-heavy operations need one atomic quota/accounting model.
4. Reversible application secrets stored in PostgreSQL need envelope encryption;
   bearer tokens should be hashed where reversibility is unnecessary.
5. Third-party container images and the ML model artifact need immutable,
   verified identities; Promtail must be replaced because it is end of life.

## Scope and methodology

The audit covered:

- Django settings, middleware, templates, sessions, authentication, MFA,
  passkeys, password/email flows, API keys, share links, webhooks and audit logs;
- upload validation, ClamAV, PDF parsing/sanitization, PDF operations, async
  tasks, downloads, media ownership and object storage;
- REST API authentication, throttling, schema/docs, batch jobs and webhooks;
- dependency locks, CI, image build, SBOM/signing/deploy and rollback logic;
- Docker Compose, Nginx, Cloudflare origin trust, PostgreSQL/PgBouncer, Redis,
  gVisor, host firewall posture, backups and monitoring;
- responsive CSS, mobile navigation, keyboard access, CSP compatibility and
  reduced-motion behavior;
- production smoke checks that did not mutate customer documents.

Techniques included manual data-flow review, abuse-case review, STRIDE threat
modeling, source and history secret scanning, SAST, dependency audit, image and
Dockerfile scanning, migration checks, deployment checks, targeted regressions,
and the complete non-E2E test suite.

Not included: a destructive production load test, a third-party black-box
penetration test, review inside vendor control planes, legal advice, or a
physical/datacenter assessment.

## Architecture and trust boundaries

```text
Internet
  -> Cloudflare proxy/WAF/TLS
  -> host Nginx (verified Cloudflare/loopback source only; normalized client IP)
  -> Docker Nginx
  -> two Django/Gunicorn web replicas under gVisor
       -> PgBouncer -> PostgreSQL/pgvector
       -> authenticated Redis (sessions/cache/Celery)
       -> ClamAV
       -> shared media and embedding cache
       -> Celery worker under gVisor
       -> optional external processors/providers
            (object storage, mail, error reporting, AI, breach-password check,
             certificate/revocation endpoints and model registry)
```

Important residual boundary: `runsc-net` uses gVisor for process/filesystem
containment but uses host networking so Docker DNS and internal services work.
It materially reduces parser escape blast radius, but it is not an egress policy
and does not protect secrets or files already readable inside the sandbox.

The long-term architecture should split untrusted ingest from normal application
workers: a networkless parser receives a file/job over a narrow handoff,
produces a sanitized artifact, and never receives production credentials.

## Remediation completed during this audit

Application commits are local until they pass the normal signed CI/deploy
workflow. Host controls explicitly marked **live** were applied and smoke-tested
on the VPS.

| Commit | Control | State |
| --- | --- | --- |
| `e498191` | Restrict backup permissions, create atomically, checksum and verify before restore | Repository; live backup directory/file modes hardened |
| `b21f07a` | Stop anonymous GETs creating sessions; bound and clean session retention | Repository; expired live sessions safely reclaimed |
| `b1e170f` | Reject ambiguous/malformed ClamAV results by default | Repository |
| `8561959` | Upgrade cryptographic dependencies; exact lock and narrow time-bounded waiver | Repository |
| `b319580` | Enforce Cloudflare-to-origin trust boundary and normalize forwarded IP | **Live**, direct-origin requests verified as denied |
| `c3c5ffd` | Deploy exact tested digest, verify Cosign identity, detect dirty trees and auto-rollback | Repository |
| `3f3f79a` | Require CSRF-protected POST for destructive PDF/history actions | Repository |
| `5e6bc7f` | Enforce CSP on app/admin/API docs and self-host executable docs assets | Repository |
| `d4b7b35` | Require current gVisor and sandbox web/worker parsing | **Live**, replicas and worker verified under `runsc-net` |
| `0d9b4a9` | Fail closed on PDF sanitization and quarantine encrypted PDFs to unprotect-only | Repository |
| `baae65d` | Pin every GitHub Action to an immutable full SHA | Repository |
| `994c210` | Redact backend exception details from public readiness responses | Repository |
| `be64c24` | Publish RFC 9116 vulnerability disclosure contact | Repository |
| `69bc42c` | Replace filename-derived source previews with opaque owner-scoped IDs | Repository |
| `3e2838b` | Add skip link, main landmark, keyboard mobile menu, focus ring, 44px targets and reduced motion | Repository |

No commit contains automated author, contributor, co-author, or tool attribution.

## Open risk register

| ID | Severity | Finding and evidence | Consequence | Required treatment |
| --- | --- | --- | --- | --- |
| R-01 | Critical | Production credentials require coordinated rotation; they are present only in the protected runtime environment, not Git | Compromise can span app signing/session material, database/cache and external providers | Revoke/rotate external credentials first, then DB/Redis and application signing material in a maintenance window; verify old credentials fail |
| R-02 | Critical | The whole-VPS script targets a plain remote although an encrypted remote exists | Off-site compromise can disclose databases, media, configuration and other VPS projects | Switch to client-side encrypted remote, use least-privilege credentials, add immutable retention and perform a clean-host restore drill |
| R-03 | High | Upload quota excludes many processed files, thumbnails, temporary files, embeddings and remote mirrors; sum-then-write is racy | Disk, memory, CPU and cost exhaustion; noisy-neighbor outage on the shared VPS | Reservation ledger + per-user/tenant hard and soft limits, operation cost units, concurrency caps, timeout/cancel and automatic retention |
| R-04 | High | TOTP seed and webhook signing secret are reversible business secrets stored as ordinary DB text; webhook APIs can re-display a secret | A DB-only compromise can enable MFA cloning or forged webhook messages | Envelope encryption with a versioned keyring outside PostgreSQL; reveal webhook secret once; rotation and audit events |
| R-05 | High | Share bearer tokens are stored directly and URL scanners can consume counters | DB read leaks valid public links; email/security scanners can exhaust limited links before users | Store token hashes, separate preview from consumption, make download count atomic and scanner tolerant, support password/domain restrictions |
| R-06 | High | API keys have no scopes, expiry, rotation overlap or optional source restrictions; last-used writes can amplify traffic | One key becomes full-account access and can create DB write load | Scoped keys, expiration, prefix/fingerprint, one-time reveal, rotation window, coarse/async last-used updates and per-key limits |
| R-07 | High | Application/webhook/certificate fetchers retain broad outbound network access; validation cannot fully prevent DNS rebinding | SSRF, metadata/internal-service access, data exfiltration and webhook abuse | Default-deny egress proxy/firewall, resolve-and-connect to validated IP, block redirects across trust zones and isolate webhook delivery |
| R-08 | High | External Compose images and the Python base use mutable tags; several stack versions are old | Rebuild drift, compromised tags, unreviewed upgrades and known-vulnerable components | Upgrade in staging, pin multi-arch manifest digests, verify provenance/signatures and let Dependabot/Renovate propose digest updates |
| R-09 | High | Promtail is EOL as of 2026-03-02 | No security/support updates for the log collector | Migrate with Grafana Alloy converter, validate positions/labels/alerts, run side-by-side, then remove Promtail |
| R-10 | High | Web/worker have no application-level container healthcheck; historical worker restarts were not directly alerted | A running but wedged worker can silently accumulate jobs | Add worker ping/heartbeat and queue-age checks, web readiness, restart alerts, graceful drain and dead-letter handling |
| R-11 | High | Backup scripts are improved but PITR/WAL, encrypted media recovery and clean-host restore evidence are incomplete; volume encryption is unverified | Ransomware/operator error can exceed recovery objectives | PostgreSQL PITR, versioned object/media backup, quarterly restore drill, measured RPO/RTO and provider encryption evidence |
| R-12 | High | Credential/security changes do not consistently require recent authentication | A stolen unlocked session can establish durable access | Re-authenticate with password/passkey/MFA for MFA disable, passkey/API-key/webhook changes, email/password and account deletion |
| R-13 | High | Runtime embedding model retrieval reaches an external registry; tests observed behavior-drift warning | Non-hermetic builds/tests, supply-chain drift, startup outage and embedding incompatibility | Vendor model by immutable revision + digest/attestation, build it into a separately scanned artifact, block runtime downloads and version the embedding schema |
| R-14 | High | Vendor governance is not encoded: AI, object storage, mail, analytics/error monitoring, backups, model registry and hosting can process personal data | GDPR transfer/retention/subprocessor and breach-response exposure | Data-flow inventory, DPA/SCC where applicable, regions, retention, deletion proof, subprocessor register, breach SLA and opt-in/disable controls |
| R-15 | Medium | Breached-password validation fails open when the external service is slow/unavailable and sends a k-anonymity prefix | Weak passwords may pass during outage; third-party privacy/availability dependency | Local denylist/cache, circuit-breaker metrics, explicit risk decision and no raw password telemetry |
| R-16 | Medium | Audit records are useful but not tamper-evident and share the primary database | An app/DB compromise can alter repudiation evidence | Append-only export, hash chaining or signed daily checkpoints, separate retention/access and tested investigation queries |
| R-17 | Medium | CSP still permits inline styles and first-party analytics loads without an explicit consent/control design | Style-injection blast radius and possible privacy non-compliance | Move page styles to versioned files, remove `style-src 'unsafe-inline'`, document cookieless behavior or implement consent/DNT controls |
| R-18 | Medium | Host kernel update requires a coordinated reboot; image/runtime updates also need a regular patch cadence | Known fixes remain inactive until reboot/recreate | Maintenance window, reboot verification, unattended security policy, image age alerts and rollback plan |
| R-19 | Low | An inactive legacy Nginx vhost remains next to the canonical one | Future operator can enable stale/insecure configuration | Remove/archive after confirming no external dependency; test `nginx -T` in deployment validation |

## STRIDE threat model

| Category | Credible attack | Existing controls | Residual action |
| --- | --- | --- | --- |
| Spoofing | Forged client IP, stolen API/share token, hijacked session | Cloudflare origin boundary, normalized proxy depth, secure sessions, MFA/passkeys, hashed API-key material | Hash share tokens, scoped/expiring API keys, recent-auth gates, credential rotation |
| Tampering | Replaced CI action/image, modified output, forged webhook | Hash lock, SHA-pinned Actions, SBOM, Cosign, digest deploy, backup checksums, webhook HMAC | Pin/verify all base images and models, encrypted/tamper-evident backups/audit |
| Repudiation | User/attacker denies operation; audit IP is wrong or logs are modified | Request IDs, audit model, normalized IP, Loki | Append-only signed export, clock checks, retention and investigation runbooks |
| Information disclosure | PDF/backup/secret leakage, verbose health errors, AI/vendor disclosure | Owner filters, opaque previews, readiness redaction, CSP, TLS, restricted origin and backup modes | Encrypt off-site and DB fields, vendor controls, egress policy, data retention |
| Denial of service | Session growth, repeated PDF/AI/OCR work, zip/image/PDF bombs, disk filling | Rate limits, session cleanup, page/file caps, ClamAV, container CPU/memory limits, gVisor | Atomic derived-storage quota, job cost/concurrency, decompression/pixel caps, queue backpressure |
| Elevation of privilege | Native parser escape, worker command execution, compromised CI deploy | Non-root/read-only/cap-drop containers, gVisor, Redis auth, signed deploy | Networkless ingest worker, default-deny egress, key separation, periodic external penetration test |

## Three most credible loss scenarios

1. **Untrusted-document/resource exhaustion.** A crafted PDF/image or repeated
   legitimate operations consume CPU, RAM, temporary storage or derived-file
   capacity. gVisor reduces host escape impact, but it does not prevent account-
   level resource abuse.
2. **Credential/data exfiltration.** A database, runtime, backup or outbound-
   request compromise exposes document content or reusable credentials across
   storage, email, cache and AI integrations.
3. **Supply-chain/operations compromise.** A mutable action, image, model or
   unsupported collector changes behavior or gains CI/runtime access. Immutable
   actions and signed app images substantially reduce this, but base artifacts
   are not all pinned yet.

## Blast radius and illustrative FAIR estimate

Potential confidentiality scope includes account identifiers, uploaded and
processed documents, share URLs, audit history, operational metadata and vendor
credentials. Integrity scope includes produced PDFs, signatures, webhooks,
account security settings and deployment artifacts. Availability scope includes
this service plus other workloads sharing the VPS, which is why container limits
and disk quotas matter.

The database/user snapshot collected during the audit is deliberately not
recorded in this repository. It should be kept in the private incident workbook
and refreshed before every tabletop.

Illustrative FAIR range, not an accounting forecast:

- threat-event/material-loss probability before P0 closure: **10–25%/year**;
- loss magnitude for investigation, recovery, downtime, notification, legal and
  vendor work: **EUR 10k–75k** (regulatory fines and reputation excluded);
- annualized loss expectancy: **EUR 1k–18.75k/year**.

Replace these assumptions with actual revenue, support hours, contractual
liability, record counts and incident costs. After P0/P1 controls, the target is
2–8% probability and a smaller recovery/notification cost through contained
blast radius and proven restore procedures.

## Detection, MTTD and response

Prometheus, Grafana, Loki, request IDs, application metrics, Sentry integration,
health endpoints and multi-window SLO alerts form a good base. Current MTTD is
unknown until alert delivery and on-call acknowledgement are exercised.

| Scenario | Detection required | Target MTTD |
| --- | --- | --- |
| Origin bypass or abusive client | Host/Cloudflare deny counters, WAF/rate-limit alerts, normalized request IDs | 5 minutes |
| Credential abuse | Provider sign-in/use alerts, impossible-source detection, key fingerprint in audit logs | 5 minutes |
| Parser crash/escape attempt | Sanitizer/ClamAV reject ratio, gVisor violations, OOM/restart and unusual egress | 5 minutes |
| Disk/quota exhaustion | Filesystem forecast, per-tenant reservations, temp/media/object growth and queue age | 10 minutes |
| Worker wedged/restart loop | Heartbeat, oldest queued job, restart count and success-rate burn | 5 minutes |
| Backup failure or unencrypted target | Signed backup manifest, remote verification, restore-canary age | 15 minutes |

Required incident playbooks:

1. leaked application/provider credential;
2. malicious document/parser compromise;
3. ransomware/data deletion and restore;
4. personal-data breach and vendor notification;
5. compromised CI/action/image/model;
6. Redis/Celery task injection or queue poisoning;
7. Cloudflare/origin routing failure.

Run one quarterly tabletop and one annual clean-host technical recovery. Record
detection time, decision time, containment time, recovery time, evidence gaps and
owners. A runbook that has not been exercised is not recovery evidence.

## GDPR and privacy readiness

For a personal-data breach likely to create risk, GDPR Article 33 requires
notification to the competent supervisory authority without undue delay and,
where feasible, within 72 hours of awareness. Article 34 requires communication
to affected people without undue delay when the risk is high. Legal counsel/DPO
must make the case-specific decision; engineering must preserve facts and the
decision timeline.

Minimum engineering packet:

- when/how discovered, affected systems/time range and controller/processor role;
- categories and approximate number of people/records;
- exposed fields/documents, encryption status and realistic consequences;
- containment, credential revocation, deletion/recovery and prevention steps;
- vendor notifications and their evidence;
- contact point, authority/user decision, justification and timestamps.

Reference: <https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679>

Privacy production gates also include retention/deletion schedules, DSAR export
and erasure verification, an analytics/AI lawful-basis decision, a public
subprocessor list, regional transfer assessment, DPA/SCC evidence where needed,
and a clear statement of whether document content leaves the VPS.

## Vendor and supply-chain checklist

| Dependency class | Examples in this system | Required evidence |
| --- | --- | --- |
| Edge/storage | Cloudflare proxy and object storage | Account MFA, scoped tokens, DPA/region, encryption, access logs, deletion/versioning |
| AI/model | Hosted AI API, Hugging Face/model publisher | Content/metadata sent, no-training/retention terms, immutable model digest, opt-out/local mode |
| Monitoring/analytics | Error reporting, first-party analytics, Grafana stack | PII scrubbing, retention, access roles, consent/lawful basis, breach SLA |
| Delivery/backups | SMTP provider and encrypted off-site remote | Scoped credentials, DPA, retention, encryption, restore/export and breach notification |
| Build/distribution | GitHub Actions, GHCR, PyPI, Debian images | SHA/digest pins, provenance, SBOM, signature identity, protected environment and minimal token permissions |
| Trust/signing | Timestamp/revocation/certificate endpoints | SSRF-safe fetch, timeouts, trust policy, failure semantics, no unrestricted network |

Special note: the Trivy ecosystem had a supply-chain incident in March 2026.
Scanner binaries/actions must therefore be pinned and provenance-verified, not
trusted solely because a familiar tag exists.

## Validation evidence

- complete Django non-E2E suite: **884/884 passed** on PostgreSQL;
- targeted upload/quarantine/ClamAV/API/job suite: **50/50 passed**;
- targeted shell/CSP/preview accessibility suite: **6/6 passed**;
- Django production check with warnings as failures: passed;
- Ruff lint and formatting for the same paths as CI: passed (96 Python files);
- `makemigrations --check --dry-run`: no missing model changes;
- Bandit SAST: no reportable finding;
- `pip-audit --strict`: no known vulnerability, one documented temporary waiver;
- Gitleaks 8.30.1: 19 commits / about 7.75 MB scanned, no leak found;
- Trivy 0.72 binary checksum and Sigstore bundle: verified;
- Trivy Dockerfile configuration: 0 HIGH/CRITICAL;
- Trivy built-image OS packages: 0 HIGH/CRITICAL;
- Trivy language finding: the documented `cryptography` waiver. Two additional
  packages reported from a third-party base SBOM were proven absent from the
  actual root filesystem and treated as scanner false positives;
- live origin request from outside the trusted path: denied;
- live public readiness after infrastructure changes: healthy;
- live web replicas and worker: running under current gVisor with zero restarts
  after controlled recreation.

## Release gates

Security-remediation rollout:

- [ ] rotate/revoke credentials in a documented order and verify the old values fail;
- [ ] produce an encrypted, verified pre-deploy backup;
- [ ] push through required CI, image scan, SBOM, Cosign signature and exact-digest deploy;
- [ ] apply migration, canary one replica, smoke upload/unprotect/preview/job/download;
- [ ] verify CSP, origin denial, gVisor runtime, queues, errors and rollback digest;
- [ ] observe at least one normal traffic window before closing the change.

General availability / new-feature gate:

- [ ] all items above;
- [ ] encrypted off-site restore drill with recorded RPO/RTO;
- [ ] atomic derived-storage and operation quota;
- [ ] supported log collector and worker health/queue alerts;
- [ ] encrypted secret fields, hashed bearer tokens and scoped API keys;
- [ ] pinned/verified external images and model artifact;
- [ ] privacy/vendor register and incident tabletop completed.
