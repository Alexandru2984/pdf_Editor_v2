# Production and product roadmap

This roadmap turns the current capable PDF toolbox into a finished, trustworthy
product. Priorities are risk- and dependency-driven: P0 protects the existing
service, P1 makes operation repeatable, P2 makes the experience coherent, P3
adds differentiated document workflows, and P4 adds team/commercial scale.

Dates are planning ranges, not promises. Each phase ships in small, reversible
commits and signed canary deployments.

## North star

A user should be able to open the site on a 320px phone or desktop, understand
what to do without training, upload a document safely, complete an operation in
the background, recover from an error, and know exactly where the document went
and when it will be deleted.

Product-level definition of success:

- first useful result in under two minutes for a new user;
- no horizontal scrolling at 320 CSS pixels;
- WCAG 2.2 AA for the application UI and a roadmap for PDF/UA output;
- 99.9% availability, p95 interactive request under one second, 99% async job
  success and 99.5% PDF operation success over the documented windows;
- every byte and expensive operation attributed to an owner/reservation;
- no document sent to a third party without a visible policy/control;
- RPO/RTO proven by restore rather than inferred from a successful upload;
- a deployment is traceable from commit to tests, SBOM, signature, digest,
  migration, canary and rollback.

## Phase map

| Priority | Planning range | Outcome |
| --- | --- | --- |
| P0 | 0–72 hours | Close credential/backup/release blockers and deploy the audited security build |
| P1 | 1–2 weeks | Bound resources, encrypt sensitive state, harden APIs/egress and make recovery observable |
| P2 | 3–6 weeks | Unified responsive UX, workspace, upload/job center and trustworthy sharing |
| P3 | 7–12 weeks | Visual editor, collaboration, advanced OCR/forms/signing and automation |
| P4 | 3–6 months | Organizations, SSO/billing, multi-tenant controls, HA and ecosystem maturity |

## P0 — release unblockers

### P0.1 Coordinated credential rotation

Rotate, do not merely edit, every in-scope credential. Use provider revocation
logs and an explicit old-credential negative test.

Order:

1. create least-privilege replacements for external AI, object storage and mail;
2. deploy consumers that accept the replacements and verify normal operations;
3. revoke old external values and alert on attempted reuse;
4. rotate Redis with a controlled broker/cache/session transition;
5. rotate PostgreSQL role password and all connection consumers;
6. rotate application signing material, accepting that sessions and outstanding
   reset/confirmation links may be invalidated;
7. rotate CI/deploy/package credentials if incident scope or provider logs make
   that necessary;
8. archive the signed rotation checklist without recording secret values.

Acceptance:

- old values fail from an isolated verification client;
- site, mail, API, jobs, object mirror and monitoring pass;
- no secret appears in shell history, process arguments, Compose output, logs,
  support tickets or Git;
- owner, timestamp, expiry/next rotation and recovery contact are recorded.

### P0.2 Encrypted and recoverable backups

- route whole-VPS backup through the existing client-side encrypted remote;
- give the backup identity write-only/append privileges where supported;
- keep local and remote retention independent;
- sign/checksum manifests and alert on age, size anomaly and verification failure;
- restore PostgreSQL plus a sample media object on a clean isolated host;
- record actual RPO/RTO and missing dependencies;
- add an immutable/offline generation to reduce ransomware blast radius.

### P0.3 Ship the audited build safely

- user reviews/pushes the local commits;
- protected CI runs the Python matrix and browser suite;
- build from hash-locked dependencies and a reviewed base digest;
- Trivy gate, SBOM and keyless Cosign signature complete;
- deploy exact digest/full commit, run migration once, canary one web replica;
- smoke login, upload, encrypted quarantine/unprotect, preview, one sync op, one
  async op, download, delete and rollback;
- verify direct origin denial, CSP, gVisor runtime and queue health;
- do not turn an unsigned local audit image into production `latest`.

### P0.4 Bound resource consumption

Introduce a storage reservation ledger instead of `SUM(size) -> write`:

- dimensions: original, sanitized source, derived output, thumbnail, temporary,
  embedding, remote mirror and retained version;
- states: reserved, committed, released, expired and orphaned;
- atomic database reservation before accepting a file/job;
- hard user/tenant quota, anonymous quota, admin emergency ceiling and VPS floor;
- operation cost units based on bytes, pages, pixels, OCR, AI tokens and expected
  expansion ratio;
- per-user concurrency, global concurrency, queue depth and wall-clock timeout;
- maximum decoded pixels, archive members/ratio, PDF object/page count and image
  dimensions before expensive decode;
- cancel/retry with idempotent cleanup and a sweeper for abandoned reservations;
- `507`/`429` responses with actionable UI, never a generic 500.

### P0.5 Replace unsupported telemetry and pin artifacts

- convert Promtail config to Grafana Alloy and review every warning;
- run both collectors briefly, compare labels/positions/gaps/duplicates, update
  dashboards and alerts, then remove Promtail;
- update observability/database/proxy images in staging;
- pin external Compose images and the Python base to manifest-list digests;
- verify signatures/provenance where publishers support it;
- attach owner, age and update SLA to every image/model exception.

## P1 — security, privacy and reliability foundation

### P1.1 Application secret storage

Build a small versioned envelope-encryption layer:

- dedicated keyring outside PostgreSQL, separate from ordinary Django signing;
- authenticated encryption with record type/owner/version as associated data;
- decrypt only at the delivery/MFA boundary;
- dual-read/new-write rotation, background re-encryption and rollback path;
- redact admin, serializer, log, error-report and export representations;
- webhook secret revealed once, with explicit rotate and previous-key overlap;
- TOTP secret encrypted; backup codes remain one-way hashed;
- migration metrics prove zero plaintext remains before dropping old columns.

Bearer values that do not need recovery are hashed instead: share tokens,
password-reset-style custom tokens and API credentials.

### P1.2 Authentication and authorization maturity

- recent-auth challenge for MFA/passkey/API-key/webhook/email/password/account
  changes;
- API key scopes (`pdf:read`, `pdf:write`, `jobs:read`, `webhooks:manage`),
  expiry, rotation overlap, optional IP/network restrictions and per-key quota;
- device/session inventory with approximate location, current-session marker,
  revoke-all and security notification;
- recovery-code regeneration revokes the previous set;
- passkey discoverable credentials and passwordless account recovery policy;
- risk-based alerts for new device, MFA disabled, key created and bulk download;
- admin roles split into support, operations, security and superuser; no casual
  unrestricted impersonation.

### P1.3 Network and SSRF containment

- separate parser, webhook and general worker networks/identities;
- default-deny egress with explicit DNS/HTTP(S) proxy policy;
- resolve once, validate every resolved IP, connect to the validated address and
  preserve TLS hostname validation;
- reject redirects to a new trust zone, non-web ports, userinfo and oversized
  responses; cap redirects/body/time;
- block link-local, loopback, private, multicast, carrier-grade NAT and cloud
  metadata ranges for IPv4 and IPv6;
- webhook destination verification/challenge and revalidation on DNS change;
- networkless ingest worker with no database/provider credentials as the target
  parser architecture.

### P1.4 Sharing and webhook trust

- hashed share token with non-secret prefix for support;
- share password, expiry, max successful downloads, domain/email allowlist and
  owner revoke-all;
- preview/HEAD/link-scanner requests do not consume a download;
- atomic counter and an explicit confirmation/download action;
- `Referrer-Policy: no-referrer`, private/no-store responses and no token in
  analytics/error reporting;
- webhook event allowlist, per-endpoint secret rotation, delivery ID/timestamp,
  replay window, idempotency key, retry schedule, dead-letter state and manual
  replay with audit log;
- store bounded/redacted response excerpts, never arbitrary sensitive bodies.

### P1.5 Reliable job platform

- web and worker healthchecks; Celery heartbeat, oldest-job and restart alerts;
- queues by cost/type (`interactive`, `pdf-heavy`, `ocr`, `ai`, `webhook`);
- fair scheduling per tenant, global backpressure and circuit breakers;
- deterministic idempotency key for every operation;
- retry only transient classes, with jitter and capped attempts;
- graceful drain on deployment, cancel signal and orphan recovery;
- dead-letter queue plus admin replay/quarantine;
- progress phases and ETA derived from real historical percentiles;
- temporary/output cleanup is transactionally tied to terminal job state.

### P1.6 Disaster recovery and data lifecycle

- PostgreSQL continuous WAL/PITR with a separate failure domain;
- object storage becomes the durable source of truth or receives synchronous
  verified replication; local media becomes a cache;
- user-configurable retention within plan bounds; anonymous retention much
  shorter than authenticated retention;
- trash with short recovery window, then verifiable purge across DB, local disk,
  object versions, embeddings, logs and backups according to policy;
- DSAR export includes account data, audit history, links, keys metadata and
  processing/vendor disclosure, but never secret material;
- quarterly clean-host restore and annual incident/tabletop exercise.

### P1.7 Supply-chain hermeticity

- pin the embedding model revision and content digest; scan and vendor it as a
  separate immutable artifact;
- prohibit model/network downloads during tests and production startup;
- version embedding algorithm/model/dimension and support controlled reindex;
- generate first-party SPDX/CycloneDX SBOM from the final image;
- verify GitHub Action SHA provenance after upstream security incidents;
- review dependency install scripts/native wheels, licenses and abandoned
  packages; remove unused packages/tools from runtime;
- multi-stage/distroless evaluation to reduce runtime packages further.

## P2 — coherent, responsive product experience

The current UI already has a design-token base, dark mode and mobile
breakpoints. The next step is not more isolated CSS; it is a single product
language and tested component system.

### P2.1 Visual system

- replace the heavy gradient/emoji mix with a restrained neutral canvas, one
  brand accent, consistent SVG icon set and clear document-centric hierarchy;
- define semantic tokens for surface, border, text, focus, success, warning,
  danger and disabled states in light/dark/high-contrast themes;
- typography scale, 4/8px spacing rhythm, consistent radii/shadows and maximum
  reading widths;
- reusable buttons, fields, selects, dialogs, toasts, progress, empty/error
  states, cards, tables, tabs and upload zones;
- visible loading/skeleton, success, partial-success, blocked, retry and offline
  states for every asynchronous action;
- remove per-template styles/scripts and then remove CSP `style-src
  'unsafe-inline'`;
- use hashed static filenames instead of manual query-string cache busting.

### P2.2 Responsive and accessibility matrix

Required viewport checks: 320, 360, 390, 412, 768, 1024 and 1440 CSS pixels,
portrait/landscape where relevant.

Required interaction checks:

- keyboard-only, visible logical focus, skip links, modal focus trap/restore and
  Escape behavior;
- VoiceOver/Safari and NVDA/Firefox smoke flows;
- correct names/roles/states, live regions and form error associations;
- 44px touch targets, safe-area insets, zoom to 200/400%, large text and long
  Romanian/English labels;
- reduced motion, forced colors/high contrast, dark mode and contrast audit;
- axe-core with no serious/critical violations, plus manual WCAG 2.2 AA review;
- Playwright visual snapshots for phone/tablet/desktop on dashboard and each
  component state;
- no horizontal scroll or clipped dialogs at 320px.

### P2.3 Onboarding and information architecture

- concise landing value proposition and privacy promise;
- first-run guided upload with example/safe demo document;
- operation picker grouped by Organize, Edit, Convert, Secure, OCR/AI and Sign;
- command/search palette for tools instead of a long card wall;
- contextual recommended next actions after each result;
- clear local/cloud/AI processing badge before the operation starts;
- persistent Help/shortcuts, examples, limitations and error recovery;
- Romanian and English parity; translation completeness gate in CI.

### P2.4 Workspace and document management

- folders/projects, tags, favorites, rename and duplicate;
- full-text metadata search and filters by type/date/status/size;
- grid/list toggle, sort, multi-select and bulk action toolbar;
- recent documents, version history, trash/restore and retention countdown;
- stable document UUID URLs and breadcrumb navigation;
- thumbnail virtualization/lazy loading and pagination/cursor loading;
- document details drawer: pages, size, encryption, sanitizer status, hashes,
  retention and processing history;
- checksum-based duplicate detection with explicit reuse choice;
- saved operation presets and reusable workflow templates.

### P2.5 Upload manager

- drag/drop, paste and mobile file/camera chooser;
- per-file validation before upload with exact size/type/page feedback;
- resumable/chunked upload, pause/resume/cancel and retry only failed chunks;
- upload queue survives navigation; background/PWA continuation where browsers
  allow it;
- multi-file progress, duplicate/conflict handling and partial-success summary;
- malware/sanitization/quarantine states explained in plain language;
- direct-to-object-storage upload only after signed reservation and with server-
  side completion verification;
- mobile scan capture with crop, rotate, deskew, contrast and multipage review.

### P2.6 Unified job and result center

- global job drawer/list with queued/running/final state, progress, ETA and cost;
- cancel, safe retry, duplicate-as-new and failure diagnostics;
- browser/email/web-push completion notification preferences;
- compare before/after size, pages and visual preview;
- download one, ZIP selected, save as new version, share or start another tool;
- output expiry countdown and one-click extend within quota;
- job history filters/export and support correlation ID copy.

### P2.7 Trustworthy sharing

- share creation wizard showing exactly who can access, expiry and download cap;
- password/domain/email restriction and one-time-link option;
- branded recipient page with filename/size/page count and owner identity policy;
- revoke/extend/rotate from one dashboard;
- access history without invasive tracking;
- safe social/email preview that does not consume or expose the document;
- optional watermarked recipient copy.

## P3 — differentiated PDF workflows

### P3.1 Visual document editor

- page canvas with zoom, thumbnails and responsive split-pane/full-screen mode;
- add/edit/move text with embedded-font awareness and explicit fallback;
- add/replace/crop/rotate images; shapes, lines, freehand drawing and stamps;
- highlight, underline, strikeout, sticky notes and comments;
- links, bookmarks, page labels, headers/footers, background and Bates numbering;
- multi-select/alignment/snap, keyboard shortcuts, undo/redo and autosaved drafts;
- non-destructive versioning plus flatten/export;
- visual diff before commit so edits never silently alter unrelated content.

### P3.2 Advanced organize/convert

- visual page extraction, insertion from another document and blank-page add;
- duplex scan cleanup, blank-page detection and automatic orientation;
- N-up/contact sheet, booklet imposition and print presets;
- reliable Office/HTML/image conversions in a separately sandboxed service;
- PDF portfolio/attachment workflow only with explicit sanitization and safe
  extraction policy;
- batch rename and metadata normalization;
- per-operation quality/size preview and target-size compression.

### P3.3 OCR, accessibility and intelligence

- language packs selectable per document/page, including mixed-language OCR;
- deskew, denoise, threshold, orientation and table/layout recognition;
- searchable text overlay quality score and manual correction;
- PDF/UA checker: title/language, headings, reading order, alt text, table
  semantics, bookmarks, contrast and form labels;
- guided accessibility remediation and PDF/UA export report;
- semantic search/Q&A with page citations and highlighted source spans;
- structured extraction for invoices/forms/tables to JSON/CSV;
- summarization templates, translation, classification and document comparison;
- PII detection/redaction suggestions with mandatory human confirmation;
- local/private AI option, provider selector, token/cost preview and no-training
  policy display.

### P3.4 Forms and signing

- visual AcroForm designer for text, checkbox, radio, select, date and signature;
- validation, calculations, required fields, tab order and accessible labels;
- import/export form data and reusable form templates;
- multi-recipient signature request, ordered/parallel routing, reminders and
  decline/expiry;
- signer identity policy, consent, evidence package and tamper-evident audit;
- signature appearance templates, certificate chain/timestamp/LTV validation;
- clearly separate simple electronic signatures from advanced/qualified legal
  assurance; obtain legal review before making compliance claims;
- organization seal/stamp and signed webhook on completion.

### P3.5 Collaboration and automation

- comments, mentions, assignments and approval/review states;
- document activity timeline and version compare/restore;
- shared project folders and role-based access;
- workflow builder: trigger -> operations -> approval -> delivery;
- watch-folder/email-ingest/API triggers with malware/quota controls;
- scheduled retention/export and webhook delivery;
- templates for redact-and-share, OCR-and-archive, sign-and-deliver and batch
  invoice extraction;
- webhook replay UI and per-workflow idempotency.

## P4 — team, platform and commercial scale

### Organizations and enterprise access

- organizations/workspaces with owner, admin, editor, reviewer and viewer roles;
- invitations, domain verification and controlled external guests;
- OIDC/SAML SSO, SCIM provisioning and group-to-role mapping;
- enforced MFA/passkeys, session lifetime, IP/network policy and device posture;
- tenant-specific retention, AI/provider policy and data residency;
- legal hold, audit export and customer-managed key evaluation.

### API and developer ecosystem

- versioned OpenAPI contract and compatibility/deprecation policy;
- idempotency keys on every write, async operation resources and consistent
  problem-details errors;
- cursor pagination, filtering, sparse fields and request correlation;
- OAuth2/service accounts in addition to scoped API keys;
- webhook signing SDK, replay tooling and event schema registry;
- official typed Python and TypeScript SDKs with examples/tests;
- sandbox environment, API explorer, usage dashboard and quota headers;
- CLI for upload, batch workflows, polling, download and verification;
- integrations for Zapier/n8n/Make and object-storage import/export.

### Plans, billing and support

- plans expressed as quota/capability policy, not scattered conditionals;
- metering ledger for storage, pages, OCR, AI tokens and operations;
- invoices/tax/VAT and payment provider integration after legal/accounting design;
- grace period, downgrade/archive/export behavior and cost alerts;
- coupon/trial/referral only after abuse controls;
- public status page, incident history and maintenance communication;
- support bundle with redacted diagnostics and correlation IDs;
- feature flags with owner, expiry and safe rollback.

### Scale and availability

- staging environment with anonymized fixtures and production-equivalent policy;
- object storage as durable media, stateless web replicas and disposable workers;
- managed/HA PostgreSQL with PITR and tested failover when economics justify it;
- Redis HA or durable broker separation; do not treat cache and task durability as
  the same requirement;
- multiple worker pools/autoscaling based on queue age/cost, not CPU alone;
- canary/blue-green deployment, online schema-change discipline and feature flags;
- capacity model, regular load/soak tests, chaos exercises and cost dashboards;
- regional deployment/data residency only after tenant isolation and operations
  can support it safely.

## Cross-cutting quality gates

Every user-facing increment is complete only when:

- threat/abuse cases and data-flow changes are reviewed;
- authorization has positive, cross-owner and anonymous negative tests;
- quotas, timeouts, cancellation, cleanup and retry behavior are defined;
- API/UI errors are actionable and do not disclose internals;
- mobile 320/390, tablet 768 and desktop 1440 flows pass;
- keyboard, reduced motion, contrast and axe checks pass;
- Romanian/English strings are complete;
- unit/integration/E2E tests, migration forward/backward plan and load impact pass;
- metrics, log redaction, alert and runbook are updated;
- privacy/retention/vendor implications are documented;
- artifact is hash/digest pinned, SBOMed, signed, canaried and rollback-tested.

## Suggested first implementation sequence after P0

1. Storage reservation/quota ledger and cleanup sweeper.
2. Worker health, queue-age alerts, graceful drain and dead-letter handling.
3. Envelope-encrypted TOTP/webhook secrets and hashed share tokens.
4. Scoped/expiring API keys plus recent-auth gates.
5. Egress proxy and isolated networkless ingest worker proof of concept.
6. Alloy migration and digest-pinned external images.
7. Workspace/upload/job-center information architecture.
8. Design component extraction, visual regression and WCAG gate.
9. Version history/trash/retention and safe sharing redesign.
10. Visual annotation/editor MVP, then OCR/accessibility/forms/signing epics.

This order closes failure modes that would otherwise become more expensive as
usage and features grow. A finished product is not the one with the longest
tool list; it is the one whose security, recovery and interaction model remain
predictable when every tool is used concurrently and something fails.
