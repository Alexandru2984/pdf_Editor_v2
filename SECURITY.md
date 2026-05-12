# Security policy

Thanks for taking the time to make PDF Editor safer.

## Reporting a vulnerability

Please **do not open a public GitHub issue** for security reports. Instead,
email the maintainer directly:

- **Contact:** `984.alexmihai@gmail.com`
- **Expected first reply:** within **5 working days**
- **Encrypted comms:** if you'd like to use PGP, ask in the first email
  and we'll exchange keys

Useful info to include:

- A clear description of the bug and its impact
- Steps to reproduce (a proof-of-concept or curl command is gold)
- The affected version (commit SHA or live URL if it's a production issue
  on `pdf.micutu.com`)
- Anything you've already tried for mitigation

## Scope

In scope:

- `pdf.micutu.com` and any subpath under it
- The container image at `ghcr.io/alexandru2984/pdf_editor_v2`
- Source code in this repository

Out of scope (please don't report):

- Best-practice complaints unrelated to a concrete attack (e.g. "you
  could add header X")
- Self-XSS scenarios that require the victim to paste attacker-controlled
  HTML into their own devtools
- Reports generated only by automated scanners with no manual validation
- Issues on third-party infrastructure (Cloudflare, Groq API, etc.) that
  we don't control
- Social-engineering attacks on individual users or contributors
- DoS via raw traffic volume (rate limits + Cloudflare handle that)

## What to expect

- An acknowledgement within 5 working days
- A fix or risk-acceptance decision within **30 days** for medium+
  severity issues — usually much faster
- Credit in the release notes (with your permission) when the fix ships
- No bug-bounty payment (the project is free-to-use, not commercial)

## What we already do

So you have a sense of the baseline:

- Dependency CVE scan (`pip-audit`) runs in CI on every push
- Python SAST (`bandit`) runs in CI on every push
- Container image vulnerability scan (`trivy`) runs before every deploy
  — deploys fail on CRITICAL or HIGH CVEs with fixes available
- Content-Security-Policy enforced at the edge (nginx)
- HSTS preload, HTTPS-only, secure cookies in production
- API key tokens stored as SHA-256 hashes (plaintext shown once)
- All file access scoped per owner; path traversal blocked via realpath
- Rate limits on every state-changing endpoint
- Audit log captures who did what, when, from where

## Hall of fame

(Empty for now — be the first!)
