# Temporary security advisory waivers

Every waiver in this file must identify the exact advisory, demonstrate why
the vulnerable path is not reachable, and have a removal condition. Waivers
are not blanket permission to ignore new findings.

## PYSEC-2026-3552 / CVE-2026-69247

- **Dependency:** `cryptography` 49.x
- **Upstream fix:** 50.0.0
- **Reason:** no stable 50.0.0 release is available yet. The affected surface
  is automatic decryption of attacker-controlled PKCS#7 EnvelopedData at
  high volume. This application does not import or call the PKCS#7 decryption
  APIs; its cryptography usage is limited to certificate/key generation for
  PDF signing.
- **Compensating control:** CI audits the exact production lockfile and ignores
  only `PYSEC-2026-3552`; all other advisories remain build-blocking.
- **Removal condition:** upgrade to cryptography 50.x and remove the CI ignore
  as soon as 50.0.0 is stable and the signing/passkey test suites pass.
- **Review deadline:** 2026-09-01, or immediately on the 50.0.0 release,
  whichever comes first.

References: [upstream advisory](https://github.com/advisories/GHSA-g6cj-pr64-35w5),
[cryptography changelog](https://cryptography.io/en/stable/changelog/).
