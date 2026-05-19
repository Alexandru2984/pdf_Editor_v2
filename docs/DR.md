# Disaster Recovery — PDF Editor

What we'd do if the VPS caught fire tomorrow. Concrete numbers
(RTO/RPO), the drill that proves they're real, and the gaps we know
about so they don't surprise anyone at 3am.

## Targets

| Metric | Target | What it means |
|--------|--------|---------------|
| **RPO** (data loss) | ≤ 24h | Worst case we lose one day of writes. Backups run daily at 03:00 UTC; anything between two snapshots is gone. |
| **RTO** (downtime)  | ≤ 4h  | From "VPS is unrecoverable" to "users can log in again", including provisioning a new box. Restore alone is minutes; the rest is plumbing. |

Both numbers assume the same datacenter region. Cross-region failover
is not in scope — see [Known gaps](#known-gaps).

## What gets backed up

| What | Where | Cadence | Retention | Script |
|------|-------|---------|-----------|--------|
| Postgres (pdfeditor DB, custom-format dump) | `/var/backups/pdfeditor/*.sql.gz` | daily 03:00 UTC | 14 days | `scripts/backup_db.sh` |
| Media tree (`./media/`, minus `temp/`) | `/var/backups/pdfeditor/media-*.tar.gz` | daily 03:30 UTC | 14 days | `scripts/backup_media.sh` |

The 30-minute offset between the two jobs keeps them from competing
for disk IO on a small VPS.

Cron entries (root crontab on the VPS):

```cron
0  3 * * * cd /home/micu/pdf_Editor_v2 && scripts/backup_db.sh    >> /var/log/pdfeditor-backup.log 2>&1
30 3 * * * cd /home/micu/pdf_Editor_v2 && scripts/backup_media.sh >> /var/log/pdfeditor-backup.log 2>&1
```

The DB dump is the system of record for *relationships*. The media
tarball is the system of record for the *bytes behind those rows*.
You need both — restoring one without the other leaves rows pointing
at files that don't exist, or files no row references.

## The drill

A backup you haven't restored isn't a backup — it's a wish.
`scripts/restore_drill.sh` proves the most recent dump is actually
recoverable: spins up an ephemeral `pgvector/pgvector:pg16` container,
restores into it, runs sanity SQL, reports RTO. Tears the container
down on the way out (or on error, via trap). Does **not** touch the
production stack.

Run it:

```bash
# Use the latest dump in /var/backups/pdfeditor/
scripts/restore_drill.sh

# Or against a specific file
scripts/restore_drill.sh /var/backups/pdfeditor/pdfeditor-2026-05-19T00-50-01Z.sql.gz
```

What it checks:

1. Container comes up and Postgres accepts connections.
2. `pg_restore --jobs=2` completes without error.
3. `pgvector` extension is present in the restored DB.
4. Core tables exist with at least the floor row counts in `FLOORS`
   (missing tables = partial restore = fail).
5. No orphan `pdfeditor_job` rows referencing deleted users (proves
   FKs survived intact, not just that rows count up).
6. Reports the time breakdown so RTO claims stay honest.

Last measured drill (4.5MB dump, 700 jobs, 6303 uploads, 8913 processed):

```
── RTO breakdown ─────────────────────────
  container up + ready :    2s
  pg_restore           :    2s
  sanity checks        :    1s
  TOTAL                :    5s
──────────────────────────────────────────
```

Five seconds is the floor for restore alone. The 4-hour RTO target
absorbs everything around it: noticing the outage, getting on the
VPS (or provisioning a new one), pulling backups from wherever they
live, deploying the stack, DNS sanity-check.

### Cadence

Run the drill **quarterly**, on the first weekday of each quarter
(Jan / Apr / Jul / Oct). Calendar it. A drill that "we'll run when we
get to it" never runs.

If the drill ever fails — drop everything and fix it before the next
prod write happens. A failing drill means the next real outage is the
one where you find out.

## Recovering for real

If this is an actual incident, not a drill:

### Scenario A — DB corruption, VPS otherwise healthy

```bash
# 1. Stop the app so it can't write more bad data.
docker compose stop web worker

# 2. Confirm which dump you want. Usually the newest pre-incident.
ls -lt /var/backups/pdfeditor/*.sql.gz | head

# 3. Wipe the broken DB. THIS IS DESTRUCTIVE. Be sure.
docker compose exec db psql -U "$POSTGRES_USER" postgres -c \
    "DROP DATABASE pdfeditor; CREATE DATABASE pdfeditor;"

# 4. Restore. --jobs requires a real file, so unpack first.
gunzip -c /var/backups/pdfeditor/pdfeditor-<STAMP>.sql.gz \
    > /tmp/restore.dump
docker compose exec -T db pg_restore \
    --username="$POSTGRES_USER" --dbname=pdfeditor \
    --no-owner --no-privileges --jobs=2 < /tmp/restore.dump
rm /tmp/restore.dump

# 5. Bring the app back.
docker compose up -d web worker
```

### Scenario B — VPS is gone

1. Provision a new box (same region, same OS, Docker installed).
2. Restore SSH keys, point DNS A record at new IP, wait for propagation.
3. `git clone` the repo into `/home/<user>/pdf_Editor_v2`.
4. Restore `.env` from your password manager (we don't back it up —
   secrets in plaintext snapshots is a worse risk than re-keying).
5. Copy the most recent `pdfeditor-*.sql.gz` and `media-*.tar.gz`
   onto the new box. **This step depends on you having an offsite
   copy** — see [Known gaps](#known-gaps) below.
6. Unpack media: `tar -xzf media-<STAMP>.tar.gz` into the project root.
7. Run Scenario A steps 3-5 against the new stack.

Expected wall-clock: 30-60 min if you have offsite backups handy and
DNS propagates fast. Add another hour or two if the previous bullet
isn't true.

## Known gaps

These are flagged on purpose — we know they exist, we've decided not
to fix them yet, and we want anyone responding to an incident to know
they exist before they discover them under pressure.

### No offsite backups

Both `*.sql.gz` and `media-*.tar.gz` live on `/var/backups/` on the
same VPS that hosts the live data. A VPS-wide failure (DC fire,
ransomware that gets root) takes the backups with it.

Mitigation when willing to spend on it: `rclone sync /var/backups/
b2:pdfeditor-backups/` to Backblaze B2 (or any S3-compatible) on a
cron, with a write-only key. ~$0.005/GB/month — pennies for this
data volume.

### No PITR (point-in-time recovery)

We dump once a day. We don't archive WAL, so the smallest recoverable
window is "yesterday's snapshot". For a user-facing app that's an
acceptable RPO; for anything financial it wouldn't be.

If we ever care: turn on `archive_mode=on` in `postgresql.conf`,
ship WAL segments offsite alongside the dumps, and use `pg_basebackup`
+ `pg_waldump` to roll forward.

### Media backup isn't drilled

`restore_drill.sh` only validates the DB. The media tarball is just
`tar` — recovering it is trivial — but "trivial" is also what we
said about the DB backup before discovering it had been silently
failing for two weeks (commit fixing it: `scripts/backup_db.sh`,
`|| true` guard on the `grep .env` lookup).

A minimal extension: `tar -tzf media-<STAMP>.tar.gz | head` to confirm
the archive is readable. Add it to the drill if disk-bytes-on-restore
ever bites.

### `.env` is not backed up

Deliberate. Backing up secrets in plaintext snapshots that get rsync'd
around is a worse risk surface than the inconvenience of re-keying
once. Store `.env` in a password manager and treat it like any other
credential.

## Related

- `scripts/backup_db.sh` — Postgres backup
- `scripts/backup_media.sh` — media tarball
- `scripts/restore_drill.sh` — quarterly proof-of-restore
- `docs/RUNBOOK.md` — what to do *during* an incident
- `docs/SLOs.md` — what "healthy" looks like, so we know when we've
  recovered
