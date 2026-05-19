#!/usr/bin/env bash
# Daily Postgres backup for the PDF Editor stack.
#
# Dumps the `db` container's database with pg_dump (custom format, so we
# can restore with pg_restore --jobs=N), gzips it, and rotates anything
# older than RETENTION_DAYS.
#
# Designed to be run from cron on the VPS. Idempotent. Exits non-zero
# on any failure so cron's MAILTO catches it.
#
# Usage:
#   scripts/backup_db.sh
#
# Suggested crontab (as root, since /var/backups is root-owned):
#   0 3 * * *  cd /home/micu/pdf_Editor_v2 && scripts/backup_db.sh >> /var/log/pdfeditor-backup.log 2>&1
set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-/var/backups/pdfeditor}"
RETENTION_DAYS="${RETENTION_DAYS:-14}"
COMPOSE_PROJECT="${COMPOSE_PROJECT:-pdf_editor_v2}"
DB_SERVICE="${DB_SERVICE:-db}"

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
    echo "✗ .env not found in $(pwd) — refusing to run." >&2
    exit 1
fi

# Source POSTGRES_USER / POSTGRES_DB from .env without polluting the
# environment with unrelated keys. `|| true` because grep exits 1 when
# the key is absent (legitimate — docker-compose.yml supplies defaults);
# without the guard `set -e` would kill the backup silently.
POSTGRES_USER="$(grep -E '^POSTGRES_USER=' .env | head -n1 | cut -d= -f2- || true)"
POSTGRES_DB="$(grep -E '^POSTGRES_DB=' .env | head -n1 | cut -d= -f2- || true)"
: "${POSTGRES_USER:=pdfeditor}"
: "${POSTGRES_DB:=pdfeditor}"

mkdir -p "$BACKUP_DIR"

STAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
OUT="$BACKUP_DIR/${POSTGRES_DB}-${STAMP}.sql.gz"

echo "→ Dumping ${POSTGRES_DB} to ${OUT}…"
# `-Fc` (custom format) gives us parallel restore + per-table selection
# at restore time. Pipes through gzip for one less file to babysit.
docker compose exec -T "$DB_SERVICE" \
    pg_dump --format=custom --compress=0 \
            --username="$POSTGRES_USER" "$POSTGRES_DB" \
    | gzip -9 > "$OUT"

SIZE="$(du -h "$OUT" | cut -f1)"
echo "✓ Wrote ${OUT} (${SIZE})"

# Rotate. -delete is silent; explicit -print so the log shows what went.
echo "→ Rotating dumps older than ${RETENTION_DAYS} days…"
find "$BACKUP_DIR" -type f -name "${POSTGRES_DB}-*.sql.gz" \
     -mtime "+${RETENTION_DAYS}" -print -delete

echo "✓ Backup done."
