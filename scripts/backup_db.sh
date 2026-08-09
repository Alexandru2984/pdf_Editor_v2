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
umask 077

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
chmod 0700 "$BACKUP_DIR"

# Repair permissions from older script versions before creating a new dump.
# Database dumps contain password hashes, audit metadata, and application
# secrets, so they must never be readable by other local VPS users.
find "$BACKUP_DIR" -maxdepth 1 -type f \
    \( -name "${POSTGRES_DB}-*.sql.gz" -o -name "${POSTGRES_DB}-*.sql.gz.sha256" \) \
    -exec chmod 0600 {} +

STAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
OUT="$BACKUP_DIR/${POSTGRES_DB}-${STAMP}.sql.gz"
OUT_NAME="$(basename "$OUT")"
TMP="$(mktemp "$BACKUP_DIR/.${OUT_NAME}.XXXXXX")"
CHECKSUM_TMP=""

cleanup_tmp() {
    if [[ -n "$TMP" && -e "$TMP" ]]; then
        rm -f -- "$TMP"
    fi
    if [[ -n "$CHECKSUM_TMP" && -e "$CHECKSUM_TMP" ]]; then
        rm -f -- "$CHECKSUM_TMP"
    fi
}
trap cleanup_tmp EXIT

echo "→ Dumping ${POSTGRES_DB} to ${OUT}…"
# `-Fc` (custom format) gives us parallel restore + per-table selection
# at restore time. Pipes through gzip for one less file to babysit.
docker compose exec -T "$DB_SERVICE" \
    pg_dump --format=custom --compress=0 \
            --username="$POSTGRES_USER" "$POSTGRES_DB" \
    | gzip -9 > "$TMP"

# Publish only a complete dump, then write its integrity sidecar atomically.
chmod 0600 "$TMP"
mv -- "$TMP" "$OUT"
TMP=""
CHECKSUM_TMP="$(mktemp "$BACKUP_DIR/.${OUT_NAME}.sha256.XXXXXX")"
(
    cd "$BACKUP_DIR"
    sha256sum "$OUT_NAME"
) > "$CHECKSUM_TMP"
chmod 0600 "$CHECKSUM_TMP"
mv -- "$CHECKSUM_TMP" "${OUT}.sha256"
CHECKSUM_TMP=""

SIZE="$(du -h "$OUT" | cut -f1)"
echo "✓ Wrote ${OUT} (${SIZE})"

# Rotate. -delete is silent; explicit -print so the log shows what went.
echo "→ Rotating dumps older than ${RETENTION_DAYS} days…"
find "$BACKUP_DIR" -maxdepth 1 -type f \
     \( -name "${POSTGRES_DB}-*.sql.gz" -o -name "${POSTGRES_DB}-*.sql.gz.sha256" \) \
     -mtime "+${RETENTION_DAYS}" -print -delete

echo "✓ Backup done."
