#!/usr/bin/env bash
# Daily media backup for the PDF Editor stack.
#
# Snapshots the on-disk media tree (uploads/processed/thumbs) into a
# rotated tar.gz. The DB-only backup in scripts/backup_db.sh is the
# system of record, but without this companion script a disk failure
# would orphan every PDF — the rows would survive a DB restore, the
# files behind them wouldn't.
#
# Designed to be run from cron on the VPS. Idempotent. Exits non-zero
# on any failure so cron's MAILTO catches it.
#
# Usage:
#   scripts/backup_media.sh
#
# Suggested crontab (offset from backup_db.sh so the two don't compete
# for IO on a small VPS):
#   30 3 * * *  cd /home/micu/pdf_Editor_v2 && scripts/backup_media.sh >> /var/log/pdfeditor-backup.log 2>&1
set -euo pipefail
umask 077

BACKUP_DIR="${BACKUP_DIR:-/var/backups/pdfeditor}"
RETENTION_DAYS="${RETENTION_DAYS:-14}"
MEDIA_DIR="${MEDIA_DIR:-./media}"

cd "$(dirname "$0")/.."

if [[ ! -d "$MEDIA_DIR" ]]; then
    echo "✗ media dir not found at $(pwd)/${MEDIA_DIR} — refusing to run." >&2
    exit 1
fi

mkdir -p "$BACKUP_DIR"
chmod 0700 "$BACKUP_DIR"

# Repair permissions from older script versions before creating a new archive.
find "$BACKUP_DIR" -maxdepth 1 -type f \
    \( -name "media-*.tar.gz" -o -name "media-*.tar.gz.sha256" \) \
    -exec chmod 0600 {} +

STAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
OUT="$BACKUP_DIR/media-${STAMP}.tar.gz"
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

echo "→ Snapshotting ${MEDIA_DIR} to ${OUT}…"
# `temp/` is scratch space for in-flight ops; skip it. Excluding via
# --exclude rather than enumerating dirs so a freshly-created category
# (added later) still gets picked up.
tar --create --gzip \
    --exclude="${MEDIA_DIR}/temp" \
    --exclude="${MEDIA_DIR}/temp/*" \
    --file="$TMP" \
    "$MEDIA_DIR"

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

echo "→ Rotating media snapshots older than ${RETENTION_DAYS} days…"
find "$BACKUP_DIR" -maxdepth 1 -type f \
     \( -name "media-*.tar.gz" -o -name "media-*.tar.gz.sha256" \) \
     -mtime "+${RETENTION_DAYS}" -print -delete

echo "✓ Media backup done."
