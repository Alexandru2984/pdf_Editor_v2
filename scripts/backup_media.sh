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

BACKUP_DIR="${BACKUP_DIR:-/var/backups/pdfeditor}"
RETENTION_DAYS="${RETENTION_DAYS:-14}"
MEDIA_DIR="${MEDIA_DIR:-./media}"

cd "$(dirname "$0")/.."

if [[ ! -d "$MEDIA_DIR" ]]; then
    echo "✗ media dir not found at $(pwd)/${MEDIA_DIR} — refusing to run." >&2
    exit 1
fi

mkdir -p "$BACKUP_DIR"

STAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
OUT="$BACKUP_DIR/media-${STAMP}.tar.gz"

echo "→ Snapshotting ${MEDIA_DIR} to ${OUT}…"
# `temp/` is scratch space for in-flight ops; skip it. Excluding via
# --exclude rather than enumerating dirs so a freshly-created category
# (added later) still gets picked up.
tar --create --gzip \
    --exclude="${MEDIA_DIR}/temp" \
    --exclude="${MEDIA_DIR}/temp/*" \
    --file="$OUT" \
    "$MEDIA_DIR"

SIZE="$(du -h "$OUT" | cut -f1)"
echo "✓ Wrote ${OUT} (${SIZE})"

echo "→ Rotating media snapshots older than ${RETENTION_DAYS} days…"
find "$BACKUP_DIR" -type f -name "media-*.tar.gz" \
     -mtime "+${RETENTION_DAYS}" -print -delete

echo "✓ Media backup done."
