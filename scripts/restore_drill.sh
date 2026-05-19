#!/usr/bin/env bash
# Restore drill — proves the most recent backup actually works.
#
# Spins up an ephemeral pgvector container, restores the latest
# DB dump into it, runs sanity SQL against the restored data, and
# reports a measured RTO. Tears the container down on the way out
# (or on error, via trap). Does NOT touch the production stack.
#
# Run this on a quarterly cadence (see docs/DR.md). A backup you
# haven't restored isn't a backup — it's a wish.
#
# Usage:
#   scripts/restore_drill.sh                      # pick newest dump
#   scripts/restore_drill.sh <path-to-dump.gz>    # specific dump
#
# Exits non-zero if:
#   * no dump found
#   * restore fails
#   * any sanity check returns 0 rows where we expect ≥ 1
set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-/var/backups/pdfeditor}"
DRILL_PORT="${DRILL_PORT:-55433}"
DRILL_DB="${DRILL_DB:-pdfeditor_drill}"
DRILL_USER="${DRILL_USER:-drilluser}"
DRILL_PASS="${DRILL_PASS:-drillpass}"
DRILL_IMAGE="${DRILL_IMAGE:-pgvector/pgvector:pg16}"
DRILL_NAME="pdfeditor-restore-drill-$$"

# ---- Pick the dump ---------------------------------------------------------

if [[ $# -ge 1 ]]; then
    DUMP="$1"
else
    DUMP="$(ls -1t "$BACKUP_DIR"/*.sql.gz 2>/dev/null | head -n1 || true)"
fi

if [[ -z "${DUMP:-}" || ! -f "$DUMP" ]]; then
    echo "✗ no DB dump found (tried: ${1:-$BACKUP_DIR/*.sql.gz})" >&2
    exit 1
fi

DUMP_SIZE_HUMAN="$(du -h "$DUMP" | cut -f1)"
echo "→ Drill against: $DUMP ($DUMP_SIZE_HUMAN)"

# ---- Bring up an ephemeral Postgres ---------------------------------------

cleanup() {
    echo "→ Cleaning up drill container…"
    docker rm -f "$DRILL_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

T_START=$(date +%s)

echo "→ Starting ephemeral $DRILL_IMAGE on :$DRILL_PORT…"
docker run -d --rm \
    --name "$DRILL_NAME" \
    -e POSTGRES_DB="$DRILL_DB" \
    -e POSTGRES_USER="$DRILL_USER" \
    -e POSTGRES_PASSWORD="$DRILL_PASS" \
    -p "127.0.0.1:${DRILL_PORT}:5432" \
    "$DRILL_IMAGE" >/dev/null

echo -n "→ Waiting for Postgres to accept connections… "
for _ in $(seq 1 60); do
    if docker exec "$DRILL_NAME" pg_isready -U "$DRILL_USER" -d "$DRILL_DB" >/dev/null 2>&1; then
        echo "ready."
        break
    fi
    sleep 1
done
if ! docker exec "$DRILL_NAME" pg_isready -U "$DRILL_USER" -d "$DRILL_DB" >/dev/null 2>&1; then
    echo "✗ Postgres never came up." >&2
    exit 1
fi

T_PG_READY=$(date +%s)

# ---- Restore ---------------------------------------------------------------

echo "→ Restoring dump (parallel, custom format)…"
# pg_restore reads the custom-format dump produced by backup_db.sh.
# Parallel restore (--jobs) requires SEEKABLE input — pipes from stdin
# don't qualify. So gunzip + copy the dump into the container first,
# then run pg_restore against the file path.
#
# --no-owner / --no-privileges so the restore user owns everything in
# the drill DB, regardless of what role names were in the source dump.
TMP_DUMP="/tmp/restore-$$.dump"
gunzip -c "$DUMP" | docker exec -i "$DRILL_NAME" sh -c "cat > $TMP_DUMP"
docker exec \
    -e PGPASSWORD="$DRILL_PASS" \
    "$DRILL_NAME" \
    pg_restore --username="$DRILL_USER" --dbname="$DRILL_DB" \
               --no-owner --no-privileges \
               --jobs=2 \
               "$TMP_DUMP"
docker exec "$DRILL_NAME" rm -f "$TMP_DUMP"

T_RESTORE=$(date +%s)

# ---- Sanity SQL ------------------------------------------------------------

echo "→ Sanity checks…"
psql_drill() {
    docker exec -e PGPASSWORD="$DRILL_PASS" "$DRILL_NAME" \
        psql --username="$DRILL_USER" --dbname="$DRILL_DB" -tAX -c "$1"
}

EXT_VECTOR=$(psql_drill "SELECT extname FROM pg_extension WHERE extname='vector';")
if [[ "$EXT_VECTOR" != "vector" ]]; then
    echo "✗ pgvector extension missing in restored DB" >&2
    exit 1
fi
echo "  ✓ pgvector extension present"

# Tables we expect to exist + a row-count sanity floor. The floor is 0
# for legitimately-empty tables (e.g. AuditLog in a quiet week) but the
# table must exist; missing tables would mean a partial restore.
declare -A FLOORS=(
    [pdfeditor_uploadedpdf]=0
    [pdfeditor_processedpdf]=0
    [pdfeditor_job]=0
    [pdfeditor_embedding]=0
    [pdfeditor_apikey]=0
    [pdfeditor_sharelink]=0
    [auth_user]=1
    [django_migrations]=10
)

FAILED=0
for TABLE in "${!FLOORS[@]}"; do
    FLOOR="${FLOORS[$TABLE]}"
    N=$(psql_drill "SELECT count(*) FROM ${TABLE};" 2>/dev/null || echo "ERR")
    if [[ "$N" == "ERR" ]]; then
        echo "  ✗ ${TABLE}: missing"
        FAILED=1
    elif (( N < FLOOR )); then
        echo "  ✗ ${TABLE}: ${N} rows (floor ${FLOOR})"
        FAILED=1
    else
        printf "  ✓ %-30s %s rows\n" "$TABLE" "$N"
    fi
done

# Foreign-key integrity. If pg_dump captured a state mid-cascade or the
# restore reordered something, FK constraints will still be enforced
# implicitly by pg_restore — but an explicit query is cheap and proves
# we can read across joins, not just count rows in isolation.
ORPHAN_JOBS=$(psql_drill "
    SELECT count(*) FROM pdfeditor_job j
    LEFT JOIN auth_user u ON u.id = j.user_id
    WHERE j.user_id IS NOT NULL AND u.id IS NULL;
")
if (( ORPHAN_JOBS > 0 )); then
    echo "  ✗ ${ORPHAN_JOBS} jobs reference missing users"
    FAILED=1
else
    echo "  ✓ no orphan jobs"
fi

T_END=$(date +%s)

# ---- Report ----------------------------------------------------------------

PG_READY=$((T_PG_READY - T_START))
RESTORE=$((T_RESTORE  - T_PG_READY))
SANITY=$((T_END       - T_RESTORE))
TOTAL=$((T_END        - T_START))

echo
echo "── RTO breakdown ─────────────────────────"
printf "  container up + ready : %4ds\n" "$PG_READY"
printf "  pg_restore           : %4ds\n" "$RESTORE"
printf "  sanity checks        : %4ds\n" "$SANITY"
printf "  TOTAL                : %4ds\n" "$TOTAL"
echo "──────────────────────────────────────────"

if (( FAILED )); then
    echo "✗ Drill FAILED — see lines marked ✗ above." >&2
    exit 2
fi

echo "✓ Drill PASSED. Backup is restorable."
