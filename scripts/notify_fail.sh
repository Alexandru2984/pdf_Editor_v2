#!/usr/bin/env bash
# Telegram alert for failed pdfeditor cron jobs.
#
# Usage: notify_fail.sh "<job name>"
#
# Reads BOT_TOKEN / CHAT_ID from /root/.backup_vps_env — the same
# credentials the weekly full-VPS backup uses — so it must run as root,
# which is fine: the callers are root cron lines shaped like
#
#   cd /home/micu/pdf_Editor_v2 && <job> >> /var/log/<job>.log 2>&1 \
#       || /home/micu/pdf_Editor_v2/scripts/notify_fail.sh "<job name>"
#
# Always exits 0, even when Telegram is unreachable or the env file is
# missing: alerting must never turn one logged failure into two.
set -u

JOB="${1:-unknown job}"
ENV_FILE="/root/.backup_vps_env"

# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && . "$ENV_FILE"
: "${BOT_TOKEN:=}"
: "${CHAT_ID:=}"
[ -n "$BOT_TOKEN" ] && [ -n "$CHAT_ID" ] || exit 0

HOST="$(hostname -s)"
# Plain text on purpose — no parse_mode, so job names never need escaping.
curl -s --max-time 10 -X POST \
    "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
    -d chat_id="${CHAT_ID}" \
    -d text="❌ pdfeditor: ${JOB} failed on ${HOST} — check /var/log/pdfeditor-*.log" \
    > /dev/null 2>&1 || true

exit 0
