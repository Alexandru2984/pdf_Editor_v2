#!/usr/bin/env bash
# Pull a prebuilt image from GHCR, retag it locally as `pdfeditor:latest`
# so docker-compose picks it up, and restart the web container.
#
# Triggered from CI via SSH. Idempotent. Safe to re-run.
#
# Usage:
#   scripts/deploy.sh <image-tag>
#
# Example:
#   scripts/deploy.sh sha-abc1234
#   scripts/deploy.sh latest
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <image-tag>" >&2
    exit 1
fi

TAG="$1"
REGISTRY_IMAGE="ghcr.io/alexandru2984/pdf_editor_v2"
LOCAL_IMAGE="pdfeditor:latest"

cd "$(dirname "$0")/.."

echo "→ Saving current latest as rollback…"
if docker image inspect "$LOCAL_IMAGE" >/dev/null 2>&1; then
    docker tag "$LOCAL_IMAGE" pdfeditor:rollback
fi

echo "→ Pulling ${REGISTRY_IMAGE}:${TAG}…"
docker pull "${REGISTRY_IMAGE}:${TAG}"

echo "→ Retagging as ${LOCAL_IMAGE}…"
docker tag "${REGISTRY_IMAGE}:${TAG}" "$LOCAL_IMAGE"

echo "→ Restarting web container…"
docker compose up -d --no-build web

echo "→ Waiting for healthy startup…"
sleep 8
docker compose ps web

echo "✓ Deployed ${REGISTRY_IMAGE}:${TAG}"
