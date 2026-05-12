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

# Pull the latest compose file + scripts from git so structural changes
# (new services, volume mounts, image upgrades) actually land. The Docker
# image itself is pulled from GHCR below; only the orchestration glue
# comes from git.
#
# If the working tree has uncommitted local edits, refuse to nuke them.
# This is the safer default for a VPS where the maintainer occasionally
# edits files in-place — losing those silently has burned us before.
echo "→ Pulling repo changes from origin/main…"
if ! git diff --quiet HEAD 2>/dev/null; then
    echo "  ⚠ working tree has uncommitted changes — refusing to reset."
    echo "  Commit or stash them, then re-run scripts/deploy.sh."
    git status --short
    exit 1
fi
git fetch --depth=1 origin main
git reset --hard origin/main

echo "→ Saving current latest as rollback…"
if docker image inspect "$LOCAL_IMAGE" >/dev/null 2>&1; then
    docker tag "$LOCAL_IMAGE" pdfeditor:rollback
fi

echo "→ Pulling ${REGISTRY_IMAGE}:${TAG}…"
docker pull "${REGISTRY_IMAGE}:${TAG}"

echo "→ Retagging as ${LOCAL_IMAGE}…"
docker tag "${REGISTRY_IMAGE}:${TAG}" "$LOCAL_IMAGE"

echo "→ Bringing up all services (picks up compose changes)…"
docker compose up -d --no-build

echo "→ Waiting for healthy startup…"
sleep 8
docker compose ps

echo "✓ Deployed ${REGISTRY_IMAGE}:${TAG}"
