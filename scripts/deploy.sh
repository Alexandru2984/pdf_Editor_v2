#!/usr/bin/env bash
# Verify and pull an immutable, signed GHCR image, promote it locally as
# `pdfeditor:latest`, and roll the application services forward.
#
# Triggered from CI via SSH. Idempotent. Safe to re-run.
#
# Usage:
#   scripts/deploy.sh <image-tag> <sha256-digest> <full-commit-sha>
#
# Example:
#   scripts/deploy.sh sha-abc1234 sha256:0123...cdef 0123...cdef
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <image-tag> <sha256-digest> <full-commit-sha>" >&2
    exit 1
fi

TAG="$1"
EXPECTED_DIGEST="$2"
EXPECTED_COMMIT="$3"
REGISTRY_IMAGE="ghcr.io/alexandru2984/pdf_editor_v2"
LOCAL_IMAGE="pdfeditor:latest"
ROLLBACK_IMAGE="pdfeditor:rollback"
COSIGN_IDENTITY="https://github.com/Alexandru2984/pdf_Editor_v2/.github/workflows/deploy.yml@refs/heads/main"
COSIGN_ISSUER="https://token.actions.githubusercontent.com"
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
SMOKE_URL="${SMOKE_URL:-https://pdf.micutu.com/readyz}"

if [[ ! "$TAG" =~ ^sha-[0-9a-f]{7,40}$ ]]; then
    echo "✗ invalid immutable image tag: $TAG" >&2
    exit 1
fi
if [[ ! "$EXPECTED_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "✗ invalid image digest" >&2
    exit 1
fi
if [[ ! "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "✗ invalid commit SHA" >&2
    exit 1
fi
if [[ "$TAG" != "sha-${EXPECTED_COMMIT:0:7}" ]]; then
    echo "✗ image tag does not match the tested commit" >&2
    exit 1
fi

for required_command in git docker cosign curl date grep runsc; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        echo "✗ required command is missing: $required_command" >&2
        exit 1
    fi
done

cd "$(dirname "$0")/.."

# Pull the latest compose file + scripts from git so structural changes
# (new services, volume mounts, image upgrades) actually land. The Docker
# image itself is pulled from GHCR below; only the orchestration glue
# comes from git.
#
# Include untracked files: a deploy must never overwrite or silently ignore a
# local operator change. Update only by fast-forward, then re-exec the script
# from the exact commit whose image passed CI.
if [[ "${PDFEDITOR_DEPLOY_REEXEC:-0}" != "1" ]]; then
    echo "→ Fast-forwarding repo to the tested commit…"
    if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
        echo "  ⚠ working tree is not clean — refusing to deploy." >&2
        git status --short >&2
        exit 1
    fi
    git fetch --depth=100 origin main
    if [[ "$(git rev-parse FETCH_HEAD)" != "$EXPECTED_COMMIT" ]]; then
        echo "✗ origin/main does not match the tested commit" >&2
        exit 1
    fi
    git merge --ff-only FETCH_HEAD
    PDFEDITOR_DEPLOY_REEXEC=1 exec "$SCRIPT_PATH" "$TAG" "$EXPECTED_DIGEST" "$EXPECTED_COMMIT"
fi

if [[ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
    echo "✗ checked-out source does not match the tested commit" >&2
    exit 1
fi

# Native PDF parsers run in both web and worker. Refuse to roll production
# onto an absent or stale sandbox runtime; Ubuntu's former 2024 package was
# more than two years behind the supported release channel.
RUNSC_MIN_RELEASE=20260803
RUNSC_MAX_AGE_DAYS="${RUNSC_MAX_AGE_DAYS:-180}"
RUNSC_VERSION_OUTPUT="$(runsc --version)"
if [[ ! "$RUNSC_VERSION_OUTPUT" =~ release-([0-9]{8}) ]]; then
    echo "✗ runsc is not an official dated release; update gVisor first" >&2
    exit 1
fi
RUNSC_RELEASE="${BASH_REMATCH[1]}"
if (( 10#$RUNSC_RELEASE < 10#$RUNSC_MIN_RELEASE )); then
    echo "✗ runsc release ${RUNSC_RELEASE} is older than required ${RUNSC_MIN_RELEASE}" >&2
    exit 1
fi
RUNSC_RELEASE_ISO="${RUNSC_RELEASE:0:4}-${RUNSC_RELEASE:4:2}-${RUNSC_RELEASE:6:2}"
RUNSC_RELEASE_EPOCH="$(date -u -d "$RUNSC_RELEASE_ISO" +%s)"
RUNSC_AGE_DAYS=$(( ($(date -u +%s) - RUNSC_RELEASE_EPOCH) / 86400 ))
if (( RUNSC_AGE_DAYS < 0 || RUNSC_AGE_DAYS > RUNSC_MAX_AGE_DAYS )); then
    echo "✗ runsc release age is ${RUNSC_AGE_DAYS} days; maximum is ${RUNSC_MAX_AGE_DAYS}" >&2
    exit 1
fi
if ! docker info --format '{{json .Runtimes}}' | grep -q '"runsc-net"'; then
    echo "✗ Docker runtime runsc-net is not configured" >&2
    exit 1
fi

TARGET_REF="${REGISTRY_IMAGE}@${EXPECTED_DIGEST}"
echo "→ Verifying Sigstore identity for ${TARGET_REF}…"
cosign verify \
    --certificate-identity "$COSIGN_IDENTITY" \
    --certificate-oidc-issuer "$COSIGN_ISSUER" \
    "$TARGET_REF" >/dev/null

echo "→ Pulling immutable digest ${TARGET_REF}…"
docker pull "$TARGET_REF"
if ! docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$TARGET_REF" \
    | grep -Fxq "$TARGET_REF"; then
    echo "✗ pulled image does not expose the expected repository digest" >&2
    exit 1
fi
TARGET_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$TARGET_REF")"

if ! docker image inspect "$LOCAL_IMAGE" >/dev/null 2>&1; then
    echo "✗ no current ${LOCAL_IMAGE}; refusing a production deploy without rollback" >&2
    exit 1
fi

ROLLBACK_ARMED=0
# Invoked indirectly by the EXIT trap below.
# shellcheck disable=SC2329
rollback_on_failure() {
    status=$?
    trap - EXIT
    if (( status == 0 || ROLLBACK_ARMED == 0 )); then
        exit "$status"
    fi

    set +e
    echo "✗ Deploy failed; automatically restoring ${ROLLBACK_IMAGE}…" >&2
    docker tag "$ROLLBACK_IMAGE" "$LOCAL_IMAGE"
    docker compose up -d --no-build --force-recreate migrate web worker nginx >&2
    docker compose up -d --no-build >&2

    rollback_healthy=0
    for _ in $(seq 1 18); do
        if curl -fsS --max-time 5 -H 'Cache-Control: no-cache' "$SMOKE_URL" >/dev/null 2>&1; then
            rollback_healthy=1
            break
        fi
        sleep 5
    done
    if (( rollback_healthy )); then
        echo "✓ Automatic rollback is healthy; deploy remains failed for investigation." >&2
    else
        echo "✗ Automatic rollback also failed; immediate operator action required." >&2
        docker compose ps >&2
        docker compose logs --tail=50 web worker nginx >&2
    fi
    exit "$status"
}
trap rollback_on_failure EXIT

echo "→ Saving current image as ${ROLLBACK_IMAGE}…"
docker tag "$LOCAL_IMAGE" "$ROLLBACK_IMAGE"
ROLLBACK_ARMED=1

echo "→ Promoting verified digest as ${LOCAL_IMAGE}…"
docker tag "$TARGET_REF" "$LOCAL_IMAGE"
docker tag "$TARGET_REF" "pdfeditor:${TAG}"

echo "→ Recreating application services with the verified image…"
docker compose up -d --no-build --force-recreate migrate web worker nginx
# Apply non-application compose changes/new services without forcing stateful
# services to restart unnecessarily.
docker compose up -d --no-build

mapfile -t application_containers < <(docker compose ps -q web worker)
if (( ${#application_containers[@]} < 3 )); then
    echo "✗ expected two web replicas and one worker" >&2
    exit 1
fi
for container_id in "${application_containers[@]}"; do
    running_image_id="$(docker inspect --format '{{.Image}}' "$container_id")"
    if [[ "$running_image_id" != "$TARGET_IMAGE_ID" ]]; then
        echo "✗ container $container_id is not running the verified image" >&2
        exit 1
    fi
done

# Smoke test through the real public path (Cloudflare → host nginx →
# internal LB → replica → DB). /readyz does a SELECT 1, so a deploy that
# boots gunicorn but can't reach Postgres still fails here — and a
# non-zero exit makes the CI deploy job go red instead of lying green.
echo "→ Smoke test: ${SMOKE_URL}…"
for _ in $(seq 1 18); do
    if curl -fsS --max-time 5 -H 'Cache-Control: no-cache' "$SMOKE_URL" >/dev/null 2>&1; then
        ROLLBACK_ARMED=0
        echo "✓ ${SMOKE_URL} healthy."
        echo "✓ Deployed ${TARGET_REF} from ${EXPECTED_COMMIT}"
        docker compose ps
        exit 0
    fi
    sleep 5
done

echo "✗ ${SMOKE_URL} still failing after ~90s — deploy is NOT healthy." >&2
echo "  Recent logs:" >&2
docker compose logs --tail=50 web worker nginx >&2 || true
exit 1
