#!/usr/bin/env python3
"""Restricted SSH entry point for this repository's signed production releases.

Install as an authorized_keys forced command with the ``restrict`` option.
SSH_ORIGINAL_COMMAND is data: parse JSON, never evaluate it as shell code.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DOCKER_BINARY = "/usr/bin/docker"
_FIELDS = frozenset({"image_tag", "image_digest", "commit_sha", "registry_username", "registry_token"})


@dataclass(frozen=True)
class Deployment:
    image_tag: str
    image_digest: str
    commit_sha: str
    registry_username: str
    registry_token: str = field(repr=False)


def parse_command(raw: str) -> Deployment | None:
    if not raw or len(raw) > 8192:
        raise ValueError("Invalid deployment request.")
    payload = json.loads(raw)
    if payload == {"command": "probe"}:
        return None
    if not isinstance(payload, dict) or payload.keys() != _FIELDS:
        raise ValueError("Invalid deployment fields.")
    if not all(isinstance(value, str) for value in payload.values()):
        raise ValueError("Deployment fields must be strings.")
    deployment = Deployment(**payload)
    if not re.fullmatch(r"[0-9a-f]{40}", deployment.commit_sha):
        raise ValueError("Invalid source revision.")
    if deployment.image_tag != f"sha-{deployment.commit_sha[:7]}":
        raise ValueError("Image tag does not match source revision.")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", deployment.image_digest):
        raise ValueError("Invalid image digest.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\[\]-]{0,99}", deployment.registry_username):
        raise ValueError("Invalid registry username.")
    token = deployment.registry_token
    if not 1 <= len(token) <= 4096 or not all(33 <= ord(char) <= 126 for char in token):
        raise ValueError("Invalid registry credential.")
    return deployment


def main() -> int:
    raw = os.environ.pop("SSH_ORIGINAL_COMMAND", "")
    try:
        deployment = parse_command(raw)
    except (TypeError, ValueError):
        print("Deployment request rejected.", file=sys.stderr)
        return 2
    if deployment is None:
        print("PDF Editor deployment SSH ready.")
        return 0

    # Registry credentials live only for this invocation. The token is sent
    # over stdin and never included in child argv, logs or environment.
    with tempfile.TemporaryDirectory(prefix="pdfeditor-ci-registry-") as config:
        env = {**os.environ, "DOCKER_CONFIG": config}
        try:
            subprocess.run(
                [
                    DOCKER_BINARY,
                    "login",
                    "ghcr.io",
                    "--username",
                    deployment.registry_username,
                    "--password-stdin",
                ],
                input=deployment.registry_token + "\n",
                text=True,
                capture_output=True,
                timeout=60,
                env=env,
                check=True,
            )
            return subprocess.run(
                [
                    str(REPOSITORY_ROOT / "scripts/deploy.sh"),
                    deployment.image_tag,
                    deployment.image_digest,
                    deployment.commit_sha,
                ],
                cwd=REPOSITORY_ROOT,
                env=env,
                check=False,
            ).returncode
        except (OSError, subprocess.SubprocessError):
            print("Registry authentication or deployment could not be started.", file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
