# Auto-deploy setup

The `build-and-deploy` workflow runs on every push to `main`:

1. **Waits** for the `tests` workflow to pass on the same commit.
2. **Builds** a Docker image from the repo and pushes it to
   `ghcr.io/<your-github-user>/pdf_editor_v2` tagged with both `latest` and
   `sha-<short-sha>` (so you can pin to a specific build).
3. **SSHes** into the VPS using a dedicated forced-command key and a pinned
   server fingerprint. `scripts/deploy_from_ci.py` parses the release JSON and
   runs `scripts/deploy.sh <tag> <digest> <commit>`.
   The script verifies the exact Sigstore identity and immutable digest, saves
   `pdfeditor:rollback`, verifies the image's OCI source revision against the
   tested commit, recreates the app services, verifies their image IDs,
   and automatically rolls back if the public readiness check fails.

## One-time setup

### On GitHub

In **Settings → Secrets and variables → Actions → New repository secret**, add:

| Secret | Value | Notes |
|---|---|---|
| `DEPLOY_HOST` | Direct SSH hostname or public IP of the current VPS | A Cloudflare-proxied website hostname is not an SSH endpoint. |
| `DEPLOY_USER` | SSH user on the VPS | the user that runs `docker compose` |
| `DEPLOY_PORT` | SSH port | optional; defaults to `22` |
| `DEPLOY_SSH_KEY` | Private SSH key (the **full** PEM, including header/footer) | matched to a public key in `~/.ssh/authorized_keys` on the VPS |
| `DEPLOY_FINGERPRINT` | SHA-256 fingerprint of the server host key negotiated by the SSH client | Verify it against the host's public key locally on the VPS. |

The deploy job uses its short-lived `GITHUB_TOKEN` with `packages: read` to pull
the package belonging to this repository. `DEPLOY_PATH`, `GHCR_PULL_USER` and
`GHCR_PULL_TOKEN` are no longer read by the workflow. The forced command fixes
the repository path on the server; the JSON cannot select a different path,
registry, shell command or verification policy.

### On the VPS

1. Generate a dedicated deploy key in a private temporary directory:
   ```bash
   setup_dir="$(mktemp -d)"
   ssh-keygen -t ed25519 -f "$setup_dir/deploy" -N "" -C pdfeditor-github-deploy
   printf 'restrict,command="/home/micu/pdf_Editor_v2/scripts/deploy_from_ci.py" %s\n' \
     "$(cat "$setup_dir/deploy.pub")" >> ~/.ssh/authorized_keys
   chmod 600 ~/.ssh/authorized_keys
   gh secret set DEPLOY_SSH_KEY --repo Alexandru2984/pdf_Editor_v2 < "$setup_dir/deploy"
   ```
   Keep existing operator keys. Verify the new key, then remove its temporary
   private-key copy. `restrict` disables forwarding, agent forwarding, X11 and
   PTY access. The forced command accepts a readiness probe or validated release
   fields; arbitrary commands such as `id` are rejected.

2. Make sure the `docker` group includes the deploy user and that
   `docker compose ps` works without `sudo` for that user.

3. Ensure the GHCR package grants this repository Actions access. The SSH entry
   point sends the job token to `docker login` through stdin. Its temporary
   `DOCKER_CONFIG` is private and removed on success or failure; the token is
   not retained in the user's persistent Docker config. No long-lived pull PAT
   is required.

4. Install cosign from the official release/distribution and verify its
   checksum or signature. Keep it and the configured `runsc-net` runtime current:
   ```bash
   cosign version
   runsc --version
   ```

5. Test SSH with the pinned host key and the JSON probe:
   ```bash
   ssh -T -i "$setup_dir/deploy" -o IdentitiesOnly=yes \
     "$DEPLOY_USER@$DEPLOY_HOST" '{"command":"probe"}'
   ```
   Expected response: `PDF Editor deployment SSH ready.` Use a known_hosts file
   populated from the VPS's verified public host key. Do not disable host-key
   checking. Also verify that a non-JSON command is rejected.

After changing VPS, update the host, user, port, private deploy key and server
fingerprint together. Confirm these before triggering the first production run.

## Rollback

Each deploy tags the previous image as `pdfeditor:rollback`. To revert:

```bash
docker tag pdfeditor:rollback pdfeditor:latest
docker compose up -d --no-build --force-recreate migrate web worker nginx
```

Or re-deploy an already-tested image using the tag, immutable digest, and full
Git commit shown in the successful Actions run:

```bash
./scripts/deploy.sh sha-abc1234 sha256:<64-hex-digest> <40-hex-commit>
```

(Images are kept in GHCR for as long as you configure retention in
**Packages → settings**.)

There is intentionally no workflow-dispatch bypass: production images are
built and deployed only after the `tests` workflow succeeds.

## Deployment-key incident checklist

The deploy account can operate Docker and therefore remains privileged. The
dedicated forced command limits this key to the validated release protocol;
the existing operator keys retain their separate access. The short-lived
registry token can read packages and is removed locally when the command exits.

If the deploy key is lost or suspected exposed, remove only its public-key
entry from `authorized_keys`, generate a replacement and update
`DEPLOY_SSH_KEY`. Verify a probe and a rejected arbitrary command before the
next release. If the VPS changes, replace the host fingerprint from the new
server's verified public key; the workflow fails closed if it is absent or
malformed. Existing rollback and readiness checks still apply.

The migration checks cover key authentication, an exact Actions-client probe,
host-key verification and rejection of shell commands. They do not constitute
a disaster-recovery or incident-response certification.

The standard-library tests in `scripts/tests/` also cover malformed release
requests, secret handling, registry-authentication failure, credential cleanup
and rejection of a signed image with a different or missing source revision.
