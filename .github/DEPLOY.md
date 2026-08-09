# Auto-deploy setup

The `build-and-deploy` workflow runs on every push to `main`:

1. **Waits** for the `tests` workflow to pass on the same commit.
2. **Builds** a Docker image from the repo and pushes it to
   `ghcr.io/<your-github-user>/pdf_editor_v2` tagged with both `latest` and
   `sha-<short-sha>` (so you can pin to a specific build).
3. **SSHes** into the VPS and runs `scripts/deploy.sh <tag> <digest> <commit>`.
   The script verifies the exact Sigstore identity and immutable digest, saves
   `pdfeditor:rollback`, recreates the app services, verifies their image IDs,
   and automatically rolls back if the public readiness check fails.

## One-time setup

### On GitHub

In **Settings → Secrets and variables → Actions → New repository secret**, add:

| Secret | Value | Notes |
|---|---|---|
| `DEPLOY_HOST` | VPS hostname or IP | e.g. `pdf.micutu.com` |
| `DEPLOY_USER` | SSH user on the VPS | the user that runs `docker compose` |
| `DEPLOY_PORT` | SSH port | optional; defaults to `22` |
| `DEPLOY_PATH` | Absolute path to the repo on the VPS | e.g. `/home/micu/pdf_Editor_v2` |
| `DEPLOY_SSH_KEY` | Private SSH key (the **full** PEM, including header/footer) | matched to a public key in `~/.ssh/authorized_keys` on the VPS |
| `GHCR_PULL_USER` | Your GitHub username | for `docker login ghcr.io` on the VPS |
| `GHCR_PULL_TOKEN` | A GitHub PAT with `read:packages` scope | classic token, generate at <https://github.com/settings/tokens> |

### On the VPS

1. Generate a key pair for the deploy user (recommend a fresh one, not your dev key):
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/github_deploy -N ""
   cat ~/.ssh/github_deploy.pub >> ~/.ssh/authorized_keys
   ```
   Copy the **private** key (`~/.ssh/github_deploy`) into the `DEPLOY_SSH_KEY` secret.

2. Make sure the `docker` group includes the deploy user and that
   `docker compose ps` works without `sudo` for that user.

3. The first time the workflow runs, the VPS needs to be able to pull
   from `ghcr.io`. The workflow does `docker login` for you using
   `GHCR_PULL_TOKEN`, but ensure the package is **public** OR the token's
   user has access to it.

4. Install cosign from the signed Ubuntu repository and keep it patched:
   ```bash
   sudo apt-get install cosign
   cosign version
   ```

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
