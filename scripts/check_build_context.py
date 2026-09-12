#!/usr/bin/env python3
"""Prove Docker filters private fixtures and copies only runtime inputs.

Uses a synthetic context and scratch images: no production files, secrets,
services, package installation or network access are involved.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRIVATE_FIXTURES = (
    ".env",
    ".env.local",
    ".env.bak-mailcow-test",
    ".env.production",
    "nested/.env.backup",
    "config.bak",
    "nested/config.bak-old",
    ".claude/settings.json",
    ".codex/session.json",
    "media/uploads/customer.pdf",
    "media/thumbs/preview.png",
    "media/future-category/customer.bin",
    "db.sqlite3",
    "server.key",
    "signing.p12",
    "pdfeditor/local/private.pem",
)
RUNTIME_FIXTURES = (
    "manage.py",
    "requirements.txt",
    "requirements.lock",
    "pdf_project/settings.py",
    "pdfeditor/templates/example.html",
    "pdfeditor/migrations/__init__.py",
    "static/js/example.js",
    "locale/en/LC_MESSAGES/django.mo",
)


def main() -> None:
    copy_lines = [line for line in (ROOT / "Dockerfile").read_text().splitlines() if line.startswith("COPY ")]
    with tempfile.TemporaryDirectory(prefix="pdfeditor-context-check-") as directory:
        root = Path(directory)
        context = root / "context"
        context.mkdir()
        shutil.copyfile(ROOT / ".dockerignore", context / ".dockerignore")
        for name in (*PRIVATE_FIXTURES, *RUNTIME_FIXTURES, "unrelated/operator-notes.txt"):
            path = context / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic build-context fixture\n")
        (context / "Dockerfile").write_text(
            "FROM scratch AS filtered\nCOPY . /context/\n"
            "FROM scratch AS runtime\nWORKDIR /app\n" + "\n".join(copy_lines) + "\n"
        )
        for target in ("filtered", "runtime"):
            output = root / target
            subprocess.run(
                [
                    "docker",
                    "build",
                    "--network=none",
                    "--progress=quiet",
                    "--target",
                    target,
                    "--output",
                    f"type=local,dest={output}",
                    str(context),
                ],
                check=True,
                timeout=120,
            )
            exported = output / ("context" if target == "filtered" else "app")
            for name in PRIVATE_FIXTURES:
                if (exported / name).exists():
                    raise SystemExit(f"FAIL: {target} includes private fixture {name}")
            for name in RUNTIME_FIXTURES:
                if not (exported / name).is_file():
                    raise SystemExit(f"FAIL: {target} omitted runtime fixture {name}")
            if target == "runtime" and (exported / "unrelated").exists():
                raise SystemExit("FAIL: runtime COPY includes unlisted local files")
    print("Docker context and runtime COPY isolation passed (synthetic fixtures).")


if __name__ == "__main__":
    main()
