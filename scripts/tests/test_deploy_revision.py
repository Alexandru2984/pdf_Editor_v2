"""A signed image from a different revision must never be promoted."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "abcdef1" + "0" * 33
DIGEST = "sha256:" + "a" * 64


class DeploymentRevisionTests(unittest.TestCase):
    def test_wrong_or_missing_image_revision_is_rejected_before_promotion(self):
        with tempfile.TemporaryDirectory(prefix="pdfeditor-deploy-revision-") as directory:
            root = Path(directory)
            mock = root / "mock.py"
            mock.write_text(
                f"#!{sys.executable}\n"
                "import json,os,sys\n"
                "from pathlib import Path\n"
                "name=Path(sys.argv[0]).name\n"
                "args=sys.argv[1:]\n"
                "with open(os.environ['MOCK_LOG'],'a') as log: log.write(json.dumps([name,*args])+'\\n')\n"
                "if name=='git': print(os.environ['MOCK_COMMIT'])\n"
                "elif name=='runsc': print('runsc version release-'+os.environ['MOCK_RELEASE']+'.0')\n"
                "elif name=='docker' and args[0]=='info': print('{\"runsc-net\":{}}')\n"
                "elif name=='docker' and '--format' in args:\n"
                "    fmt=args[args.index('--format')+1]\n"
                "    if 'RepoDigests' in fmt: print('ghcr.io/alexandru2984/pdf_editor_v2@'+os.environ['MOCK_DIGEST'])\n"
                "    elif 'revision' in fmt: print(os.environ['MOCK_REVISION'])\n"
                "    else: print('sha256:'+'b'*64)\n"
            )
            mock.chmod(0o755)
            for name in ("git", "docker", "cosign", "curl", "runsc"):
                (root / name).symlink_to(mock)
            for revision in ("f" * 40, "<no value>"):
                with self.subTest(revision=revision):
                    log = root / "commands.jsonl"
                    log.write_text("")
                    env = {
                        **os.environ,
                        "PATH": str(root) + os.pathsep + os.environ["PATH"],
                        "PDFEDITOR_DEPLOY_REEXEC": "1",
                        "MOCK_LOG": str(log),
                        "MOCK_COMMIT": COMMIT,
                        "MOCK_DIGEST": DIGEST,
                        "MOCK_REVISION": revision,
                        "MOCK_RELEASE": datetime.now(timezone.utc).strftime("%Y%m%d"),
                    }
                    result = subprocess.run(
                        ["bash", str(ROOT / "scripts/deploy.sh"), "sha-abcdef1", DIGEST, COMMIT],
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=20,
                    )
                    commands = [json.loads(line) for line in log.read_text().splitlines()]
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("source revision does not match", result.stderr)
                    self.assertTrue(any(command[0] == "cosign" for command in commands))
                    self.assertFalse(any(command[:2] == ["docker", "tag"] for command in commands))
                    self.assertFalse(any(command[:2] == ["docker", "compose"] for command in commands))


if __name__ == "__main__":
    unittest.main()
