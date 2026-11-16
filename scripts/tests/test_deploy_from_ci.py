"""Exercise the SSH command boundary without registry or production access."""

import contextlib
import io
import json
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import deploy_from_ci


class DeploymentEntryPointTests(unittest.TestCase):
    def payload(self, **changes):
        value = {
            "image_tag": "sha-abcdef1",
            "image_digest": "sha256:" + "a" * 64,
            "commit_sha": "abcdef1" + "0" * 33,
            "registry_username": "release-fixture",
            "registry_token": "fixture-token-not-a-real-credential",
        }
        value.update(changes)
        return value

    def test_probe_authenticates_without_executing_commands(self):
        with (
            patch.dict(os.environ, {"SSH_ORIGINAL_COMMAND": '{"command":"probe"}'}),
            patch.object(deploy_from_ci.subprocess, "run") as run,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(deploy_from_ci.main(), 0)
        run.assert_not_called()
        self.assertIn("deployment SSH ready", output.getvalue())

    def test_shell_commands_and_nonobject_json_are_rejected(self):
        for raw in ("id", "set -e; docker ps", "$(id)", "[]", "null", '"shell"', "x" * 8193):
            with self.subTest(raw=raw[:20]), self.assertRaises(ValueError):
                deploy_from_ci.parse_command(raw)

    def test_unapproved_fields_and_mismatched_artifacts_are_rejected(self):
        for change in (
            {"command": "id"},
            {"image_tag": "latest"},
            {"image_tag": "sha-0000000"},
            {"commit_sha": "a" * 39},
            {"image_digest": "sha256:" + "g" * 64},
            {"registry_username": "--config=/tmp/untrusted"},
            {"registry_username": "--debug"},
            {"registry_token": "token\nextra-command"},
            {"registry_token": "x" * 4097},
            {"registry_token": None},
        ):
            with self.subTest(fields=list(change)), self.assertRaises(ValueError):
                deploy_from_ci.parse_command(json.dumps(self.payload(**change)))

    def test_secret_is_omitted_from_representation(self):
        value = deploy_from_ci.parse_command(json.dumps(self.payload()))
        self.assertNotIn(self.payload()["registry_token"], repr(value))

    def test_valid_release_uses_stdin_and_removes_temporary_credentials(self):
        calls = []

        def run(command, **kwargs):
            calls.append((command, kwargs))
            config = Path(kwargs["env"]["DOCKER_CONFIG"])
            self.assertTrue(config.is_dir())
            self.assertEqual(config.stat().st_mode & 0o777, 0o700)
            self.assertNotIn("SSH_ORIGINAL_COMMAND", kwargs["env"])
            self.assertNotIn(self.payload()["registry_token"], repr(command))
            self.assertNotIn(self.payload()["registry_token"], repr(kwargs["env"]))
            if command[0] == deploy_from_ci.DOCKER_BINARY:
                self.assertEqual(kwargs["input"], self.payload()["registry_token"] + "\n")
                (config / "config.json").write_text("synthetic registry configuration")
            return subprocess.CompletedProcess(command, 0)

        with (
            patch.dict(os.environ, {"SSH_ORIGINAL_COMMAND": json.dumps(self.payload())}),
            patch.object(deploy_from_ci.subprocess, "run", side_effect=run),
        ):
            self.assertEqual(deploy_from_ci.main(), 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            calls[1][0][1:], [self.payload()[name] for name in ("image_tag", "image_digest", "commit_sha")]
        )
        self.assertFalse(Path(calls[0][1]["env"]["DOCKER_CONFIG"]).exists())

    def test_login_failure_does_not_deploy_or_leave_credentials(self):
        configs = []

        def fail_login(command, **kwargs):
            configs.append(Path(kwargs["env"]["DOCKER_CONFIG"]))
            raise subprocess.CalledProcessError(1, command)

        with (
            patch.dict(os.environ, {"SSH_ORIGINAL_COMMAND": json.dumps(self.payload())}),
            patch.object(deploy_from_ci.subprocess, "run", side_effect=fail_login) as run,
            contextlib.redirect_stderr(io.StringIO()) as errors,
        ):
            self.assertEqual(deploy_from_ci.main(), 1)
        self.assertEqual(run.call_count, 1)
        self.assertFalse(configs[0].exists())
        self.assertNotIn(self.payload()["registry_token"], errors.getvalue())

    def test_deployment_failure_is_reported_after_credential_cleanup(self):
        with (
            patch.dict(os.environ, {"SSH_ORIGINAL_COMMAND": json.dumps(self.payload())}),
            patch.object(
                deploy_from_ci.subprocess,
                "run",
                side_effect=[subprocess.CompletedProcess([], 0), subprocess.CompletedProcess([], 9)],
            ) as run,
        ):
            self.assertEqual(deploy_from_ci.main(), 9)
        self.assertFalse(Path(run.call_args.kwargs["env"]["DOCKER_CONFIG"]).exists())


if __name__ == "__main__":
    unittest.main()
