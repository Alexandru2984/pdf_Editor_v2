"""CLI smoke tests — invoke main() with mocked client."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from pdf_editor.cli import build_parser, main


class ParserTests(unittest.TestCase):
    def test_all_subcommands_registered(self):
        parser = build_parser()
        choices = parser._subparsers._group_actions[0].choices.keys()  # noqa: SLF001
        self.assertEqual(
            set(choices),
            {"upload", "list", "compress", "summarize", "batch", "ocr", "cancel"},
        )


class MainEntrypointTests(unittest.TestCase):
    def _run(self, argv):
        with patch.dict("os.environ", {"PDF_EDITOR_URL": "https://x.example", "PDF_EDITOR_API_KEY": "k"}):
            return main(argv)

    @patch("pdf_editor.cli.PdfEditorClient")
    def test_summarize_prints_just_the_summary(self, MockClient):
        instance = MockClient.return_value
        instance.summarize.return_value = {"summary": "Hello.", "truncated": False, "chars_used": 6}

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = self._run(["summarize", "pdf-1", "--language", "English"])
        self.assertEqual(rc, 0)
        self.assertEqual(buf.getvalue().strip(), "Hello.")
        instance.summarize.assert_called_once_with("pdf-1", language="English")

    @patch("pdf_editor.cli.PdfEditorClient")
    def test_list_jobs_passes_filters(self, MockClient):
        instance = MockClient.return_value
        instance.list_jobs.return_value = {"results": []}
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = self._run(["list", "jobs", "--status", "queued", "--status", "running"])
        self.assertEqual(rc, 0)
        instance.list_jobs.assert_called_once_with(status=["queued", "running"], kind=None)

    @patch("pdf_editor.cli.PdfEditorClient")
    def test_batch_waits_unless_no_wait(self, MockClient):
        instance = MockClient.return_value
        instance.batch.return_value = {"job_id": "j1"}
        instance.wait_for.return_value = {"job_id": "j1", "status": "done"}
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = self._run(["batch", "compress", "a", "b", "--params", '{"quality":"low"}'])
        self.assertEqual(rc, 0)
        instance.wait_for.assert_called_once()

    @patch("pdf_editor.cli.PdfEditorClient")
    def test_batch_no_wait_returns_submission(self, MockClient):
        instance = MockClient.return_value
        instance.batch.return_value = {"job_id": "j1"}
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = self._run(["batch", "compress", "a", "--no-wait"])
        self.assertEqual(rc, 0)
        instance.wait_for.assert_not_called()

    def test_missing_credentials_exits(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("pdf_editor.cli._load_config", return_value={}),
            self.assertRaises(SystemExit) as ctx,
        ):
            main(["upload", "x.pdf"])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
