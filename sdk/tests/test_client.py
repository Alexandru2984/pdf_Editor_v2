"""Unit tests for the PDF Editor SDK client (stdlib unittest, requests mocked)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from pdf_editor.client import ApiError, PdfEditorClient


def _make_client() -> tuple[PdfEditorClient, MagicMock]:
    session = MagicMock()
    client = PdfEditorClient("https://example.com/", api_key="k", session=session)
    return client, session


def _resp(status_code: int = 200, json_body=None, content: bytes | None = None):
    r = MagicMock()
    r.status_code = status_code
    r.headers = {"content-type": "application/json"} if json_body is not None else {}
    r.content = content if content is not None else (b'{"ok":true}' if json_body is None else b"{}")
    r.json.return_value = json_body if json_body is not None else {"ok": True}
    return r


class ClientTests(unittest.TestCase):
    def test_url_normalises_trailing_slash(self):
        c = PdfEditorClient("https://example.com/", api_key="k")
        self.assertEqual(c.base_url, "https://example.com")

    def test_api_key_header_is_set(self):
        c, session = _make_client()
        # Headers are stored on the session at init time.
        self.assertEqual(c._s.headers.update.call_args[0][0]["X-API-Key"], "k")

    def test_check_returns_json_on_2xx(self):
        c, _ = _make_client()
        self.assertEqual(c._check(_resp(200, {"id": 1})), {"id": 1})

    def test_check_raises_on_4xx(self):
        c, _ = _make_client()
        r = MagicMock(status_code=400, content=b'{"detail":"bad"}')
        r.json.return_value = {"detail": "bad"}
        with self.assertRaises(ApiError) as ctx:
            c._check(r)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_compress_posts_correctly(self):
        c, session = _make_client()
        session.post.return_value = _resp(201, {"id": "abc", "kind": "compress"})
        out = c.compress("pdf-1", quality="high")
        self.assertEqual(out["id"], "abc")
        called = session.post.call_args
        self.assertIn("/api/v1/ops/compress/", called[0][0])
        self.assertEqual(called.kwargs["json"], {"pdf_id": "pdf-1", "quality": "high"})

    def test_summarize_posts_correctly(self):
        c, session = _make_client()
        session.post.return_value = _resp(200, {"summary": "Hi", "truncated": False, "chars_used": 12})
        result = c.summarize("p1", language="Romanian")
        self.assertEqual(result["summary"], "Hi")
        self.assertEqual(session.post.call_args.kwargs["json"], {"pdf_id": "p1", "language": "Romanian"})

    def test_batch_posts_payload(self):
        c, session = _make_client()
        session.post.return_value = _resp(202, {"job_id": "j1"})
        c.batch("compress", ["a", "b"], params={"quality": "low"})
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(payload, {"op": "compress", "pdf_ids": ["a", "b"], "params": {"quality": "low"}})

    def test_wait_for_polls_until_terminal(self):
        c, session = _make_client()
        # Two non-terminal then one terminal response.
        responses = [
            _resp(200, {"id": "j1", "status": "running", "is_terminal": False}),
            _resp(200, {"id": "j1", "status": "running", "is_terminal": False}),
            _resp(200, {"id": "j1", "status": "done", "is_terminal": True}),
        ]
        session.get.side_effect = responses

        with patch("pdf_editor.client.time.sleep"):
            final = c.wait_for({"job_id": "j1"}, poll_interval=0.01, timeout=10)
        self.assertEqual(final["status"], "done")
        self.assertEqual(session.get.call_count, 3)

    def test_wait_for_times_out(self):
        c, session = _make_client()
        session.get.return_value = _resp(200, {"id": "j1", "status": "running", "is_terminal": False})
        with patch("pdf_editor.client.time.sleep"), self.assertRaises(TimeoutError):
            c.wait_for({"job_id": "j1"}, poll_interval=0.001, timeout=0)

    def test_wait_for_missing_id_raises(self):
        c, _ = _make_client()
        with self.assertRaises(ValueError):
            c.wait_for({"unrelated": "x"})


if __name__ == "__main__":
    unittest.main()
