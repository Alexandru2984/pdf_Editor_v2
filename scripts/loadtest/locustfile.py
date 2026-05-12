"""Locust load test for PDF Editor.

Three traffic profiles run in parallel via task weights:

  Visitor      — anonymous, browses the dashboard, looks at /api/v1/.
  Uploader     — anonymous, uploads a tiny PDF + runs compress.
  ApiClient    — authenticated via X-API-Key, hits /api/v1/ ops.

Run against a local container or staging:

    cd scripts/loadtest
    locust --host=https://pdf.micutu.com           \\
           --users=50 --spawn-rate=5 --run-time=2m \\
           --headless --html report.html

Be a good citizen: don't aim this at the prod URL above 10 concurrent
users without coordinating with the owner — rate limits and quotas will
kick in and the numbers won't reflect baseline performance anyway.

Environment variables:
    LOCUST_API_KEY    — set to enable the ApiClient profile
    LOCUST_PDF_BYTES  — preload a 1-page PDF; defaults to a generated one
"""

from __future__ import annotations

import contextlib
import io
import os
import random

from locust import HttpUser, between, events, task

# Minimal valid 1-page PDF, base64-ish encoded as raw bytes. Generated
# on first run if a real one isn't supplied via LOCUST_PDF_BYTES.
_PDF_BYTES: bytes | None = None


def _make_tiny_pdf() -> bytes:
    """Generate a 1-page PDF in memory using PyMuPDF if available, otherwise
    fall back to a static minimal-valid PDF byte string."""
    try:
        import fitz

        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), "Load test page", fontsize=12)
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        return buf.getvalue()
    except ImportError:
        # Hardcoded smallest-possible-PDF as a fallback. Valid enough that
        # PyMuPDF in the server accepts it; pages reads as 1.
        return (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
            b"0000000052 00000 n \n0000000101 00000 n \n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n160\n%%EOF\n"
        )


@events.test_start.add_listener
def _on_start(environment, **_):
    global _PDF_BYTES
    raw = os.environ.get("LOCUST_PDF_BYTES")
    if raw and os.path.exists(raw):
        with open(raw, "rb") as f:
            _PDF_BYTES = f.read()
        print(f"loaded PDF fixture from {raw} ({len(_PDF_BYTES)} bytes)")
    else:
        _PDF_BYTES = _make_tiny_pdf()
        print(f"generated 1-page PDF fixture ({len(_PDF_BYTES)} bytes)")


class Visitor(HttpUser):
    """Anonymous user clicking around. Should be cheap on the server."""

    weight = 5
    wait_time = between(1, 4)

    @task(10)
    def dashboard(self):
        self.client.get("/", name="GET /")

    @task(3)
    def api_root(self):
        self.client.get("/api/v1/", name="GET /api/v1/")

    @task(2)
    def api_docs(self):
        self.client.get("/api/v1/docs/", name="GET /api/v1/docs/")

    @task(1)
    def healthz(self):
        self.client.get("/healthz", name="GET /healthz")


class Uploader(HttpUser):
    """Anonymous user that actually does work — upload + compress.
    Subject to anon rate limits, so expect 429s if you push too hard."""

    weight = 3
    wait_time = between(2, 6)

    def on_start(self):
        # Prime a session cookie so we get a consistent owner_filter.
        self.client.get("/")

    @task
    def upload_then_compress(self):
        files = {"pdf_file": ("loadtest.pdf", _PDF_BYTES, "application/pdf")}
        csrf = self.client.cookies.get("csrftoken", "")
        headers = {"X-CSRFToken": csrf, "Referer": f"{self.host}/"}
        with self.client.post(
            "/upload/", files=files, headers=headers, name="POST /upload/", catch_response=True
        ) as r:
            if r.status_code not in (200, 302):
                r.failure(f"upload returned {r.status_code}")
                return

        # The latest uploaded PDF is the default selection on /compress/.
        with self.client.post(
            "/compress/",
            data={"quality": random.choice(["low", "medium", "high"]), "csrfmiddlewaretoken": csrf},
            headers=headers,
            name="POST /compress/",
            allow_redirects=False,
            catch_response=True,
        ) as r:
            if r.status_code not in (200, 302):
                r.failure(f"compress returned {r.status_code}")


class ApiClient(HttpUser):
    """Authenticated API client. Disabled unless LOCUST_API_KEY is set."""

    weight = 2
    wait_time = between(1, 3)
    abstract = bool(not os.environ.get("LOCUST_API_KEY"))

    def on_start(self):
        self._api_key = os.environ.get("LOCUST_API_KEY", "")
        self._pdf_id: str | None = None

    @property
    def auth_headers(self):
        return {"X-API-Key": self._api_key}

    @task(2)
    def list_pdfs(self):
        self.client.get("/api/v1/pdfs/", headers=self.auth_headers, name="GET /api/v1/pdfs/")

    @task(2)
    def list_outputs(self):
        self.client.get("/api/v1/outputs/", headers=self.auth_headers, name="GET /api/v1/outputs/")

    @task(1)
    def upload_pdf(self):
        files = {"pdf_file": ("apiloadtest.pdf", _PDF_BYTES, "application/pdf")}
        with self.client.post(
            "/api/v1/pdfs/",
            files=files,
            headers=self.auth_headers,
            name="POST /api/v1/pdfs/",
            catch_response=True,
        ) as r:
            if r.status_code == 201:
                with contextlib.suppress(ValueError):
                    self._pdf_id = r.json().get("id")
            elif r.status_code != 429:  # 429 is expected under heavy load
                r.failure(f"upload returned {r.status_code}")

    @task(3)
    def compress_pdf(self):
        if not self._pdf_id:
            return
        with self.client.post(
            "/api/v1/ops/compress/",
            json={"pdf_id": self._pdf_id, "quality": "medium"},
            headers=self.auth_headers,
            name="POST /api/v1/ops/compress/",
            catch_response=True,
        ) as r:
            if r.status_code not in (201, 429):
                r.failure(f"compress returned {r.status_code}")
