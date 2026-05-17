"""HTTP client wrapping the PDF Editor REST API.

The client returns plain ``dict`` payloads (matching what the API
serializes) so callers don't need a model layer. The only added value
is request shaping, auth, paginated list helpers, and a job-polling
``wait_for`` helper for the async ops.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests

_DEFAULT_TIMEOUT = 60  # seconds


class ApiError(RuntimeError):
    """Raised on non-2xx responses. Carries status code + response body."""

    def __init__(self, status_code: int, body: Any):
        self.status_code = status_code
        self.body = body
        super().__init__(f"PDF Editor API error {status_code}: {body!r}")


class PdfEditorClient:
    """Thin synchronous client.

    :param base_url:  e.g. ``"https://pdf.micutu.com"``. Trailing slash optional.
    :param api_key:   per-user X-API-Key issued from the profile page.
    :param timeout:   per-request timeout in seconds (default 60).
    :param session:   inject a preconfigured ``requests.Session`` for tests
                      / proxy / connection pooling tweaks.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._s = session or requests.Session()
        self._s.headers.update({"X-API-Key": api_key, "Accept": "application/json"})

    # ----- core HTTP helpers -------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/v1{path}"

    def _check(self, resp: requests.Response) -> Any:
        if 200 <= resp.status_code < 300:
            if resp.status_code == 204 or not resp.content:
                return None
            ctype = resp.headers.get("content-type", "")
            if "application/json" in ctype:
                return resp.json()
            return resp.content
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        raise ApiError(resp.status_code, body)

    def _get(self, path: str, params: dict | None = None) -> Any:
        return self._check(self._s.get(self._url(path), params=params, timeout=self.timeout))

    def _post(self, path: str, json: dict | None = None, files: dict | None = None, data: dict | None = None) -> Any:
        return self._check(
            self._s.post(self._url(path), json=json, files=files, data=data, timeout=self.timeout)
        )

    # ----- PDFs --------------------------------------------------------------

    def upload(self, file_path: str) -> dict:
        """Upload a PDF; returns the new ``UploadedPDF`` row."""
        with open(file_path, "rb") as fh:
            return self._post(
                "/pdfs/",
                files={"pdf_file": (os.path.basename(file_path), fh, "application/pdf")},
            )

    def list_pdfs(self, page: int = 1) -> dict:
        """Paginated list of the caller's UploadedPDFs."""
        return self._get("/pdfs/", params={"page": page})

    def delete_pdf(self, pdf_id: str) -> None:
        resp = self._s.delete(self._url(f"/pdfs/{pdf_id}/"), timeout=self.timeout)
        self._check(resp)

    # ----- Outputs -----------------------------------------------------------

    def list_outputs(self, page: int = 1) -> dict:
        return self._get("/outputs/", params={"page": page})

    def download(self, output_id: str, save_to: str) -> str:
        """Stream a ProcessedPDF to disk. Returns the saved path."""
        resp = self._s.get(self._url(f"/outputs/{output_id}/download/"), timeout=self.timeout, stream=True)
        if resp.status_code != 200:
            self._check(resp)
        with open(save_to, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if chunk:
                    fh.write(chunk)
        return save_to

    # ----- Sync ops (return a ProcessedPDF directly) -------------------------

    def compress(self, pdf_id: str, quality: str = "medium") -> dict:
        return self._post("/ops/compress/", json={"pdf_id": pdf_id, "quality": quality})

    def rotate(self, pdf_id: str, *, angle: int, page_range: str = "") -> dict:
        return self._post(
            "/ops/rotate/",
            json={"pdf_id": pdf_id, "rotation_angle": angle, "page_range": page_range},
        )

    def watermark(self, pdf_id: str, *, text: str, opacity: float = 0.3) -> dict:
        return self._post(
            "/ops/watermark/",
            json={"pdf_id": pdf_id, "text": text, "opacity": opacity},
        )

    def merge(self, pdf_ids: list[str]) -> dict:
        return self._post("/ops/merge/", json={"pdf_ids": pdf_ids})

    def summarize(self, pdf_id: str, language: str = "English") -> dict:
        return self._post("/ops/summarize/", json={"pdf_id": pdf_id, "language": language})

    # ----- Async ops (return a Job) ------------------------------------------

    def ocr(self, pdf_id: str, *, language: str = "eng+ron", dpi: int = 200) -> dict:
        return self._post(
            "/ops/searchable/", json={"pdf_id": pdf_id, "language": language, "dpi": dpi}
        )

    def to_pdfa(self, pdf_id: str, version: str = "2b") -> dict:
        return self._post("/ops/pdfa/", json={"pdf_id": pdf_id, "version": version})

    def convert_to_docx(self, pdf_id: str) -> dict:
        return self._post("/ops/convert-docx/", json={"pdf_id": pdf_id})

    def to_images(self, pdf_id: str, *, fmt: str = "png", dpi: int = 150) -> dict:
        return self._post("/ops/to-images/", json={"pdf_id": pdf_id, "fmt": fmt, "dpi": dpi})

    def batch(self, op: str, pdf_ids: list[str], params: dict | None = None) -> dict:
        return self._post("/ops/batch/", json={"op": op, "pdf_ids": pdf_ids, "params": params or {}})

    # ----- Jobs --------------------------------------------------------------

    def get_job(self, job_id: str) -> dict:
        return self._get(f"/jobs/{job_id}/")

    def list_jobs(self, *, status: list[str] | None = None, kind: str | None = None) -> dict:
        params = {}
        if status:
            params["status"] = status  # requests serialises list → repeated param
        if kind:
            params["kind"] = kind
        return self._get("/jobs/", params=params)

    def cancel_job(self, job_id: str) -> dict:
        return self._post(f"/jobs/{job_id}/cancel/")

    def wait_for(self, job_or_response: dict, *, poll_interval: float = 1.0, timeout: float = 300.0) -> dict:
        """Poll a job until it hits a terminal state. Accepts either a Job
        dict or the 202 response from an async op (which contains ``job_id``)."""
        job_id = job_or_response.get("job_id") or job_or_response.get("id")
        if not job_id:
            raise ValueError("Pass a Job or a 202 dict with 'job_id'.")
        deadline = time.monotonic() + timeout
        while True:
            job = self.get_job(str(job_id))
            if job.get("is_terminal"):
                return job
            if time.monotonic() > deadline:
                raise TimeoutError(f"Job {job_id} did not finish within {timeout}s.")
            time.sleep(poll_interval)
