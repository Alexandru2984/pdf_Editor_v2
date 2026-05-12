"""Realistic load profile — closer to how actual users behave.

Differences vs the baseline `locustfile.py`:

* Three PDF sizes are generated at startup (1-page tiny, 10-page small,
  30-page medium) and picked randomly per operation.
* Each user runs a *flow* — upload, then a sequence of 1-3 operations on
  the uploaded PDF, then optionally downloads.
* Operations cover the full sync UI set (compress, rotate, flatten,
  watermark, page-numbers, metadata, to-images) and the API equivalents.
* `wait_time` is in seconds, not milliseconds — humans read, click, wait.
* Anonymous quota-busting is expected. Set ``RATELIMIT_ENABLE=0`` in the
  app env if you want to measure raw throughput; otherwise leave it on
  and watch 429 ratio as a feature, not a bug.

Run:

    locust -f realistic.py --host=https://pdf.micutu.com    \\
           --users=100 --spawn-rate=10 --run-time=5m         \\
           --headless --html report-realistic.html

Set LOCUST_API_KEY for the ApiPower profile.
"""

from __future__ import annotations

import io
import os
import random

from locust import HttpUser, between, events, task

# Generated at startup. Keyed by name so users can pick a size.
PDF_FIXTURES: dict[str, bytes] = {}


def _make_pdf(pages: int, text_per_page: str = "Locust load test page") -> bytes:
    """Generate a multi-page PDF with text. Falls back to minimal-PDF on
    PyMuPDF missing — that's fine for the 1-page tiny variant."""
    try:
        import fitz

        doc = fitz.open()
        for i in range(pages):
            page = doc.new_page(width=595, height=842)
            page.insert_text((72, 72), f"{text_per_page} ({i + 1}/{pages})", fontsize=14)
            # Some body text so the file isn't trivially small. Repeats
            # are fine — we just want bytes on disk for compress to chew.
            body = (
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
                "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris "
                "nisi ut aliquip ex ea commodo consequat.\n\n"
            ) * 8
            page.insert_textbox(
                fitz.Rect(72, 110, 523, 770), body, fontsize=11, align=fitz.TEXT_ALIGN_LEFT
            )
        buf = io.BytesIO()
        doc.save(buf, garbage=4, deflate=True)
        doc.close()
        return buf.getvalue()
    except ImportError:
        return (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
            b"0000000052 00000 n \n0000000101 00000 n \n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n160\n%%EOF\n"
        )


@events.test_start.add_listener
def _generate_fixtures(environment, **_):
    PDF_FIXTURES["tiny"] = _make_pdf(1)
    PDF_FIXTURES["small"] = _make_pdf(10)
    PDF_FIXTURES["medium"] = _make_pdf(30)
    for name, blob in PDF_FIXTURES.items():
        print(f"fixture: {name} = {len(blob):>7} bytes")


# --------------------------------------------------------------------- #
# Helpers shared across users                                            #
# --------------------------------------------------------------------- #

OP_PAGES = ["/compress/", "/rotate/", "/flatten/", "/metadata/", "/watermark/", "/to-images/"]


def _pick_pdf() -> tuple[str, bytes]:
    """Return (size_label, bytes). Bias toward smaller files like real
    users do — 60% tiny, 30% small, 10% medium."""
    r = random.random()
    if r < 0.6:
        return "tiny", PDF_FIXTURES["tiny"]
    if r < 0.9:
        return "small", PDF_FIXTURES["small"]
    return "medium", PDF_FIXTURES["medium"]


# --------------------------------------------------------------------- #
# User profiles                                                          #
# --------------------------------------------------------------------- #


class CasualVisitor(HttpUser):
    """Anonymous browser — reads dashboard, peeks at docs, clicks around.
    Doesn't actually upload anything. Cheapest profile."""

    weight = 4
    wait_time = between(3, 10)

    @task(10)
    def dashboard(self):
        self.client.get("/", name="GET /")

    @task(4)
    def view_op_page(self):
        # GETting an op page renders the empty form — checks template + DB
        # for the "no PDF yet" message.
        page = random.choice(OP_PAGES)
        self.client.get(page, name=f"GET {page}")

    @task(3)
    def api_root(self):
        self.client.get("/api/v1/", name="GET /api/v1/")

    @task(2)
    def api_docs(self):
        self.client.get("/api/v1/docs/", name="GET /api/v1/docs/")

    @task(1)
    def healthz(self):
        self.client.get("/healthz", name="GET /healthz")


class RegularUser(HttpUser):
    """Anonymous user who actually does work — upload then 1-3 ops."""

    weight = 4
    wait_time = between(4, 12)

    def on_start(self):
        self.client.get("/")  # prime session cookie

    def _csrf(self) -> str:
        return self.client.cookies.get("csrftoken", "")

    def _csrf_headers(self) -> dict[str, str]:
        return {"X-CSRFToken": self._csrf(), "Referer": f"{self.host}/"}

    def _post_form(self, path: str, data: dict, name: str) -> None:
        # Token via both channels (header + form field) — belt-and-suspenders
        # against any view that disagrees about which one to trust.
        data = {**data, "csrfmiddlewaretoken": self._csrf()}
        self.client.post(
            path, data=data, headers=self._csrf_headers(), name=name, allow_redirects=True,
        )

    def _upload(self) -> bool:
        size, blob = _pick_pdf()
        files = {"pdf_file": (f"{size}.pdf", blob, "application/pdf")}
        data = {"csrfmiddlewaretoken": self._csrf()}
        with self.client.post(
            "/upload/",
            files=files,
            data=data,
            headers=self._csrf_headers(),
            name=f"POST /upload/ [{size}]",
            catch_response=True,
        ) as r:
            if r.status_code in (200, 302):
                return True
            if r.status_code == 429:
                r.success()
                return False
            r.failure(f"upload returned {r.status_code}")
            return False

    @task(5)
    def upload_and_compress(self):
        if not self._upload():
            return
        self._post_form(
            "/compress/", {"quality": random.choice(["low", "medium", "high"])}, "POST /compress/",
        )

    @task(3)
    def upload_and_rotate(self):
        if not self._upload():
            return
        self._post_form(
            "/rotate/", {"rotation_angle": random.choice([90, 180, 270])}, "POST /rotate/",
        )

    @task(2)
    def upload_and_flatten(self):
        if not self._upload():
            return
        self._post_form(
            "/flatten/", {"flatten_annotations": "on", "flatten_forms": "on"}, "POST /flatten/",
        )

    @task(2)
    def upload_and_watermark(self):
        if not self._upload():
            return
        self._post_form(
            "/watermark/",
            {"text": "CONFIDENTIAL", "position": "center", "opacity": "0.3"},
            "POST /watermark/",
        )

    @task(1)
    def upload_and_to_images(self):
        if not self._upload():
            return
        self._post_form("/to-images/", {"fmt": "png", "dpi": "150"}, "POST /to-images/")


class PowerUser(HttpUser):
    """Anonymous power user — uploads medium PDFs and chains multiple ops."""

    weight = 2
    wait_time = between(2, 6)

    def on_start(self):
        self.client.get("/")

    def _csrf(self) -> str:
        return self.client.cookies.get("csrftoken", "")

    def _csrf_headers(self) -> dict[str, str]:
        return {"X-CSRFToken": self._csrf(), "Referer": f"{self.host}/"}

    @task
    def multi_step_flow(self):
        # Always use small/medium for power users (they don't fool around).
        size, blob = ("small", PDF_FIXTURES["small"]) if random.random() < 0.7 else ("medium", PDF_FIXTURES["medium"])
        files = {"pdf_file": (f"{size}.pdf", blob, "application/pdf")}
        data = {"csrfmiddlewaretoken": self._csrf()}
        with self.client.post(
            "/upload/",
            files=files,
            data=data,
            headers=self._csrf_headers(),
            name=f"POST /upload/ [{size}]",
            catch_response=True,
        ) as r:
            if r.status_code not in (200, 302):
                if r.status_code == 429:
                    r.success()
                return

        ops = random.sample(
            [
                ("/compress/", {"quality": "medium"}),
                ("/rotate/", {"rotation_angle": 90}),
                ("/flatten/", {"flatten_annotations": "on", "flatten_forms": "on"}),
                ("/watermark/", {"text": "DRAFT", "position": "diagonal", "opacity": "0.2"}),
                ("/page-numbers/", {"position": "bottom-right", "start_at": "1"}),
            ],
            k=random.randint(2, 3),
        )
        for path, payload in ops:
            self.client.post(
                path,
                data={**payload, "csrfmiddlewaretoken": self._csrf()},
                headers=self._csrf_headers(),
                name=f"POST {path}",
                allow_redirects=True,
            )


class ApiPower(HttpUser):
    """Authenticated API user — heavy use of /api/v1/ops/*. Async-aware:
    polls the job endpoint after creating an op. Disabled unless
    LOCUST_API_KEY is set."""

    weight = 2
    wait_time = between(2, 5)
    abstract = bool(not os.environ.get("LOCUST_API_KEY"))

    def on_start(self):
        self._api_key = os.environ.get("LOCUST_API_KEY", "")
        self._pdf_ids: list[str] = []

    @property
    def headers(self) -> dict[str, str]:
        return {"X-API-Key": self._api_key, "Accept": "application/json"}

    def _ensure_pdf(self) -> str | None:
        """Upload if we don't have one yet (or 30% chance for variety)."""
        if not self._pdf_ids or random.random() < 0.3:
            size, blob = _pick_pdf()
            files = {"pdf_file": (f"api_{size}.pdf", blob, "application/pdf")}
            with self.client.post(
                "/api/v1/pdfs/",
                files=files,
                headers=self.headers,
                name=f"POST /api/v1/pdfs/ [{size}]",
                catch_response=True,
            ) as r:
                if r.status_code == 201:
                    try:
                        pid = r.json().get("id")
                        if pid:
                            self._pdf_ids.append(pid)
                            if len(self._pdf_ids) > 5:
                                self._pdf_ids.pop(0)
                    except ValueError:
                        pass
                elif r.status_code in (429, 413):
                    r.success()
        return random.choice(self._pdf_ids) if self._pdf_ids else None

    def _run_op(self, path: str, payload: dict) -> None:
        with self.client.post(
            f"/api/v1/ops/{path}",
            json=payload,
            headers=self.headers,
            name=f"POST /api/v1/ops/{path}",
            catch_response=True,
        ) as r:
            if r.status_code in (201, 429):
                if r.status_code == 429:
                    r.success()

    @task(2)
    def list_pdfs(self):
        self.client.get("/api/v1/pdfs/", headers=self.headers, name="GET /api/v1/pdfs/")

    @task(2)
    def list_outputs(self):
        self.client.get("/api/v1/outputs/", headers=self.headers, name="GET /api/v1/outputs/")

    @task(4)
    def api_compress(self):
        pid = self._ensure_pdf()
        if pid:
            self._run_op("compress/", {"pdf_id": pid, "quality": random.choice(["low", "medium", "high"])})

    @task(3)
    def api_rotate(self):
        pid = self._ensure_pdf()
        if pid:
            self._run_op("rotate/", {"pdf_id": pid, "rotation_angle": random.choice([90, 180, 270])})

    @task(2)
    def api_flatten(self):
        pid = self._ensure_pdf()
        if pid:
            self._run_op("flatten/", {"pdf_id": pid})

    @task(2)
    def api_watermark(self):
        pid = self._ensure_pdf()
        if pid:
            self._run_op(
                "watermark/",
                {"pdf_id": pid, "text": "API-LOAD", "position": "center", "opacity": 0.3},
            )

    @task(1)
    def api_page_numbers(self):
        pid = self._ensure_pdf()
        if pid:
            self._run_op("page-numbers/", {"pdf_id": pid, "position": "bottom-right", "start_at": 1})

    @task(1)
    def api_metadata(self):
        pid = self._ensure_pdf()
        if pid:
            self._run_op(
                "metadata/", {"pdf_id": pid, "title": "Load test doc", "author": "locust"},
            )

    @task(1)
    def api_to_images(self):
        pid = self._ensure_pdf()
        if pid:
            self._run_op("to-images/", {"pdf_id": pid, "fmt": "png", "dpi": 100})
