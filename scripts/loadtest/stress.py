"""Stress profile — find the breaking point.

Differences vs realistic.py:

* Almost-zero think time (0-1s vs realistic's 4-12s) — users hammer back-
  to-back, like a script gone wrong or a synthetic monitor on a tight loop.
* Heavier op mix — upload sizes biased toward `medium`, includes the
  expensive `/to-images/` PyMuPDF rasterizer at higher DPI.
* Bigger fixtures generated at startup (medium = 50 pages with images).
* All 4 profiles run regardless of API key (key-less ones still get a fair
  share of traffic).

Run:

    LOCUST_API_KEY=... locust -f stress.py                            \\
        --host=https://pdf.micutu.com                                 \\
        --users=500 --spawn-rate=25 --run-time=5m                     \\
        --headless --html=stress-500.html --csv=stress-500

For an even rougher test, use locust's distributed mode:

    # master:
    locust -f stress.py --master --host=...
    # worker(s):
    locust -f stress.py --worker --master-host=127.0.0.1
"""

from __future__ import annotations

import io
import os
import random

from locust import HttpUser, between, events, task

PDF_FIXTURES: dict[str, bytes] = {}


def _make_pdf(pages: int, with_image: bool = False) -> bytes:
    try:
        import fitz

        doc = fitz.open()
        for i in range(pages):
            page = doc.new_page(width=595, height=842)
            page.insert_text((72, 72), f"Stress page {i + 1}/{pages}", fontsize=14)
            body = (
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            ) * 20
            page.insert_textbox(
                fitz.Rect(72, 110, 523, 770), body, fontsize=10, align=fitz.TEXT_ALIGN_LEFT
            )
            if with_image:
                # Embed a small synthetic rect as filler — makes compress
                # have something to chew on beyond pure text.
                page.draw_rect(fitz.Rect(100, 300, 200, 400), fill=(0.8, 0.2, 0.2))
                page.draw_rect(fitz.Rect(250, 350, 400, 450), fill=(0.2, 0.6, 0.8))
        buf = io.BytesIO()
        doc.save(buf, garbage=4, deflate=True)
        doc.close()
        return buf.getvalue()
    except ImportError:
        return b"%PDF-1.4\n%%EOF\n"  # noqa: only hit when PyMuPDF missing


@events.test_start.add_listener
def _fixtures(environment, **_):
    PDF_FIXTURES["small"] = _make_pdf(5)
    PDF_FIXTURES["medium"] = _make_pdf(20, with_image=True)
    PDF_FIXTURES["large"] = _make_pdf(50, with_image=True)
    for name, blob in PDF_FIXTURES.items():
        print(f"stress fixture: {name} = {len(blob):>7} bytes")


def _pick_pdf() -> tuple[str, bytes]:
    # Biased toward larger files than realistic.py — we *want* to stress.
    r = random.random()
    if r < 0.4:
        return "small", PDF_FIXTURES["small"]
    if r < 0.85:
        return "medium", PDF_FIXTURES["medium"]
    return "large", PDF_FIXTURES["large"]


# --------------------------------------------------------------------- #


class HotBrowser(HttpUser):
    """Hammers the read paths with no mercy. 0-1s think time."""

    weight = 5
    wait_time = between(0, 1)

    @task(15)
    def home(self):
        self.client.get("/", name="GET /")

    @task(5)
    def api_root(self):
        self.client.get("/api/v1/", name="GET /api/v1/")

    @task(3)
    def healthz(self):
        self.client.get("/healthz", name="GET /healthz")

    @task(2)
    def op_page(self):
        page = random.choice(["/compress/", "/rotate/", "/flatten/", "/to-images/", "/watermark/"])
        self.client.get(page, name=f"GET {page}")


class HeavyAnonUploader(HttpUser):
    """Anonymous user that uploads + runs ops. Minimal pause between actions."""

    weight = 4
    wait_time = between(0, 2)

    def on_start(self):
        self.client.get("/")

    def _csrf(self) -> str:
        return self.client.cookies.get("csrftoken", "")

    def _csrf_h(self) -> dict[str, str]:
        return {"X-CSRFToken": self._csrf(), "Referer": f"{self.host}/"}

    def _upload(self) -> bool:
        size, blob = _pick_pdf()
        files = {"pdf_file": (f"{size}.pdf", blob, "application/pdf")}
        data = {"csrfmiddlewaretoken": self._csrf()}
        with self.client.post(
            "/upload/",
            files=files,
            data=data,
            headers=self._csrf_h(),
            name=f"POST /upload/ [{size}]",
            catch_response=True,
        ) as r:
            if r.status_code in (200, 302):
                return True
            if r.status_code == 429:
                r.success()
                return False
            r.failure(f"upload {r.status_code}")
            return False

    def _post(self, path: str, payload: dict, name: str) -> None:
        self.client.post(
            path,
            data={**payload, "csrfmiddlewaretoken": self._csrf()},
            headers=self._csrf_h(),
            name=name,
            allow_redirects=True,
        )

    @task(4)
    def upload_compress(self):
        if self._upload():
            self._post(
                "/compress/", {"quality": random.choice(["low", "medium", "high"])}, "POST /compress/",
            )

    @task(3)
    def upload_rotate(self):
        if self._upload():
            self._post(
                "/rotate/", {"rotation_angle": random.choice([90, 180, 270])}, "POST /rotate/",
            )

    @task(2)
    def upload_flatten(self):
        if self._upload():
            self._post(
                "/flatten/",
                {"flatten_annotations": "on", "flatten_forms": "on"},
                "POST /flatten/",
            )

    @task(2)
    def upload_watermark(self):
        if self._upload():
            self._post(
                "/watermark/",
                {"text": "STRESS", "position": "center", "opacity": "0.3"},
                "POST /watermark/",
            )

    @task(1)
    def upload_to_images(self):
        # The expensive one — PyMuPDF rasterize at 150 DPI.
        if self._upload():
            self._post("/to-images/", {"fmt": "png", "dpi": "150"}, "POST /to-images/")


class ApiHammer(HttpUser):
    """Authenticated API client running back-to-back ops."""

    weight = 3
    wait_time = between(0, 1)
    abstract = bool(not os.environ.get("LOCUST_API_KEY"))

    def on_start(self):
        self._key = os.environ.get("LOCUST_API_KEY", "")
        self._pdf_ids: list[str] = []
        self._warm_pdf()

    @property
    def headers(self) -> dict[str, str]:
        return {"X-API-Key": self._key, "Accept": "application/json"}

    def _warm_pdf(self) -> None:
        size, blob = _pick_pdf()
        files = {"pdf_file": (f"warm_{size}.pdf", blob, "application/pdf")}
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
                except ValueError:
                    pass
            elif r.status_code in (429, 413):
                r.success()

    def _pid(self) -> str | None:
        if not self._pdf_ids or random.random() < 0.15:
            self._warm_pdf()
        return random.choice(self._pdf_ids) if self._pdf_ids else None

    def _run(self, path: str, payload: dict) -> None:
        with self.client.post(
            f"/api/v1/ops/{path}",
            json=payload,
            headers=self.headers,
            name=f"POST /api/v1/ops/{path}",
            catch_response=True,
        ) as r:
            if r.status_code == 429:
                r.success()

    @task(2)
    def list_pdfs(self):
        self.client.get("/api/v1/pdfs/", headers=self.headers, name="GET /api/v1/pdfs/")

    @task(2)
    def list_outputs(self):
        self.client.get("/api/v1/outputs/", headers=self.headers, name="GET /api/v1/outputs/")

    @task(5)
    def compress(self):
        if pid := self._pid():
            self._run("compress/", {"pdf_id": pid, "quality": random.choice(["low", "medium", "high"])})

    @task(4)
    def rotate(self):
        if pid := self._pid():
            self._run("rotate/", {"pdf_id": pid, "rotation_angle": random.choice([90, 180, 270])})

    @task(3)
    def flatten(self):
        if pid := self._pid():
            self._run("flatten/", {"pdf_id": pid})

    @task(3)
    def watermark(self):
        if pid := self._pid():
            self._run(
                "watermark/",
                {"pdf_id": pid, "text": "STRESS-API", "position": "center", "opacity": 0.3},
            )

    @task(2)
    def page_numbers(self):
        if pid := self._pid():
            self._run("page-numbers/", {"pdf_id": pid, "position": "bottom-right", "start_at": 1})

    @task(1)
    def to_images(self):
        # Heavy op — make sure we exercise this path under stress.
        if pid := self._pid():
            self._run("to-images/", {"pdf_id": pid, "fmt": "png", "dpi": 100})


class GhostOpsBurst(HttpUser):
    """Anonymous burst hitting the unauthenticated API endpoints — these
    don't need a PDF, they just stress the routing + auth-bypass path."""

    weight = 2
    wait_time = between(0, 1)

    @task(3)
    def api_docs(self):
        self.client.get("/api/v1/docs/", name="GET /api/v1/docs/")

    @task(2)
    def api_schema(self):
        self.client.get("/api/v1/schema/", name="GET /api/v1/schema/")

    @task(1)
    def redoc(self):
        self.client.get("/api/v1/redoc/", name="GET /api/v1/redoc/")
