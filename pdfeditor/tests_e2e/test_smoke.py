"""Golden-path E2E: login → upload → compress → download.

If this passes, the whole user-facing chain — Django views, templates,
form-submit JS, file upload multipart, the compress op, session-based
result handoff, and the download endpoint — is working end-to-end.

If it fails, the diff between unit tests passing and this failing tells
you something only a browser can see: a missing JS file, a broken form
selector, a redirect that lost a query param.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import fitz  # PyMuPDF — already a prod dep, used for the fixture PDF.
from django.test import override_settings

from .base import PlaywrightTestCase


def _make_fixture_pdf() -> bytes:
    """Generate a tiny but valid PDF — three pages with a sentence each.

    Generating in-memory keeps tests reproducible and lets us cleanly cap
    the file at whatever size the compress op needs. Reading a real PDF
    from disk would couple tests to a checked-in fixture path.
    """
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"E2E test fixture, page {i + 1}.")
    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True)
    doc.close()
    return buf.getvalue()


# Each test class gets its own MEDIA_ROOT so uploads from previous runs
# don't bleed in. tempfile.mkdtemp leaks the dir on test failure, which
# is what we want — easier to inspect when a CI run breaks.
@override_settings(MEDIA_ROOT=tempfile.mkdtemp(prefix="pdfedit_e2e_"))
class SmokeFlowTests(PlaywrightTestCase):
    def test_login_upload_compress_download(self) -> None:
        self.make_user()
        self.login()

        # ---- Upload ------------------------------------------------------
        self.page.goto(f"{self.live_server_url}/upload/")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(_make_fixture_pdf())
            fixture_path = tmp.name
        try:
            upload_form = self.page.locator("form.upload-form")
            upload_form.locator('input[name="pdf_file"]').set_input_files(fixture_path)
            upload_form.locator('button[type="submit"]').click()
            self.page.wait_for_url(f"{self.live_server_url}/")
        finally:
            Path(fixture_path).unlink(missing_ok=True)

        # ---- Compress ----------------------------------------------------
        # If the upload didn't land, compress_view redirects to /; asserting
        # we stayed on /compress/ proves the upload actually persisted.
        self.page.goto(f"{self.live_server_url}/compress/")
        self.page.wait_for_url(f"{self.live_server_url}/compress/")
        # The compress page's <form method="post"> wraps the quality radios
        # and the submit button. Default quality is pre-checked.
        compress_form = self.page.locator('form[method="post"]:has(input[name="quality"])')
        compress_form.locator('button[type="submit"]').click()
        self.page.wait_for_url(f"{self.live_server_url}/compress/result/")

        # ---- Download ----------------------------------------------------
        # The result page has a download anchor — use Playwright's
        # expect_download context to capture the file rather than
        # navigating the page away.
        with self.page.expect_download() as download_info:
            self.page.click("a[href*='/download_compressed']")
        download = download_info.value
        # Saving forces Playwright to finish streaming and surface errors.
        save_path = Path(tempfile.gettempdir()) / "e2e-compressed.pdf"
        download.save_as(str(save_path))
        try:
            data = save_path.read_bytes()
            assert data.startswith(b"%PDF-"), f"downloaded bytes are not a PDF: {data[:20]!r}"
            assert len(data) > 0
        finally:
            save_path.unlink(missing_ok=True)


@override_settings(MEDIA_ROOT=tempfile.mkdtemp(prefix="pdfedit_e2e_"))
class AuthGuardrailTests(PlaywrightTestCase):
    def test_bad_password_stays_on_login(self) -> None:
        """Wrong password → render login.html again with an error message,
        do NOT redirect to dashboard. Catches the case where a regression
        accidentally trusts the form."""
        self.make_user()  # creates `alice` with the fixture password from base.py

        self.page.goto(f"{self.live_server_url}/accounts/login/")
        form = self.page.locator("form.edit-form")
        form.locator('input[name="username"]').fill("alice")
        form.locator('input[name="password"]').fill("definitely-wrong")
        form.locator('button[type="submit"]').click()

        # Stayed on the login URL — Django's LoginView re-renders on bad creds.
        # We can't `wait_for_url` because the URL didn't change; instead,
        # check that the password input is still on the page.
        self.page.wait_for_selector('input[name="password"]')
        assert "/accounts/login" in self.page.url
