"""Browser regressions for document discovery, favorites and upload preflight."""

from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from django.test import override_settings
from django.utils import timezone

from ..models import MfaDevice, UploadedPDF
from .base import PlaywrightTestCase
from .test_smoke import _make_fixture_pdf


class WorkspaceTests(PlaywrightTestCase):
    def setUp(self):
        super().setUp()
        media = TemporaryDirectory(prefix="pdfeditor_workspace_")
        self.addCleanup(media.cleanup)
        config = override_settings(MEDIA_ROOT=media.name)
        config.enable()
        self.addCleanup(config.disable)
        self.user = self.make_user()
        for index, (name, status) in enumerate(
            [
                ("Factură 10.pdf", UploadedPDF.INGEST_SANITIZED),
                ("Contract 2.pdf", UploadedPDF.INGEST_SANITIZED),
                ("Secret.pdf", UploadedPDF.INGEST_ENCRYPTED),
            ]
        ):
            path = Path(media.name) / f"fixture-{index}.pdf"
            path.write_bytes(_make_fixture_pdf())
            pdf = UploadedPDF.objects.create(
                user=self.user, name=name, path=str(path), size=(index + 1) * 1000, ingest_status=status
            )
            UploadedPDF.objects.filter(pk=pdf.pk).update(uploaded_at=timezone.now() + timedelta(days=index))
        self.login()

    def test_document_search_sort_and_tool_favorites(self):
        errors = []
        self.page.on("pageerror", lambda error: errors.append(str(error)))
        self.page.locator("#documents-search").fill("factura")
        self.assertEqual(self.page.locator("#documents-grid .pdf-item-card:visible").count(), 1)
        self.assertIn("1 of 3", self.page.locator("#documents-count").inner_text())
        self.page.locator("#documents-search").fill("")
        self.page.locator("#documents-filter").select_option("encrypted")
        self.assertEqual(
            self.page.locator("#documents-grid .pdf-item-name:visible").inner_text(), "Secret.pdf"
        )
        self.page.locator("#documents-filter").select_option("all")
        self.page.locator("#documents-sort").select_option("name")
        self.assertEqual(
            self.page.locator("#documents-grid .pdf-item-name").first.inner_text(), "Contract 2.pdf"
        )
        self.page.locator("#documents-sort").select_option("largest")
        self.assertEqual(self.page.locator("#documents-grid .pdf-item-name").first.inner_text(), "Secret.pdf")
        self.assertEqual(self.page.locator("time").count(), 3)
        self.page.locator("#documents-search").fill("no such file")
        self.assertTrue(self.page.locator("#documents-no-results").is_visible())

        favorite = self.page.get_by_role("button", name="Favorite tool: Compress", exact=True)
        favorite.click()
        self.assertEqual(favorite.get_attribute("aria-pressed"), "true")
        self.page.reload()
        self.page.locator('[data-cat-filter="favorites"]').click()
        self.assertEqual(self.page.locator("#operations-grid .operation-card:visible").count(), 1)
        self.assertTrue(self.page.locator('[data-feature="compress"]').is_visible())
        self.page.get_by_role("button", name="Favorite tool: Compress", exact=True).click()
        self.assertTrue(self.page.locator("#tools-no-results").is_visible())
        self.assertEqual(errors, [])

    def test_mobile_layout_and_keyboard_search_in_both_languages(self):
        MfaDevice.objects.create(user=self.user, confirmed=True)
        for language in ("en", "ro"):
            self.page.goto(f"{self.live_server_url}/{language}/")
            for width in (320, 390, 768, 1440):
                with self.subTest(language=language, width=width):
                    self.page.set_viewport_size({"width": width, "height": 900})
                    self.assertTrue(
                        self.page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
                    )
                    box = self.page.locator(".tool-favorite").first.bounding_box()
                    self.assertGreaterEqual(box["width"], 44)
                    self.assertGreaterEqual(box["height"], 44)
                    upload_input = self.page.locator("#file-input").bounding_box()
                    self.assertLessEqual(upload_input["width"], 1)
                    self.assertLessEqual(upload_input["height"], 1)
            self.page.locator("body").click(position={"x": 3, "y": 400})
            self.page.keyboard.press("/")
            self.assertTrue(
                self.page.locator("#tools-search-input").evaluate("el => el === document.activeElement")
            )
            self.page.goto(f"{self.live_server_url}/accounts/profile/")
            self.page.set_viewport_size({"width": 320, "height": 900})
            self.assertTrue(
                self.page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"),
                self.page.evaluate(
                    "Array.from(document.querySelectorAll('main *'))"
                    ".filter(el => el.getBoundingClientRect().right > innerWidth + 1)"
                    ".slice(0, 8).map(el => el.tagName + '.' + el.className)"
                ),
            )


@override_settings(PDF_MAX_UPLOAD_BYTES=1024, PDF_MAX_BATCH_UPLOAD_BYTES=1500)
class UploadPreflightTests(PlaywrightTestCase):
    def test_invalid_files_never_submit_and_can_be_reselected(self):
        submitted = []
        self.page.on(
            "request", lambda request: submitted.append(request.url) if request.method == "POST" else None
        )
        self.page.goto(f"{self.live_server_url}/upload/")
        field = self.page.locator('input[name="pdf_file"]')
        field.set_input_files({"name": "notes.txt", "mimeType": "text/plain", "buffer": b"text"})
        self.assertIn("PDF files only", self.page.locator("#upload-error").inner_text())
        field.set_input_files({"name": "large.pdf", "mimeType": "application/pdf", "buffer": b"x" * 1025})
        self.assertIn("no larger", self.page.locator("#upload-error").inner_text())
        field.set_input_files(
            [
                {"name": name, "mimeType": "application/pdf", "buffer": b"x" * 800}
                for name in ("one.pdf", "two.pdf")
            ]
        )
        self.assertIn("batch size limit", self.page.locator("#upload-error").inner_text())
        self.page.locator("#upload-submit-btn").click()
        self.assertEqual(submitted, [])
        self.assertFalse(self.page.locator("#upload-submit-btn").is_disabled())
        field.set_input_files({"name": "small.pdf", "mimeType": "application/pdf", "buffer": b"%PDF-test"})
        self.assertTrue(self.page.locator("#upload-error").is_hidden())

        self.page.goto(f"{self.live_server_url}/en/")
        self.page.locator("#file-input").set_input_files(
            [
                {"name": f"part-{index}.pdf", "mimeType": "application/pdf", "buffer": b"x"}
                for index in range(21)
            ]
        )
        self.assertIn("at most 20", self.page.locator("#upload-error").inner_text())
        self.assertEqual(submitted, [])
