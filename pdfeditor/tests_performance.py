"""Regression budgets for the public shell and production static delivery."""

from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase, override_settings
from django.urls import reverse


class PublicPerformanceBudgetTests(SimpleTestCase):
    def test_optimized_shell_assets_stay_within_transfer_budgets(self):
        static = Path(settings.BASE_DIR) / "static"
        budgets = {
            "images/logo-64.webp": 8_000,
            "images/logo-192.webp": 25_000,
            "images/logo-512.webp": 100_000,
            "images/apple-touch-icon.png": 100_000,
            "css/style.css": 90_000,
            "js/pdf-modal.js": 12_000,
        }
        for relative_path, maximum_bytes in budgets.items():
            with self.subTest(asset=relative_path):
                asset = static / relative_path
                self.assertTrue(asset.is_file())
                self.assertLessEqual(asset.stat().st_size, maximum_bytes)

    @override_settings(SITE_URL="https://pdf.example.test")
    def test_public_hub_defers_pdfjs_and_reserves_logo_dimensions(self):
        response = self.client.get(reverse("tools_hub_en"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "logo-64.webp")
        self.assertContains(response, 'width="52"')
        self.assertContains(response, 'height="52"')
        self.assertContains(response, 'data-pdfjs-url="/static/vendor/pdfjs/pdf.min.js"')
        self.assertNotContains(
            response,
            '<script src="/static/vendor/pdfjs/pdf.min.js',
            html=False,
        )
        self.assertNotContains(response, "analytics.micutu.com/script.js")

    @override_settings(SITE_URL="https://pdf.example.test", ANALYTICS_ENABLED=True)
    def test_production_analytics_remains_available_when_enabled(self):
        response = self.client.get(reverse("tools_hub_en"))

        self.assertContains(response, "https://analytics.micutu.com/script.js")

    def test_pdfjs_loader_waits_for_an_actual_preview_request(self):
        loader = (Path(settings.BASE_DIR) / "static/js/pdf-modal.js").read_text()

        self.assertIn("async ensurePdfJs()", loader)
        self.assertIn("await this.ensurePdfJs()", loader)
        self.assertIn('document.querySelector("[data-pdf-modal]")', loader)
        self.assertNotIn('onclick="pdfModal.close()"', loader)

    def test_production_nginx_caches_only_fingerprinted_assets_immutably(self):
        config = (Path(settings.BASE_DIR) / "infra/nginx/pdf_editor.conf").read_text()

        self.assertIn("max-age=31536000, immutable", config)
        self.assertIn("max-age=3600", config)
        self.assertIn("gzip on;", config)
        self.assertIn("add_header Cache-Control $pdf_static_cache_control always;", config)
