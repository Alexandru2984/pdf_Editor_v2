"""Tests for the legal pages (privacy policy + terms of service).

The suite runs with LANGUAGE_CODE="en" (see settings), so the default
variant served in tests is English; the Romanian variant is selected via
the Accept-Language header, exactly like a browser would.
"""

from django.test import TestCase
from django.urls import reverse


class LegalPagesTests(TestCase):
    def test_privacy_returns_english_variant(self):
        response = self.client.get(reverse("privacy_en"), HTTP_ACCEPT_LANGUAGE="ro")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Privacy Policy")
        self.assertContains(response, "24 hours")
        self.assertEqual(response["Content-Language"], "en")

    def test_privacy_returns_romanian_variant(self):
        response = self.client.get(reverse("privacy_ro"), HTTP_ACCEPT_LANGUAGE="en")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Politica de confidențialitate")
        self.assertEqual(response["Content-Language"], "ro")

    def test_terms_returns_english_variant(self):
        response = self.client.get(reverse("terms_en"), HTTP_ACCEPT_LANGUAGE="ro")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Terms of Service")

    def test_terms_returns_romanian_variant(self):
        response = self.client.get(reverse("terms_ro"), HTTP_ACCEPT_LANGUAGE="en")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Termenii serviciului")

    def test_unknown_language_falls_back_to_english(self):
        response = self.client.get(reverse("privacy"), HTTP_ACCEPT_LANGUAGE="de")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Privacy Policy")

    def test_footer_links_present_on_dashboard(self):
        response = self.client.get(reverse("dashboard_en"))
        self.assertContains(response, reverse("privacy_en"))
        self.assertContains(response, reverse("terms_en"))

    def test_register_page_links_terms_and_privacy(self):
        response = self.client.get(reverse("register"))
        self.assertContains(response, reverse("terms_en"))
        self.assertContains(response, reverse("privacy_en"))

    def test_legal_pages_reject_non_safe_methods(self):
        for url_name in ("privacy", "privacy_en", "privacy_ro", "terms", "terms_en", "terms_ro"):
            with self.subTest(url_name=url_name):
                response = self.client.post(reverse(url_name))

                self.assertEqual(response.status_code, 405)
