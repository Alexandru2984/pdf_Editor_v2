"""Tests for the legal pages (privacy policy + terms of service).

The suite runs with LANGUAGE_CODE="en" (see settings), so the default
variant served in tests is English; the Romanian variant is selected via
the Accept-Language header, exactly like a browser would.
"""

from django.test import TestCase
from django.urls import reverse


class LegalPagesTests(TestCase):
    def test_privacy_returns_english_variant(self):
        response = self.client.get(reverse("privacy"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Privacy Policy")
        self.assertContains(response, "24 hours")

    def test_privacy_returns_romanian_variant(self):
        response = self.client.get(reverse("privacy"), HTTP_ACCEPT_LANGUAGE="ro")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Politica de confidențialitate")

    def test_terms_returns_english_variant(self):
        response = self.client.get(reverse("terms"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Terms of Service")

    def test_terms_returns_romanian_variant(self):
        response = self.client.get(reverse("terms"), HTTP_ACCEPT_LANGUAGE="ro")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Termenii serviciului")

    def test_unknown_language_falls_back_to_english(self):
        response = self.client.get(reverse("privacy"), HTTP_ACCEPT_LANGUAGE="de")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Privacy Policy")

    def test_footer_links_present_on_dashboard(self):
        response = self.client.get(reverse("dashboard"))
        self.assertContains(response, reverse("privacy"))
        self.assertContains(response, reverse("terms"))

    def test_register_page_links_terms_and_privacy(self):
        response = self.client.get(reverse("register"))
        self.assertContains(response, reverse("terms"))
        self.assertContains(response, reverse("privacy"))
