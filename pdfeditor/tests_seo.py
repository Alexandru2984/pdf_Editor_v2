"""SEO metadata, discovery endpoints and fail-closed indexing tests."""

from __future__ import annotations

import json
import re
from xml.etree import ElementTree

from django.contrib.auth import get_user_model
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from .seo import NOINDEX_ROBOTS, SEO_PAGES


@override_settings(SITE_URL="https://pdf.example.test")
class SeoTests(TestCase):
    def test_public_dashboard_has_canonical_social_metadata_and_valid_json_ld(self):
        response = self.client.get(f"{reverse('dashboard')}?utm_source=test")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("X-Robots-Tag", response)
        self.assertContains(
            response,
            '<link rel="canonical" href="https://pdf.example.test/">',
            html=True,
        )
        self.assertContains(response, '<meta name="robots" content="index, follow,', html=False)
        self.assertContains(response, 'property="og:url" content="https://pdf.example.test/"')
        self.assertContains(response, 'name="twitter:card" content="summary"')

        match = re.search(
            rb'<script nonce="[^"]+" type="application/ld\+json">(.*?)</script>',
            response.content,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        structured_data = json.loads(match.group(1))
        self.assertEqual(structured_data["@context"], "https://schema.org")
        types = {item["@type"] for item in structured_data["@graph"]}
        self.assertEqual(types, {"WebSite", "WebApplication"})

    def test_romanian_request_gets_localized_metadata(self):
        response = self.client.get(reverse("dashboard"), HTTP_ACCEPT_LANGUAGE="ro")

        self.assertContains(response, "Instrumente PDF online gratuite")
        self.assertContains(response, 'property="og:locale" content="ro_RO"')
        self.assertContains(response, "Editează, unește, separă")

    def test_private_page_has_meta_and_header_noindex(self):
        response = self.client.get(reverse("login"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["X-Robots-Tag"], NOINDEX_ROBOTS)
        self.assertContains(response, f'<meta name="robots" content="{NOINDEX_ROBOTS}">', html=True)
        self.assertNotContains(response, 'rel="canonical"')
        self.assertNotContains(response, 'type="application/ld+json"')

    def test_authenticated_dashboard_is_not_indexable(self):
        user = get_user_model().objects.create_user(username="seo-user", password="test-password")
        self.client.force_login(user)

        response = self.client.get(reverse("dashboard"))

        self.assertEqual(response["X-Robots-Tag"], NOINDEX_ROBOTS)
        self.assertContains(response, f'<meta name="robots" content="{NOINDEX_ROBOTS}">', html=True)
        self.assertNotContains(response, 'rel="canonical"')

    def test_session_scoped_dashboard_is_not_indexable(self):
        session = self.client.session
        session["owner_state"] = "present"
        session.save()

        response = self.client.get(reverse("dashboard"))

        self.assertEqual(response["X-Robots-Tag"], NOINDEX_ROBOTS)
        self.assertNotContains(response, 'rel="canonical"')

    def test_post_to_public_tool_is_not_indexable(self):
        response = self.client.post(reverse("images_to_pdf"), {})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["X-Robots-Tag"], NOINDEX_ROBOTS)
        self.assertContains(response, f'<meta name="robots" content="{NOINDEX_ROBOTS}">', html=True)

    def test_robots_points_to_canonical_sitemap_and_limits_technical_routes(self):
        response = self.client.get(reverse("robots_txt"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/plain; charset=utf-8")
        self.assertContains(response, "User-agent: *")
        self.assertContains(response, "Disallow: /api/")
        self.assertContains(response, "Sitemap: https://pdf.example.test/sitemap.xml")
        self.assertNotContains(response, "Disallow: /accounts/")
        self.assertEqual(response["Cache-Control"], "public, max-age=3600")

    def test_sitemap_contains_only_reviewed_absolute_canonical_urls(self):
        response = self.client.get(reverse("sitemap_xml"))

        self.assertEqual(response.status_code, 200)
        root = ElementTree.fromstring(response.content)
        namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        locations = [node.text for node in root.findall("sm:url/sm:loc", namespace)]
        expected = [f"https://pdf.example.test{reverse(page.url_name)}" for page in SEO_PAGES.values()]
        self.assertEqual(locations, expected)
        self.assertNotIn("https://pdf.example.test/accounts/login/", locations)
        self.assertNotIn("https://pdf.example.test/result/", locations)

    def test_every_sitemap_page_is_a_fresh_indexable_get(self):
        for page in SEO_PAGES.values():
            with self.subTest(page=page.url_name):
                response = Client().get(reverse(page.url_name))

                self.assertEqual(response.status_code, 200)
                self.assertNotIn("X-Robots-Tag", response)
                self.assertContains(response, '<link rel="canonical" href="https://pdf.example.test')
                self.assertContains(response, 'type="application/ld+json"')

    @override_settings(ALLOWED_HOSTS=["*"])
    def test_canonical_origin_never_uses_untrusted_host_header(self):
        response = self.client.get(reverse("dashboard"), HTTP_HOST="attacker.example")

        self.assertContains(response, 'href="https://pdf.example.test/"')
        self.assertNotContains(response, "attacker.example")

    def test_head_is_supported_for_discovery_endpoints(self):
        robots = self.client.head(reverse("robots_txt"))
        sitemap = self.client.head(reverse("sitemap_xml"))

        self.assertEqual(robots.status_code, 200)
        self.assertEqual(sitemap.status_code, 200)

    def test_non_safe_discovery_requests_are_rejected_and_noindexed(self):
        robots = self.client.post(reverse("robots_txt"))
        sitemap = self.client.post(reverse("sitemap_xml"))

        self.assertEqual(robots.status_code, 405)
        self.assertEqual(sitemap.status_code, 405)
        self.assertEqual(robots["X-Robots-Tag"], NOINDEX_ROBOTS)
        self.assertEqual(sitemap["X-Robots-Tag"], NOINDEX_ROBOTS)
