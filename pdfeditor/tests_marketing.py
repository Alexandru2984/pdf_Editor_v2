"""Public PDF tool catalogue, landing content and upload continuation tests."""

from __future__ import annotations

import json
import re
import shutil
import tempfile

import fitz
from django.contrib.sessions.models import Session
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from .seo import NOINDEX_ROBOTS
from .tool_catalog import DIRECT_TOOL_LINKS, TOOL_LANDINGS, is_upload_destination_allowed


class _TemporaryMediaMixin:
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._media_root = tempfile.mkdtemp(prefix="pdfeditor_marketing_tests_")
        cls._media_override = override_settings(MEDIA_ROOT=cls._media_root)
        cls._media_override.enable()

    @classmethod
    def tearDownClass(cls):
        cls._media_override.disable()
        shutil.rmtree(cls._media_root, ignore_errors=True)
        super().tearDownClass()


@override_settings(SITE_URL="https://pdf.example.test")
class MarketingPagesTests(_TemporaryMediaMixin, TestCase):
    @staticmethod
    def _pdf(name: str = "source.pdf") -> SimpleUploadedFile:
        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), "SEO upload continuation test")
        payload = document.tobytes()
        document.close()
        return SimpleUploadedFile(name, payload, content_type="application/pdf")

    def test_catalog_content_is_unique_and_complete(self):
        self.assertGreaterEqual(len(TOOL_LANDINGS), 12)
        self.assertEqual(len({tool.key for tool in TOOL_LANDINGS}), len(TOOL_LANDINGS))
        self.assertEqual(len({tool.slug for tool in TOOL_LANDINGS}), len(TOOL_LANDINGS))
        self.assertEqual(len({tool.url_name for tool in TOOL_LANDINGS}), len(TOOL_LANDINGS))
        self.assertEqual(
            len({tool.localized_url_name(language) for tool in TOOL_LANDINGS for language in ("en", "ro")}),
            len(TOOL_LANDINGS) * 2,
        )
        self.assertEqual(len({tool.seo_title.en for tool in TOOL_LANDINGS}), len(TOOL_LANDINGS))
        self.assertEqual(len({tool.seo_description.en for tool in TOOL_LANDINGS}), len(TOOL_LANDINGS))
        for tool in TOOL_LANDINGS:
            with self.subTest(tool=tool.key):
                self.assertEqual(len(tool.highlights), 3)
                self.assertEqual(len(tool.steps), 3)
                self.assertGreaterEqual(len(tool.faqs), 2)
                self.assertLessEqual(len(tool.seo_title.en), 60)
                self.assertLessEqual(len(tool.seo_title.ro), 60)
                self.assertGreaterEqual(len(tool.seo_description.en), 110)
                self.assertLessEqual(len(tool.seo_description.en), 160)
                self.assertGreaterEqual(len(tool.seo_description.ro), 110)
                self.assertLessEqual(len(tool.seo_description.ro), 160)

    def test_hub_lists_guides_and_direct_tools(self):
        response = self.client.get(reverse("tools_hub_en"))

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("X-Robots-Tag", response)
        self.assertContains(
            response,
            'class="public-tool-card"',
            count=len(TOOL_LANDINGS) + len(DIRECT_TOOL_LINKS),
        )
        self.assertContains(response, reverse("images_to_pdf"))
        self.assertContains(response, reverse("verify_signature"))
        for tool in TOOL_LANDINGS:
            self.assertContains(response, reverse(tool.localized_url_name("en")))

    def test_each_landing_is_indexable_unique_content_with_one_h1(self):
        before_sessions = Session.objects.count()
        for tool in TOOL_LANDINGS:
            with self.subTest(tool=tool.key):
                response = Client().get(reverse(tool.localized_url_name("en")))

                self.assertEqual(response.status_code, 200)
                self.assertNotIn("X-Robots-Tag", response)
                self.assertContains(response, tool.name.en)
                self.assertContains(response, tool.intro.en)
                self.assertContains(response, tool.note.en)
                self.assertContains(response, f"?next={tool.operation_url_name}")
                self.assertEqual(len(re.findall(rb"<h1\b", response.content)), 1)
        self.assertEqual(Session.objects.count(), before_sessions)

    def test_romanian_landing_content_is_explicit(self):
        tool = TOOL_LANDINGS[0]
        response = self.client.get(
            reverse(tool.localized_url_name("ro")),
            HTTP_ACCEPT_LANGUAGE="en",
        )

        self.assertContains(response, tool.name.ro)
        self.assertContains(response, tool.intro.ro)
        self.assertContains(response, "Încarcă și continuă")
        self.assertContains(response, 'property="og:locale" content="ro_RO"')
        self.assertEqual(response["Content-Language"], "ro")

    def test_landing_json_ld_matches_visible_how_to_steps(self):
        tool = TOOL_LANDINGS[0]
        response = self.client.get(reverse(tool.localized_url_name("en")))
        match = re.search(
            rb'<script nonce="[^"]+" type="application/ld\+json">(.*?)</script>',
            response.content,
            re.DOTALL,
        )

        self.assertIsNotNone(match)
        structured_data = json.loads(match.group(1))
        webpage = next(item for item in structured_data["@graph"] if item["@type"] == "WebPage")
        how_to = webpage["mainEntity"]
        self.assertEqual(how_to["@type"], "HowTo")
        self.assertEqual([step["text"] for step in how_to["step"]], [step.en for step in tool.steps])
        self.assertTrue(any(item["@type"] == "BreadcrumbList" for item in structured_data["@graph"]))

    def test_hub_json_ld_lists_every_visible_tool(self):
        response = self.client.get(reverse("tools_hub_en"))
        match = re.search(
            rb'<script nonce="[^"]+" type="application/ld\+json">(.*?)</script>',
            response.content,
            re.DOTALL,
        )

        self.assertIsNotNone(match)
        structured_data = json.loads(match.group(1))
        page = next(item for item in structured_data["@graph"] if item["@type"] == "CollectionPage")
        item_list = page["mainEntity"]
        self.assertEqual(item_list["numberOfItems"], len(TOOL_LANDINGS) + len(DIRECT_TOOL_LINKS))
        self.assertEqual(len(item_list["itemListElement"]), item_list["numberOfItems"])
        listed_urls = {item["url"] for item in item_list["itemListElement"]}
        self.assertIn(
            f"https://pdf.example.test{reverse(TOOL_LANDINGS[0].localized_url_name('en'))}",
            listed_urls,
        )

    def test_hub_and_landings_support_head_requests(self):
        names = [
            *(f"tools_hub_{language}" for language in ("en", "ro")),
            *(tool.localized_url_name(language) for tool in TOOL_LANDINGS for language in ("en", "ro")),
        ]
        for name in names:
            with self.subTest(name=name):
                response = self.client.head(reverse(name))

                self.assertEqual(response.status_code, 200)
                self.assertNotIn("X-Robots-Tag", response)

    def test_merge_landing_accepts_multiple_files(self):
        response = self.client.get(reverse("tool_merge_pdf_en"))

        self.assertContains(response, "multiple", html=False)
        self.assertContains(response, "Select at least two PDFs")

    def test_upload_can_continue_to_fixed_internal_tool(self):
        response = self.client.post(
            f"{reverse('upload')}?next=compress",
            {"pdf_file": self._pdf()},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("compress"))

    def test_upload_rejects_external_or_unknown_continuation(self):
        for destination in ("https://attacker.example/", "admin:index", "missing-view"):
            with self.subTest(destination=destination):
                response = Client().post(
                    f"{reverse('upload')}?next={destination}",
                    {"pdf_file": self._pdf(name=f"{len(destination)}.pdf")},
                )

                self.assertEqual(response.status_code, 302)
                self.assertEqual(response.url, reverse("dashboard"))

    def test_upload_destination_allowlist_is_exact(self):
        for tool in TOOL_LANDINGS:
            self.assertTrue(is_upload_destination_allowed(tool.operation_url_name))
        self.assertFalse(is_upload_destination_allowed(""))
        self.assertFalse(is_upload_destination_allowed("dashboard"))
        self.assertFalse(is_upload_destination_allowed("https://attacker.example/"))

    def test_unknown_tool_and_post_are_noindexed(self):
        missing = self.client.get("/tools/not-a-real-tool/")
        localized_missing = self.client.get("/en/pdf-tools/not-a-real-tool/")
        post = self.client.post(reverse(TOOL_LANDINGS[0].localized_url_name("en")))

        self.assertEqual(missing.status_code, 404)
        self.assertEqual(localized_missing.status_code, 404)
        self.assertEqual(post.status_code, 405)
        self.assertEqual(missing["X-Robots-Tag"], NOINDEX_ROBOTS)
        self.assertEqual(localized_missing["X-Robots-Tag"], NOINDEX_ROBOTS)
        self.assertEqual(post["X-Robots-Tag"], NOINDEX_ROBOTS)

    def test_generic_urls_negotiate_language_and_remain_noindex(self):
        tool = TOOL_LANDINGS[0]
        cases = (
            (reverse("tools_hub"), "en", reverse("tools_hub_en")),
            (reverse("tools_hub"), "ro", reverse("tools_hub_ro")),
            (
                reverse(tool.url_name),
                "en",
                reverse(tool.localized_url_name("en")),
            ),
            (
                reverse(tool.url_name),
                "ro",
                reverse(tool.localized_url_name("ro")),
            ),
        )
        for source, language, destination in cases:
            with self.subTest(source=source, language=language):
                response = Client().get(source, HTTP_ACCEPT_LANGUAGE=language)

                self.assertEqual(response.status_code, 302)
                self.assertEqual(response.url, destination)
                self.assertEqual(response["X-Robots-Tag"], NOINDEX_ROBOTS)
                self.assertEqual(response["Cache-Control"], "private, no-store")
                vary = {item.strip() for item in response["Vary"].split(",")}
                self.assertTrue({"Accept-Language", "Cookie"}.issubset(vary))

    def test_localized_pages_publish_reciprocal_hreflang_and_switch_target(self):
        tool = TOOL_LANDINGS[0]
        generic_url = f"https://pdf.example.test{reverse(tool.url_name)}"
        for language, other_language in (("en", "ro"), ("ro", "en")):
            with self.subTest(language=language):
                url_name = tool.localized_url_name(language)
                response = self.client.get(reverse(url_name))
                canonical = f"https://pdf.example.test{reverse(url_name)}"

                self.assertContains(response, f'<link rel="canonical" href="{canonical}">')
                for alternate_language in ("en", "ro"):
                    alternate = (
                        f"https://pdf.example.test{reverse(tool.localized_url_name(alternate_language))}"
                    )
                    self.assertContains(
                        response,
                        f'<link rel="alternate" hreflang="{alternate_language}" href="{alternate}">',
                    )
                self.assertContains(
                    response,
                    f'<link rel="alternate" hreflang="x-default" href="{generic_url}">',
                )
                self.assertContains(
                    response,
                    f'name="next" type="hidden" value="{reverse(tool.localized_url_name(other_language))}"',
                )
                self.assertEqual(response["Content-Language"], language)
                self.assertContains(response, f'<html lang="{language}">')
