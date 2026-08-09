"""Search-engine metadata and discovery endpoints.

The application mixes public landing pages with authenticated/session-scoped
workflows.  Keep the allowlist of indexable pages deliberately small: a new
view is private to crawlers until it is reviewed and added here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit
from xml.etree import ElementTree

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.serializers.json import DjangoJSONEncoder
from django.http import HttpRequest, HttpResponse
from django.urls import reverse
from django.utils.safestring import mark_safe
from django.utils.translation import get_language
from django.views.decorators.http import require_safe

from .tool_catalog import DIRECT_TOOL_LINKS, TOOL_LANDINGS, TOOL_LANDINGS_BY_URL_NAME


@dataclass(frozen=True, slots=True)
class SeoPage:
    """Metadata for a public page that may appear in search results."""

    url_name: str
    title_en: str
    title_ro: str
    description_en: str
    description_ro: str
    anonymous_only: bool = False
    page_type: str = "WebPage"

    def title(self, language: str) -> str:
        return self.title_ro if language == "ro" else self.title_en

    def description(self, language: str) -> str:
        return self.description_ro if language == "ro" else self.description_en


# Only include URLs that return useful, non-sensitive content to a fresh GET.
# Operation/result/account pages intentionally remain noindex until they have a
# dedicated public landing page that does not depend on an uploaded document.
SEO_PAGES: dict[str, SeoPage] = {
    page.url_name: page
    for page in (
        SeoPage(
            url_name="dashboard",
            title_en="Free Online PDF Tools | PDF Platform",
            title_ro="Instrumente PDF online gratuite | PDF Platform",
            description_en=(
                "Edit, merge, split, compress, convert, protect, sign, OCR and organize PDF files "
                "online with one privacy-conscious toolkit."
            ),
            description_ro=(
                "Editează, unește, separă, comprimă, convertește, protejează, semnează și organizează "
                "fișiere PDF online, într-o singură platformă."
            ),
            anonymous_only=True,
            page_type="WebApplication",
        ),
        SeoPage(
            url_name="images_to_pdf",
            title_en="Convert Images to PDF Online | PDF Platform",
            title_ro="Convertește imagini în PDF online | PDF Platform",
            description_en=(
                "Combine JPG, PNG, TIFF, WebP, BMP or GIF images into one PDF. Reorder pages and choose "
                "print-ready page sizing directly in your browser."
            ),
            description_ro=(
                "Combină imagini JPG, PNG, TIFF, WebP, BMP sau GIF într-un singur PDF. Reordonează paginile "
                "și alege dimensiunea potrivită pentru tipărire."
            ),
        ),
        SeoPage(
            url_name="verify_signature",
            title_en="Verify PDF Digital Signatures Online | PDF Platform",
            title_ro="Verifică semnături digitale PDF online | PDF Platform",
            description_en=(
                "Inspect PDF digital signatures for document integrity, signer identity, certificate trust, "
                "revocation status and trusted timestamps."
            ),
            description_ro=(
                "Verifică semnăturile digitale PDF: integritatea documentului, identitatea semnatarului, "
                "încrederea certificatului, revocarea și marca temporală."
            ),
        ),
        SeoPage(
            url_name="privacy",
            title_en="Privacy Policy | PDF Platform",
            title_ro="Politica de confidențialitate | PDF Platform",
            description_en=(
                "Learn what PDF Platform processes, where document data goes, how long it is retained and "
                "which privacy rights you can exercise."
            ),
            description_ro=(
                "Află ce date prelucrează PDF Platform, unde ajung documentele, cât timp sunt păstrate și "
                "cum îți poți exercita drepturile."
            ),
        ),
        SeoPage(
            url_name="terms",
            title_en="Terms of Service | PDF Platform",
            title_ro="Termenii serviciului | PDF Platform",
            description_en=(
                "Read the rules, responsibilities and service conditions that apply when using PDF Platform."
            ),
            description_ro=(
                "Consultă regulile, responsabilitățile și condițiile aplicabile utilizării PDF Platform."
            ),
        ),
    )
}

SEO_PAGES["tools_hub"] = SeoPage(
    url_name="tools_hub",
    title_en="Free Online PDF Tools | Convert, Edit & Secure",
    title_ro="Instrumente PDF online | Conversie, editare și securitate",
    description_en=(
        "Explore focused online PDF tools for merging, splitting, compression, OCR, conversion, "
        "redaction, passwords, signatures and page organization."
    ),
    description_ro=(
        "Descoperă instrumente PDF online pentru unire, separare, compresie, OCR, conversie, "
        "redactare, parole, semnături și organizarea paginilor."
    ),
    page_type="CollectionPage",
)
SEO_PAGES.update(
    {
        tool.url_name: SeoPage(
            url_name=tool.url_name,
            title_en=tool.seo_title.en,
            title_ro=tool.seo_title.ro,
            description_en=tool.seo_description.en,
            description_ro=tool.seo_description.ro,
        )
        for tool in TOOL_LANDINGS
    }
)

_DEFAULT_DESCRIPTION = {
    "en": "Private, practical tools for editing and processing PDF documents online.",
    "ro": "Instrumente practice pentru editarea și procesarea documentelor PDF online.",
}
_INDEX_ROBOTS = "index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1"
NOINDEX_ROBOTS = "noindex, nofollow, nosnippet"


def _language() -> str:
    return "ro" if (get_language() or "en").split("-")[0].lower() == "ro" else "en"


def public_origin() -> str:
    """Return a validated origin from settings, never from the Host header."""

    raw = str(getattr(settings, "SITE_URL", "")).strip()
    parsed = urlsplit(raw)
    has_root_path = parsed.path in ("", "/")
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or not has_root_path
        or parsed.query
        or parsed.fragment
    ):
        raise ImproperlyConfigured(
            "SITE_URL must be an http(s) origin without credentials, path, query or fragment"
        )
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, "", "", "")).rstrip("/")


def _url_for(page: SeoPage) -> str:
    return f"{public_origin()}{reverse(page.url_name)}"


def _resolver_name(request: HttpRequest) -> str:
    match = getattr(request, "resolver_match", None)
    return str(match.url_name or "") if match else ""


def is_indexable_request(request: HttpRequest) -> bool:
    """Fail closed: only reviewed, safe GET/HEAD responses may be indexed."""

    if request.method not in {"GET", "HEAD"}:
        return False
    page = SEO_PAGES.get(_resolver_name(request))
    if page is None:
        return False
    if page.anonymous_only:
        user = getattr(request, "user", None)
        if user is not None and user.is_authenticated:
            return False
        session = getattr(request, "session", None)
        if session is not None and session.session_key:
            return False
    return True


def _json_for_script(data: object) -> str:
    """Serialize JSON-LD without allowing a value to close the script tag."""

    payload = json.dumps(data, cls=DjangoJSONEncoder, ensure_ascii=False, separators=(",", ":"))
    return (
        payload.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _structured_data(page: SeoPage, *, language: str, canonical: str) -> dict:
    origin = public_origin()
    website = {
        "@type": "WebSite",
        "@id": f"{origin}/#website",
        "url": f"{origin}/",
        "name": "PDF Platform",
        "inLanguage": language,
    }
    webpage: dict[str, object] = {
        "@type": page.page_type,
        "@id": f"{canonical}#webpage",
        "url": canonical,
        "name": page.title(language),
        "description": page.description(language),
        "inLanguage": language,
        "isPartOf": {"@id": website["@id"]},
    }
    graph: list[dict[str, object]] = [website, webpage]
    if page.url_name == "dashboard":
        webpage.update(
            {
                "applicationCategory": "UtilitiesApplication",
                "operatingSystem": "Any",
                "browserRequirements": "Requires JavaScript and a modern web browser",
                "isAccessibleForFree": True,
                "offers": {"@type": "Offer", "price": 0},
                "featureList": [
                    "Edit PDF text",
                    "Merge and split PDFs",
                    "Compress PDFs",
                    "Convert PDF and image formats",
                    "OCR scanned documents",
                    "Protect and digitally sign PDFs",
                    "Redact sensitive text",
                    "Reorder, rotate and crop pages",
                ],
            }
        )
    elif page.url_name == "tools_hub":
        hub_items = [(tool.name.get(language), f"{origin}{reverse(tool.url_name)}") for tool in TOOL_LANDINGS]
        hub_items.extend(
            (tool.name.get(language), f"{origin}{reverse(tool.url_name)}") for tool in DIRECT_TOOL_LINKS
        )
        webpage["mainEntity"] = {
            "@type": "ItemList",
            "numberOfItems": len(hub_items),
            "itemListElement": [
                {
                    "@type": "ListItem",
                    "position": position,
                    "name": name,
                    "url": url,
                }
                for position, (name, url) in enumerate(hub_items, start=1)
            ],
        }
    elif tool := TOOL_LANDINGS_BY_URL_NAME.get(page.url_name):
        breadcrumb_id = f"{canonical}#breadcrumb"
        webpage["breadcrumb"] = {"@id": breadcrumb_id}
        webpage["mainEntity"] = {
            "@type": "HowTo",
            "name": tool.name.get(language),
            "description": tool.intro.get(language),
            "step": [
                {
                    "@type": "HowToStep",
                    "position": position,
                    "name": step.get(language),
                    "text": step.get(language),
                }
                for position, step in enumerate(tool.steps, start=1)
            ],
        }
        graph.append(
            {
                "@type": "BreadcrumbList",
                "@id": breadcrumb_id,
                "itemListElement": [
                    {
                        "@type": "ListItem",
                        "position": 1,
                        "name": "PDF Platform",
                        "item": f"{origin}/",
                    },
                    {
                        "@type": "ListItem",
                        "position": 2,
                        "name": "Instrumente PDF" if language == "ro" else "PDF tools",
                        "item": f"{origin}{reverse('tools_hub')}",
                    },
                    {
                        "@type": "ListItem",
                        "position": 3,
                        "name": tool.name.get(language),
                        "item": canonical,
                    },
                ],
            }
        )
    return {"@context": "https://schema.org", "@graph": graph}


def seo_context(request: HttpRequest) -> dict[str, dict[str, object]]:
    """Template context for metadata, canonical URLs and JSON-LD."""

    language = _language()
    page = SEO_PAGES.get(_resolver_name(request))
    indexable = page is not None and is_indexable_request(request)
    title = page.title(language) if page else "PDF Platform"
    description = page.description(language) if page else _DEFAULT_DESCRIPTION[language]
    seo: dict[str, object] = {
        "title": title,
        "description": description,
        "robots": _INDEX_ROBOTS if indexable else NOINDEX_ROBOTS,
        "canonical": "",
        "image": "",
        "image_alt": "PDF Platform",
        "locale": "ro_RO" if language == "ro" else "en_US",
        "json_ld": "",
    }
    if indexable and page is not None:
        canonical = _url_for(page)
        seo.update(
            {
                "canonical": canonical,
                "image": f"{public_origin()}/static/images/logo.jpg",
                "json_ld": mark_safe(
                    _json_for_script(_structured_data(page, language=language, canonical=canonical))
                ),
            }
        )
    return {"seo": seo}


@require_safe
def robots_txt(request: HttpRequest) -> HttpResponse:
    """Advertise the canonical sitemap and avoid wasting crawl budget."""

    sitemap_url = f"{public_origin()}{reverse('sitemap_xml')}"
    body = "\n".join(
        (
            "User-agent: *",
            "Allow: /",
            "Disallow: /api/",
            "Disallow: /media/",
            "Disallow: /metrics",
            "Disallow: /healthz",
            "Disallow: /readyz",
            "Disallow: /csp-report/",
            "Disallow: /jsi18n/",
            f"Sitemap: {sitemap_url}",
            "",
        )
    )
    response = HttpResponse(body, content_type="text/plain; charset=utf-8")
    response["Cache-Control"] = "public, max-age=3600"
    return response


@require_safe
def sitemap_xml(request: HttpRequest) -> HttpResponse:
    """Return only reviewed canonical URLs, built from the same SEO registry."""

    namespace = "http://www.sitemaps.org/schemas/sitemap/0.9"
    urlset = ElementTree.Element("urlset", xmlns=namespace)
    for page in SEO_PAGES.values():
        url = ElementTree.SubElement(urlset, "url")
        ElementTree.SubElement(url, "loc").text = _url_for(page)
    payload = ElementTree.tostring(urlset, encoding="utf-8", xml_declaration=True)
    response = HttpResponse(payload, content_type="application/xml; charset=utf-8")
    response["Cache-Control"] = "public, max-age=3600"
    return response
