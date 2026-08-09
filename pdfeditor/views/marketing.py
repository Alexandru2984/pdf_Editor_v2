"""Public, read-only marketing pages for the PDF tool catalogue."""

from __future__ import annotations

from urllib.parse import urlencode

from django.conf import settings
from django.http import Http404, HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import translation
from django.utils.cache import patch_vary_headers
from django.utils.translation import get_language
from django.views.decorators.http import require_safe

from ..tool_catalog import DIRECT_TOOL_LINKS, TOOL_LANDINGS, TOOL_LANDINGS_BY_SLUG

_UI = {
    "en": {
        "hub_eyebrow": "PDF toolbox",
        "hub_title": "Choose the right PDF tool",
        "hub_subtitle": (
            "Clear, focused tools for converting, organizing, securing and extracting PDF content. "
            "No account is required for the basic workflow."
        ),
        "search_label": "Find a PDF tool",
        "search_placeholder": "Search merge, OCR, password…",
        "all_categories": "All categories",
        "open_tool": "Open guide",
        "tools_found": "tools shown",
        "no_results": "No tools match that search. Try a shorter term or another category.",
        "breadcrumb_home": "Home",
        "breadcrumb_tools": "PDF tools",
        "why_title": "What this tool gives you",
        "steps_title": "How to use it",
        "note_title": "Good to know before you start",
        "faq_title": "Frequently asked questions",
        "related_title": "Related PDF tools",
        "upload_title": "Start with your PDF",
        "upload_multiple_title": "Start with your PDF files",
        "upload_help": "Select a PDF up to {max_mb} MB and {max_pages} pages.",
        "upload_multiple_help": "Select at least two PDFs, up to {max_mb} MB and {max_pages} pages each.",
        "choose_pdf": "Choose PDF",
        "choose_pdfs": "Choose PDFs",
        "continue_tool": "Upload and continue",
        "already_uploaded": "Already uploaded your file?",
        "open_operation": "Open the tool",
        "privacy_one": "No registration required",
        "privacy_two": "Validated and sanitized uploads",
        "privacy_three": "Files removed automatically within {hours} hours",
        "direct_badge": "Ready to use",
    },
    "ro": {
        "hub_eyebrow": "Trusă PDF",
        "hub_title": "Alege instrumentul PDF potrivit",
        "hub_subtitle": (
            "Instrumente clare pentru conversia, organizarea, securizarea și extragerea conținutului PDF. "
            "Fluxul de bază nu necesită cont."
        ),
        "search_label": "Găsește un instrument PDF",
        "search_placeholder": "Caută unire, OCR, parolă…",
        "all_categories": "Toate categoriile",
        "open_tool": "Deschide ghidul",
        "tools_found": "instrumente afișate",
        "no_results": "Niciun instrument nu corespunde. Încearcă un termen mai scurt sau altă categorie.",
        "breadcrumb_home": "Acasă",
        "breadcrumb_tools": "Instrumente PDF",
        "why_title": "Ce îți oferă instrumentul",
        "steps_title": "Cum îl folosești",
        "note_title": "Bine de știut înainte să începi",
        "faq_title": "Întrebări frecvente",
        "related_title": "Instrumente PDF asociate",
        "upload_title": "Începe cu PDF-ul tău",
        "upload_multiple_title": "Începe cu fișierele tale PDF",
        "upload_help": "Alege un PDF de maximum {max_mb} MB și {max_pages} pagini.",
        "upload_multiple_help": "Alege cel puțin două PDF-uri, fiecare de maximum {max_mb} MB și {max_pages} pagini.",
        "choose_pdf": "Alege PDF",
        "choose_pdfs": "Alege PDF-uri",
        "continue_tool": "Încarcă și continuă",
        "already_uploaded": "Ai încărcat deja fișierul?",
        "open_operation": "Deschide instrumentul",
        "privacy_one": "Nu necesită înregistrare",
        "privacy_two": "Fișiere validate și sanitizate",
        "privacy_three": "Fișiere șterse automat în maximum {hours} ore",
        "direct_badge": "Gata de utilizare",
    },
}


def marketing_language() -> str:
    return "ro" if (get_language() or "en").split("-")[0].lower() == "ro" else "en"


def _ui(language: str) -> dict[str, str]:
    cleanup_hours = int(getattr(settings, "PDF_CLEANUP_HOURS", 24))
    return {
        key: value.format(
            max_mb=int(getattr(settings, "PDF_MAX_UPLOAD_BYTES", 10 * 1024 * 1024)) // (1024 * 1024),
            max_pages=int(getattr(settings, "PDF_MAX_PAGES", 500)),
            hours=cleanup_hours,
        )
        for key, value in _UI[language].items()
    }


def _hub_cards(language: str) -> list[dict[str, object]]:
    cards: list[dict[str, object]] = []
    for tool in TOOL_LANDINGS:
        card = tool.localized(language)
        card["url"] = reverse(tool.localized_url_name(language))
        card["direct"] = False
        cards.append(card)
    for direct in DIRECT_TOOL_LINKS:
        card = direct.localized(language)
        card["url"] = reverse(direct.url_name)
        card["direct"] = True
        cards.append(card)
    return cards


def _language_redirect(url_name: str) -> HttpResponse:
    """Redirect without letting a shared cache reuse another user's locale."""

    response = redirect(url_name, permanent=False)
    response["Cache-Control"] = "private, no-store"
    patch_vary_headers(response, ("Accept-Language", "Cookie"))
    return response


@require_safe
def tools_hub_view(request: HttpRequest, language: str | None = None) -> HttpResponse:
    if language not in {"en", "ro"}:
        selected_language = marketing_language()
        return _language_redirect(f"tools_hub_{selected_language}")

    with translation.override(language):
        cards = _hub_cards(language)
        categories = sorted({str(card["category"]) for card in cards})
        response = render(
            request,
            "pdfeditor/marketing/tools_hub.html",
            {
                "cards": cards,
                "categories": categories,
                "ui": _ui(language),
            },
        )
    response["Content-Language"] = language
    return response


@require_safe
def tool_landing_view(
    request: HttpRequest,
    tool_slug: str,
    language: str | None = None,
) -> HttpResponse:
    definition = TOOL_LANDINGS_BY_SLUG.get(tool_slug)
    if definition is None:
        raise Http404

    if language not in {"en", "ro"}:
        selected_language = marketing_language()
        return _language_redirect(definition.localized_url_name(selected_language))

    with translation.override(language):
        tool = definition.localized(language)
        related_definitions = [
            candidate
            for candidate in TOOL_LANDINGS
            if candidate.key != definition.key
            and candidate.category.get(language) == definition.category.get(language)
        ]
        related_definitions.extend(
            candidate
            for candidate in TOOL_LANDINGS
            if candidate.key != definition.key and candidate not in related_definitions
        )
        related = []
        for candidate in related_definitions[:3]:
            item = candidate.localized(language)
            item["url"] = reverse(candidate.localized_url_name(language))
            related.append(item)

        upload_query = urlencode({"next": definition.operation_url_name})
        response = render(
            request,
            "pdfeditor/marketing/tool_landing.html",
            {
                "tool": tool,
                "ui": _ui(language),
                "related_tools": related,
                "tools_hub_url": reverse(f"tools_hub_{language}"),
                "upload_action": f"{reverse('upload')}?{upload_query}",
                "operation_url": reverse(definition.operation_url_name),
            },
        )
    response["Content-Language"] = language
    return response
