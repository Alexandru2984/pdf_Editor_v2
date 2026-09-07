"""Static legal pages: privacy policy + terms of service.

Each document is one full template per language instead of dozens of
gettext fragments — legal text reads, translates, and diffs as a
document. The view picks the variant matching the active language and
falls back to English for anything that isn't Romanian.
"""

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.utils.translation import get_language
from django.views.decorators.http import require_safe

from ._localization import localized_public_view

_SUPPORTED = ("ro", "en")


def _lang_suffix() -> str:
    lang = (get_language() or "en").split("-")[0]
    return lang if lang in _SUPPORTED else "en"


@localized_public_view("privacy")
@require_safe
def privacy_view(request: HttpRequest) -> HttpResponse:
    return render(request, f"pdfeditor/legal/privacy_{_lang_suffix()}.html")


@localized_public_view("terms")
@require_safe
def terms_view(request: HttpRequest) -> HttpResponse:
    return render(request, f"pdfeditor/legal/terms_{_lang_suffix()}.html")
