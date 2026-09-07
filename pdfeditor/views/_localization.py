"""Helpers for public routes whose language is fixed by the URL."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from django.http import Http404, HttpRequest, HttpResponse
from django.shortcuts import redirect
from django.utils import translation
from django.utils.cache import patch_vary_headers
from django.utils.translation import get_language

PUBLIC_LANGUAGES = frozenset({"en", "ro"})


def negotiated_language() -> str:
    """The language LocaleMiddleware picked, restricted to what marketing pages ship.

    Falls back to "en", matching every other marketing-language helper in
    this codebase (`seo._language`, `marketing.marketing_language`,
    `legal._lang_suffix`) — and `settings.LANGUAGE_CODE` under the test
    runner, so an unmatched Accept-Language behaves the same in tests and in
    production (where LocaleMiddleware itself already falls back to "ro").
    """

    language = (get_language() or "").split("-")[0].lower()
    return language if language in PUBLIC_LANGUAGES else "en"


def language_redirect(target_url_name: str) -> HttpResponse:
    """Permanently redirect to a language-fixed URL without letting a shared
    cache reuse another visitor's negotiated locale.

    Permanent (301): several of these bare routes replaced a URL search
    engines had already indexed (e.g. "/privacy/") once per-language routing
    shipped. A 302 does not let a crawler consolidate that page's existing
    ranking signal onto the new URL the way a 301 does.
    """

    response = redirect(target_url_name, permanent=True)
    response["Cache-Control"] = "private, no-store"
    patch_vary_headers(response, ("Accept-Language", "Cookie"))
    return response


def _is_fresh_anonymous_visit(request: HttpRequest) -> bool:
    user = getattr(request, "user", None)
    if user is not None and user.is_authenticated:
        return False
    session = getattr(request, "session", None)
    return not (session is not None and session.session_key)


def localized_public_view(
    url_name: str, *, redirect_only_fresh_visits: bool = False
) -> Callable[[Callable[..., HttpResponse]], Callable[..., HttpResponse]]:
    """Honor an optional URL language, and permanently redirect the legacy,
    unprefixed route to it (see `language_redirect`).

    Generic legacy routes omit ``language``; canonical localized aliases pass
    it explicitly and receive a matching Content-Language response. By
    default a safe (GET/HEAD) request to the legacy route always redirects
    to the negotiated language. Pass ``redirect_only_fresh_visits=True`` for
    a view whose legacy route also serves a signed-in or session-scoped
    visitor (e.g. the dashboard) — that visitor keeps working at the legacy
    URL instead of an extra hop on every in-app link back to it.
    """

    def decorator(view: Callable[..., HttpResponse]) -> Callable[..., HttpResponse]:
        @wraps(view)
        def wrapper(
            request: HttpRequest,
            *args: Any,
            language: str | None = None,
            **kwargs: Any,
        ) -> HttpResponse:
            if language is None:
                if request.method in {"GET", "HEAD"} and (
                    not redirect_only_fresh_visits or _is_fresh_anonymous_visit(request)
                ):
                    return language_redirect(f"{url_name}_{negotiated_language()}")
                return view(request, *args, **kwargs)
            if language not in PUBLIC_LANGUAGES:
                raise Http404
            with translation.override(language):
                response = view(request, *args, **kwargs)
            response["Content-Language"] = language
            return response

        return wrapper

    return decorator
