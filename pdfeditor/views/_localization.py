"""Helpers for public routes whose language is fixed by the URL."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from django.http import Http404, HttpRequest, HttpResponse
from django.utils import translation

PUBLIC_LANGUAGES = frozenset({"en", "ro"})


def localized_public_view(view: Callable[..., HttpResponse]) -> Callable[..., HttpResponse]:
    """Honor an optional URL language without changing the user's cookie.

    Generic legacy routes omit ``language`` and continue to use
    ``LocaleMiddleware``. Canonical localized aliases pass it explicitly and
    receive a matching Content-Language response.
    """

    @wraps(view)
    def wrapper(
        request: HttpRequest,
        *args: Any,
        language: str | None = None,
        **kwargs: Any,
    ) -> HttpResponse:
        if language is None:
            return view(request, *args, **kwargs)
        if language not in PUBLIC_LANGUAGES:
            raise Http404

        with translation.override(language):
            response = view(request, *args, **kwargs)
        response["Content-Language"] = language
        return response

    return wrapper
