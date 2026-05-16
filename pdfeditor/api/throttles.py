"""Scope-aware throttling for the REST API.

The throttle picks its rate at request time based on two axes:

* ``auth_method`` — ``api_key`` (X-API-Key), ``user`` (session), ``anon``.
* ``throttle_scope_category`` — ``read``, ``op`` or ``upload``; set by the
  view (a viewset can change it per-action via ``get_throttles``).

So a single throttle handles all 9 combinations (e.g. ``api_key_upload``,
``anon_read``). Rates live in ``settings.REST_FRAMEWORK["DEFAULT_THROTTLE_RATES"]``;
a missing key means "unlimited for this combination".
"""

from __future__ import annotations

from rest_framework.throttling import SimpleRateThrottle

from ..models import ApiKey

_DEFAULT_CATEGORY = "op"
_VALID_CATEGORIES = frozenset({"read", "op", "upload"})


class ScopedAuthAwareThrottle(SimpleRateThrottle):
    """One throttle, nine scopes — chosen per request from auth method × category."""

    # SimpleRateThrottle.__init__ tries to read self.scope and parse a rate
    # from settings at construction time. We don't know the scope until
    # allow_request() runs (it depends on the request's auth state and the
    # view's category), so defer all setup until then.
    scope = "scoped"

    def __init__(self):  # noqa: D401 — see class docstring
        pass

    @staticmethod
    def _auth_method(request) -> str:
        if isinstance(getattr(request, "auth", None), ApiKey):
            return "api_key"
        user = getattr(request, "user", None)
        if user is not None and getattr(user, "is_authenticated", False):
            return "user"
        return "anon"

    def _category(self, view) -> str:
        category = getattr(view, "throttle_scope_category", _DEFAULT_CATEGORY)
        if category not in _VALID_CATEGORIES:
            category = _DEFAULT_CATEGORY
        return category

    def _identity(self, request) -> str | int | None:
        method = self._auth_method(request)
        if method == "api_key":
            return request.auth.pk
        if method == "user":
            return request.user.pk
        return self.get_ident(request)

    def get_cache_key(self, request, view):
        ident = self._identity(request)
        if ident is None:
            return None
        return self.cache_format % {"scope": self.scope, "ident": ident}

    def allow_request(self, request, view):
        self.scope = f"{self._auth_method(request)}_{self._category(view)}"
        self.rate = self.get_rate()
        # Missing rate = "unlimited for this combination" (e.g. you might
        # leave anon_read uncapped while still capping anon_upload tightly).
        if not self.rate:
            return True
        self.num_requests, self.duration = self.parse_rate(self.rate)

        self.key = self.get_cache_key(request, view)
        if self.key is None:
            return True

        self.history = self.cache.get(self.key, [])
        self.now = self.timer()
        while self.history and self.history[-1] <= self.now - self.duration:
            self.history.pop()
        if len(self.history) >= self.num_requests:
            return self.throttle_failure()
        return self.throttle_success()
