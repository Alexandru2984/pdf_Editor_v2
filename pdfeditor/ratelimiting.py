"""Auth-aware rate limiting.

When the request is authenticated, quotas are keyed by ``user.pk`` (so the
same user is throttled regardless of which IP they roam through); otherwise
keyed by remote IP. Authenticated users get higher quotas because we
actually know who they are — making abuse traceable and remediable.

Two helpers:

* ``auth_aware_ratelimit`` — decorator for sync views.
* ``check_rate_limit`` — coroutine-safe core for async views (use it inside
  ``async def`` handlers via ``await sync_to_async(check_rate_limit)(...)``).

Both share the same key derivation (``_compute_key``) so anonymous and
authenticated quotas never alias.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from django.http import HttpRequest
from django_ratelimit.core import is_ratelimited
from django_ratelimit.exceptions import Ratelimited

from .netutils import client_ip


def _compute_key_and_rate(request: HttpRequest, *, anon_rate: str, user_rate: str) -> tuple[str, str]:
    user = getattr(request, "user", None)
    try:
        if user is not None and user.is_authenticated:
            return f"user:{user.pk}", user_rate
    except Exception:  # noqa: BLE001
        # Lazy user resolution can fail on stale/corrupt sessions; rate
        # limiting must degrade to the anonymous bucket, not break the view.
        user = None
    # Real client IP via netutils.client_ip (reads X-Forwarded-For with the
    # trusted-proxy count) — NOT raw REMOTE_ADDR, which behind nginx is the
    # proxy's own IP and would funnel every anon user into one shared bucket.
    # "noip" keeps a missing address (e.g. test request factories) in its own
    # slot rather than clustering all such traffic together.
    ip = client_ip(request) or "noip"
    return f"ip:{ip}", anon_rate


def check_rate_limit(
    request: HttpRequest,
    *,
    group: str,
    anon_rate: str,
    user_rate: str,
    method: str = "POST",
    increment: bool = True,
) -> bool:
    """Return True iff the request is over-quota. Suitable for async views."""
    key_str, rate = _compute_key_and_rate(request, anon_rate=anon_rate, user_rate=user_rate)
    return is_ratelimited(
        request=request,
        group=group,
        key=lambda _g, _r: key_str,
        rate=rate,
        method=method,
        increment=increment,
    )


def auth_aware_ratelimit(
    *,
    anon_rate: str,
    user_rate: str,
    method: str = "POST",
    block: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: anonymous requesters → ``anon_rate`` per IP, authenticated → ``user_rate`` per user."""

    def decorator(view: Callable[..., Any]) -> Callable[..., Any]:
        group = f"{view.__module__}.{view.__qualname__}"

        @wraps(view)
        def wrapper(request: HttpRequest, *args: Any, **kwargs: Any) -> Any:
            if check_rate_limit(
                request,
                group=group,
                anon_rate=anon_rate,
                user_rate=user_rate,
                method=method,
            ):
                if block:
                    raise Ratelimited()
                request.limited = True
            return view(request, *args, **kwargs)

        return wrapper

    return decorator
