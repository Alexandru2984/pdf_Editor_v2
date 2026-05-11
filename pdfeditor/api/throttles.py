"""Per-API-key throttling for the REST API."""

from rest_framework.throttling import SimpleRateThrottle

from ..models import ApiKey


class ApiKeyRateThrottle(SimpleRateThrottle):
    """Throttle requests by API key, falling through for session-authenticated
    or anonymous traffic so the default DRF throttles can handle them."""

    scope = "api_key"

    def get_cache_key(self, request, view):
        api_key = getattr(request, "auth", None)
        if not isinstance(api_key, ApiKey):
            return None
        return self.cache_format % {"scope": self.scope, "ident": api_key.pk}
