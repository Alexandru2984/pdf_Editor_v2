"""``X-API-Key`` header authentication for the REST API."""

from django.utils import timezone
from drf_spectacular.extensions import OpenApiAuthenticationExtension
from rest_framework import authentication, exceptions

from ..models import ApiKey


class ApiKeyAuthentication(authentication.BaseAuthentication):
    """Authenticate a request via the ``X-API-Key`` header.

    The header value is hashed and matched against ``ApiKey.key_hash``.
    Revoked keys are rejected. On success the matched key is attached to
    ``request.auth`` so views can access it (e.g. for per-key throttling).
    """

    keyword = "X-API-Key"

    def authenticate(self, request):
        raw = request.META.get("HTTP_X_API_KEY", "").strip()
        if not raw:
            return None  # let other auth classes try

        key_hash = ApiKey.hash_token(raw)
        try:
            api_key = ApiKey.objects.select_related("user").get(key_hash=key_hash)
        except ApiKey.DoesNotExist as exc:
            raise exceptions.AuthenticationFailed("Invalid API key.") from exc

        if api_key.revoked_at is not None:
            raise exceptions.AuthenticationFailed("API key revoked.")
        if not api_key.user.is_active:
            raise exceptions.AuthenticationFailed("User is inactive.")

        # Update last_used_at without bumping auto_now timestamps elsewhere.
        ApiKey.objects.filter(pk=api_key.pk).update(last_used_at=timezone.now())
        return (api_key.user, api_key)

    def authenticate_header(self, request):
        return self.keyword


class ApiKeyAuthenticationScheme(OpenApiAuthenticationExtension):
    """Make Swagger UI recognise our X-API-Key header and show an Authorize button."""

    target_class = "pdfeditor.api.auth.ApiKeyAuthentication"
    name = "ApiKeyAuth"

    def get_security_definition(self, auto_schema):
        return {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "Per-user API key. Create one from your profile page.",
        }
