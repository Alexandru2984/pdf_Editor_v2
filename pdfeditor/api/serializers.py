"""DRF serializers for API responses."""

from drf_spectacular.utils import extend_schema_field
from rest_framework import serializers

from ..models import ApiKey, Job, ProcessedPDF, ShareLink, UploadedPDF, Webhook, WebhookDelivery

_SENSITIVE_PARAM_TOKENS = ("password", "secret", "token", "key")
_REDACTED = "[redacted]"


def redact_sensitive_params(value):
    """Return a copy of a JSON-like value with sensitive keys redacted."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            key_s = str(key)
            if any(token in key_s.lower() for token in _SENSITIVE_PARAM_TOKENS):
                out[key] = _REDACTED
            else:
                out[key] = redact_sensitive_params(item)
        return out
    if isinstance(value, list):
        return [redact_sensitive_params(item) for item in value]
    return value


class UploadedPDFSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedPDF
        fields = ["id", "name", "size", "uploaded_at"]
        read_only_fields = fields


class ProcessedPDFSerializer(serializers.ModelSerializer):
    kind_display = serializers.CharField(source="get_kind_display", read_only=True)
    download_url = serializers.SerializerMethodField()
    source_id = serializers.PrimaryKeyRelatedField(source="source", read_only=True)

    class Meta:
        model = ProcessedPDF
        fields = ["id", "kind", "kind_display", "name", "size", "created_at", "source_id", "download_url"]
        read_only_fields = fields

    @extend_schema_field(serializers.URLField(allow_null=True))
    def get_download_url(self, obj) -> str | None:
        request = self.context.get("request")
        if request is None:
            return None
        from django.urls import reverse

        return request.build_absolute_uri(reverse("api:output-download", args=[obj.id]))


class ShareLinkSerializer(serializers.ModelSerializer):
    processed_pdf_id = serializers.UUIDField()
    url = serializers.SerializerMethodField()

    class Meta:
        model = ShareLink
        fields = [
            "id",
            "processed_pdf_id",
            "token",
            "url",
            "expires_at",
            "max_downloads",
            "download_count",
            "created_at",
        ]
        read_only_fields = ["id", "token", "url", "download_count", "created_at"]

    def get_url(self, obj):
        from django.urls import reverse

        request = self.context.get("request")
        path = reverse("public_share_download", args=[obj.token])
        return request.build_absolute_uri(path) if request else path


class JobSerializer(serializers.ModelSerializer):
    output_id = serializers.PrimaryKeyRelatedField(source="output", read_only=True)
    output_download_url = serializers.SerializerMethodField()
    is_terminal = serializers.SerializerMethodField()
    params = serializers.SerializerMethodField()

    class Meta:
        model = Job
        fields = [
            "id",
            "kind",
            "status",
            "is_terminal",
            "progress",
            "error_message",
            "output_id",
            "output_download_url",
            "params",
            "created_at",
            "started_at",
            "finished_at",
        ]
        read_only_fields = fields

    @extend_schema_field(serializers.BooleanField())
    def get_is_terminal(self, obj) -> bool:
        return obj.is_terminal()

    @extend_schema_field(serializers.JSONField())
    def get_params(self, obj):
        return redact_sensitive_params(obj.params or {})

    @extend_schema_field(serializers.URLField(allow_null=True))
    def get_output_download_url(self, obj) -> str | None:
        if not obj.output_id:
            return None
        request = self.context.get("request")
        if request is None:
            return None
        from django.urls import reverse

        return request.build_absolute_uri(reverse("api:output-download", args=[obj.output_id]))


class ApiKeySerializer(serializers.ModelSerializer):
    """Returned without the plaintext token — only on creation does the view
    include the secret in the response payload."""

    class Meta:
        model = ApiKey
        fields = ["id", "label", "prefix", "last_used_at", "revoked_at", "created_at"]
        read_only_fields = fields


class WebhookSerializer(serializers.ModelSerializer):
    """A user's webhook endpoint. ``secret`` is read-only but always returned —
    it's the caller's own HMAC signing key (over authenticated TLS), and API
    clients need it to verify deliveries; unlike API-key tokens it isn't a
    credential that grants access, so idempotent retrieval is the friendlier
    contract."""

    class Meta:
        model = Webhook
        fields = [
            "id",
            "url",
            "description",
            "secret",
            "is_active",
            "created_at",
            "last_triggered_at",
            "last_status",
            "failure_count",
        ]
        read_only_fields = [
            "id",
            "secret",
            "created_at",
            "last_triggered_at",
            "last_status",
            "failure_count",
        ]

    def validate_url(self, value: str) -> str:
        # Same anti-SSRF gate as the web UI: public https only, checked on every
        # write (create or PATCH), re-checked again at delivery time.
        from .. import webhooks

        try:
            webhooks.validate_webhook_url(value)
        except webhooks.InvalidWebhookURL as exc:
            raise serializers.ValidationError(str(exc)) from exc
        return value


class WebhookDeliverySerializer(serializers.ModelSerializer):
    """One terminal delivery attempt in a webhook's history."""

    class Meta:
        model = WebhookDelivery
        fields = ["id", "event", "ok", "status", "created_at"]
        read_only_fields = fields
