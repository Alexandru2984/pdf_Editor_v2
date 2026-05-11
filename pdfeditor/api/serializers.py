"""DRF serializers for API responses."""

from rest_framework import serializers

from ..models import ApiKey, ProcessedPDF, ShareLink, UploadedPDF


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

    def get_download_url(self, obj):
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


class ApiKeySerializer(serializers.ModelSerializer):
    """Returned without the plaintext token — only on creation does the view
    include the secret in the response payload."""

    class Meta:
        model = ApiKey
        fields = ["id", "label", "prefix", "last_used_at", "revoked_at", "created_at"]
        read_only_fields = fields
