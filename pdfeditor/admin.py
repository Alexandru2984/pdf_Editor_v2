from django.contrib import admin

from .models import ApiKey, AuditLog, Job, ProcessedPDF, ShareLink, TrustAnchor, UploadedPDF


@admin.register(UploadedPDF)
class UploadedPDFAdmin(admin.ModelAdmin):
    list_display = ("name", "size", "session_key_short", "uploaded_at")
    list_filter = ("uploaded_at",)
    search_fields = ("name", "session_key")
    readonly_fields = ("id", "uploaded_at")

    @admin.display(description="Session", ordering="session_key")
    def session_key_short(self, obj):
        return f"{obj.session_key[:8]}…" if obj.session_key else ""


@admin.register(TrustAnchor)
class TrustAnchorAdmin(admin.ModelAdmin):
    list_display = ("name", "is_active", "added_by", "created_at")
    list_filter = ("is_active", "created_at")
    search_fields = ("name",)
    readonly_fields = ("id", "added_by", "created_at")

    def save_model(self, request, obj, form, change):
        if not change and not obj.added_by_id:
            obj.added_by = request.user
        super().save_model(request, obj, form, change)


@admin.register(ProcessedPDF)
class ProcessedPDFAdmin(admin.ModelAdmin):
    list_display = ("name", "kind", "size", "session_key_short", "created_at")
    list_filter = ("kind", "created_at")
    search_fields = ("name", "session_key")
    readonly_fields = ("id", "created_at")
    raw_id_fields = ("source",)

    @admin.display(description="Session", ordering="session_key")
    def session_key_short(self, obj):
        return f"{obj.session_key[:8]}…" if obj.session_key else ""


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ("created_at", "kind", "user", "output_name", "ip_address")
    list_filter = ("kind", "created_at")
    search_fields = (
        "user__username",
        "user__email",
        "session_key",
        "output_name",
        "source_name",
        "ip_address",
    )
    readonly_fields = (
        "id",
        "user",
        "session_key",
        "kind",
        "source_name",
        "output_name",
        "output_size",
        "ip_address",
        "user_agent",
        "created_at",
    )

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False


@admin.register(ShareLink)
class ShareLinkAdmin(admin.ModelAdmin):
    list_display = (
        "token_short",
        "processed_pdf",
        "creator",
        "download_count",
        "max_downloads",
        "expires_at",
        "created_at",
    )
    list_filter = ("created_at", "expires_at")
    search_fields = ("token", "creator__username", "session_key", "processed_pdf__name")
    readonly_fields = ("id", "token", "creator", "session_key", "created_at", "download_count")
    raw_id_fields = ("processed_pdf",)

    @admin.display(description="Token", ordering="token")
    def token_short(self, obj):
        return f"{obj.token[:10]}…" if obj.token else ""


@admin.register(ApiKey)
class ApiKeyAdmin(admin.ModelAdmin):
    list_display = ("label", "prefix", "user", "last_used_at", "revoked_at", "created_at")
    list_filter = ("created_at", "revoked_at")
    search_fields = ("label", "prefix", "user__username", "user__email")
    readonly_fields = ("id", "key_hash", "prefix", "last_used_at", "created_at")
    raw_id_fields = ("user",)

    def has_add_permission(self, request):
        # Keys must be created through the profile UI so the plaintext token
        # is shown to the user. Admin-side creation would leave the user
        # without the secret.
        return False


@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ("created_at", "kind", "status", "user", "progress", "duration")
    list_filter = ("status", "kind", "created_at")
    search_fields = ("user__username", "session_key", "kind", "error_message")
    readonly_fields = (
        "id",
        "user",
        "session_key",
        "kind",
        "source",
        "second_source",
        "output",
        "params",
        "created_at",
        "started_at",
        "finished_at",
    )
    raw_id_fields = ("source", "second_source", "output")

    @admin.display(description="Duration")
    def duration(self, obj):
        if obj.started_at and obj.finished_at:
            delta = obj.finished_at - obj.started_at
            return f"{delta.total_seconds():.1f}s"
        return "—"

    def has_add_permission(self, request):
        return False
