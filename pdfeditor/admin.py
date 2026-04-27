from django.contrib import admin

from .models import ProcessedPDF, UploadedPDF


@admin.register(UploadedPDF)
class UploadedPDFAdmin(admin.ModelAdmin):
    list_display = ("name", "size", "session_key_short", "uploaded_at")
    list_filter = ("uploaded_at",)
    search_fields = ("name", "session_key")
    readonly_fields = ("id", "uploaded_at")

    @admin.display(description="Session", ordering="session_key")
    def session_key_short(self, obj):
        return f"{obj.session_key[:8]}…" if obj.session_key else ""


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
