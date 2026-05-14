from django.apps import AppConfig


class PdfeditorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "pdfeditor"

    def ready(self) -> None:
        from . import signals  # noqa: F401 — register post_delete receivers
