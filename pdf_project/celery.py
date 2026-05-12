"""Celery application for the pdf_project Django project.

Tasks live in ``pdfeditor/tasks.py`` and are auto-discovered. Configuration
comes from Django settings under the ``CELERY_*`` namespace.
"""

import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pdf_project.settings")

app = Celery("pdf_project")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
