"""
Django settings for pdf_project project.
"""

import logging
import os
import sys
from pathlib import Path

import dj_database_url
import sentry_sdk
from dotenv import load_dotenv
from sentry_sdk.integrations.django import DjangoIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

TESTING = "test" in sys.argv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


def env_bool(name, default=False):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


SECRET_KEY = os.environ["SECRET_KEY"]

DEBUG = env_bool("DEBUG", False)

ALLOWED_HOSTS = [h.strip() for h in os.environ.get("ALLOWED_HOSTS", "").split(",") if h.strip()]

CSRF_TRUSTED_ORIGINS = [
    "https://pdf.micutu.com",
    "http://127.0.0.1:8001",
    "http://localhost:8001",
]

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Upload limits (10 MB, 500 pages)
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024
PDF_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
PDF_MAX_PAGES = int(os.environ.get("PDF_MAX_PAGES", 500))

# Per-owner storage quota for uploaded PDFs (sum of file sizes). 0 = unlimited.
# Default: anon 50 MB, authenticated 500 MB. Overridable via env.
PDF_QUOTA_ANON_BYTES = int(os.environ.get("PDF_QUOTA_ANON_BYTES", 50 * 1024 * 1024))
PDF_QUOTA_USER_BYTES = int(os.environ.get("PDF_QUOTA_USER_BYTES", 500 * 1024 * 1024))

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "axes",
    "rest_framework",
    "drf_spectacular",
    "django_prometheus",
    "pdfeditor",
]

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "pdfeditor.api.auth.ApiKeyAuthentication",
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    # Throttle classes — empty list when RATELIMIT_ENABLE=0 (load testing).
    "DEFAULT_THROTTLE_CLASSES": (
        [
            "pdfeditor.api.throttles.ApiKeyRateThrottle",
            "rest_framework.throttling.AnonRateThrottle",
        ]
        if os.environ.get("RATELIMIT_ENABLE", "1").lower() not in ("0", "false", "no")
        else []
    ),
    "DEFAULT_THROTTLE_RATES": {
        "anon": "20/hour",
        "api_key": "300/hour",
    },
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
}

# Celery: broker is Redis. Result backend uses a separate DB so task results
# don't churn the broker's keyspace. In tests, CELERY_TASK_ALWAYS_EAGER=True
# runs tasks inline (no worker needed) — set in tests via override_settings.
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TIMEZONE = "UTC"
# Acknowledge tasks AFTER they complete — if a worker crashes mid-task, the
# task gets redelivered to another worker rather than being silently lost.
CELERY_TASK_ACKS_LATE = True
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
# Hard kill at 10 min — protects against runaway tesseract/ghostscript subprocs.
CELERY_TASK_TIME_LIMIT = 600
CELERY_TASK_SOFT_TIME_LIMIT = 540
# In Django's test runner, run tasks inline so we don't need a real broker
# or worker container. Detected via DJANGO_TEST=1 (set by manage.py test).
if "test" in sys.argv:
    CELERY_TASK_ALWAYS_EAGER = True
    CELERY_TASK_EAGER_PROPAGATES = True

SPECTACULAR_SETTINGS = {
    "TITLE": "PDF Editor API",
    "DESCRIPTION": (
        "Programmatic access to the PDF Editor toolbox: upload PDFs, run "
        "operations (compress, OCR, sign, redact, PDF/A, …), and download "
        "the results. Authenticate by sending your API key in the "
        "`X-API-Key` header."
    ),
    "VERSION": "1.0.0",
    "SERVE_INCLUDE_SCHEMA": False,
    "COMPONENT_SPLIT_REQUEST": True,
    "SCHEMA_PATH_PREFIX": "/api/v1",
    "TAGS": [
        {"name": "PDFs", "description": "Upload and manage source PDFs"},
        {"name": "Operations", "description": "Run a PDF processing operation"},
        {"name": "Outputs", "description": "List and download processed PDFs"},
    ],
}

MIDDLEWARE = [
    # django-prometheus must wrap everything: Before starts the timer,
    # After records latency + per-view counters. Keep them as outermost.
    "django_prometheus.middleware.PrometheusBeforeMiddleware",
    "pdfeditor.middleware.RequestIDMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.locale.LocaleMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "axes.middleware.AxesMiddleware",
    "django_prometheus.middleware.PrometheusAfterMiddleware",
]

AUTHENTICATION_BACKENDS = [
    "axes.backends.AxesStandaloneBackend",
    "pdfeditor.auth_backends.CaseInsensitiveModelBackend",
]

# django-axes: lock account for 1h after 5 failed admin logins
AXES_ENABLED = not TESTING  # tests use Client.login() which can't pass a request to axes
AXES_FAILURE_LIMIT = 5
AXES_COOLOFF_TIME = 1
AXES_LOCKOUT_PARAMETERS = ["ip_address", "username"]
AXES_RESET_ON_SUCCESS = True
# We sit behind nginx — without these, ipware reads REMOTE_ADDR (the proxy
# itself) and every lockout records ip=None / 127.0.0.1.
AXES_BEHIND_REVERSE_PROXY = True
AXES_IPWARE_PROXY_COUNT = 1

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "request_id": {"()": "pdfeditor.middleware.RequestIDLogFilter"},
    },
    "formatters": {
        "structured": {
            "format": "%(asctime)s %(levelname)s [req=%(request_id)s] %(name)s %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "structured",
            "filters": ["request_id"],
        },
    },
    "root": {"handlers": ["console"], "level": "INFO"},
    "loggers": {
        "pdfeditor": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "axes": {"handlers": ["console"], "level": "WARNING", "propagate": False},
    },
}

ROOT_URLCONF = "pdf_project.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        # Project templates win over third-party app templates (notably
        # django.contrib.admin's registration/* templates, which would
        # otherwise shadow the project's custom auth/email templates).
        "DIRS": [BASE_DIR / "pdfeditor" / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.template.context_processors.i18n",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "pdf_project.wsgi.application"

# DATABASES — controlled by DATABASE_URL when set (Postgres, MySQL, etc.),
# otherwise falls back to local SQLite at db.sqlite3 / DATABASE_PATH.
_database_url = os.environ.get("DATABASE_URL")
if _database_url:
    DATABASES = {
        "default": dj_database_url.parse(_database_url, conn_max_age=600, conn_health_checks=True),
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": os.environ.get("DATABASE_PATH", str(BASE_DIR / "db.sqlite3")),
        },
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
    # Block passwords that appear in HIBP's public breach corpus. Fails
    # open (allow) on network errors so an HIBP outage doesn't block sign-up.
    {"NAME": "pdfeditor.password_validators.PwnedPasswordValidator"},
]

# Tests assert against English msgids; force English when running them.
LANGUAGE_CODE = "en" if "test" in sys.argv else "ro"
LANGUAGES = [
    ("ro", "Română"),
    ("en", "English"),
]
LOCALE_PATHS = [BASE_DIR / "locale"]
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "static"]

PDF_CLEANUP_HOURS = 24

# RFC 3161 timestamp authority used when signing with `add_timestamp` toggled.
# freetsa.org is free, public, and supports anonymous requests over HTTPS.
PDF_SIGN_TSA_URL = os.getenv("PDF_SIGN_TSA_URL", "https://freetsa.org/tsr")

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Auth flow URLs.
LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "dashboard"
LOGOUT_REDIRECT_URL = "dashboard"

# Email backend — console for dev (prints emails to stdout), SMTP for prod.
# Override EMAIL_BACKEND in .env for prod (e.g. django.core.mail.backends.smtp.EmailBackend).
EMAIL_BACKEND = os.environ.get(
    "EMAIL_BACKEND",
    "django.core.mail.backends.console.EmailBackend"
    if DEBUG
    else "django.core.mail.backends.smtp.EmailBackend",
)
# Accept both Django-standard EMAIL_* names and the shorter SMTP_* aliases.
EMAIL_HOST = os.environ.get("EMAIL_HOST") or os.environ.get("SMTP_HOST", "")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT") or os.environ.get("SMTP_PORT") or "587")
EMAIL_HOST_USER = os.environ.get("EMAIL_HOST_USER") or os.environ.get("SMTP_USER", "")
EMAIL_HOST_PASSWORD = os.environ.get("EMAIL_HOST_PASSWORD") or os.environ.get("SMTP_PASS", "")
EMAIL_USE_TLS = env_bool("EMAIL_USE_TLS", True)
DEFAULT_FROM_EMAIL = (
    os.environ.get("DEFAULT_FROM_EMAIL") or os.environ.get("SMTP_FROM") or "no-reply@pdf.micutu.com"
)
SERVER_EMAIL = os.environ.get("SERVER_EMAIL", DEFAULT_FROM_EMAIL)

# Public site URL — used in confirmation/reset emails for absolute links.
SITE_URL = os.environ.get("SITE_URL", "http://localhost:8000")

# Security hardening (prod behind HTTPS)
SECURE_SSL = env_bool("SECURE_SSL", not DEBUG) and not TESTING
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SESSION_COOKIE_SECURE = SECURE_SSL
CSRF_COOKIE_SECURE = SECURE_SSL
SECURE_SSL_REDIRECT = SECURE_SSL
SECURE_HSTS_SECONDS = 31536000 if SECURE_SSL else 0
SECURE_HSTS_INCLUDE_SUBDOMAINS = SECURE_SSL
SECURE_HSTS_PRELOAD = SECURE_SSL
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "same-origin"
X_FRAME_OPTIONS = "DENY"
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"

# Sentry — only initialised when SENTRY_DSN is set in the environment.
# Tests skip init explicitly to avoid spurious network attempts.
# Prometheus /metrics endpoint allowlist — comma-separated IPs that may
# scrape. Empty in DEBUG = open, empty in prod = closed (404). Default
# allows just localhost so an in-VPS Prometheus / Grafana Agent can hit
# it; for Grafana Cloud, add their pull endpoint to the env var.
_metrics_allow_raw = os.environ.get("PROMETHEUS_METRICS_ALLOW", "127.0.0.1")
PROMETHEUS_METRICS_ALLOW = {ip.strip() for ip in _metrics_allow_raw.split(",") if ip.strip()}

# django-ratelimit kill-switch. Default on; flip to "0"/"false" only for
# load testing or local dev. Don't disable in prod — the per-op anon
# quotas are how we keep upload/compress from getting hammered.
RATELIMIT_ENABLE = os.environ.get("RATELIMIT_ENABLE", "1").lower() not in ("0", "false", "no")

SENTRY_DSN = os.environ.get("SENTRY_DSN", "")
if SENTRY_DSN and not TESTING:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[
            DjangoIntegration(transaction_style="url"),
            LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
        ],
        traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
        send_default_pii=False,
        environment=os.environ.get("SENTRY_ENVIRONMENT", "production" if not DEBUG else "development"),
    )
