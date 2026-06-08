FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps: tesseract for OCR (eng + ron language packs), ghostscript for
# PDF/A conversion, libgl/libglib for Pillow image ops, libgomp1 for the
# ONNX Runtime backend that powers fastembed (RAG embeddings).
#
# `apt-get upgrade` pulls security patches that the python:3.12-slim base
# image hasn't picked up yet — Debian ships fixes faster than the official
# python image rebuilds. Trivy gates the CI build on CRITICAL/HIGH OS CVEs,
# so skipping the upgrade means the next stale-base CVE breaks the build.
#
# APT_REFRESH_DATE is passed from CI as today's date so this layer is
# rebuilt at least daily — otherwise BuildKit reuses the cached layer
# with a stale `apt-get update` index and Trivy flags freshly-disclosed
# CVEs even though Debian has fixes available.
ARG APT_REFRESH_DATE=unset
RUN echo "apt refresh: $APT_REFRESH_DATE" \
    && apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-ron \
        ghostscript \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
# Upgrade pip first — the base image's pip carries install-time CVEs
# (CVE-2025-8869 et al.); a patched pip builds the rest of the deps.
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt && pip install gunicorn

COPY . .

RUN useradd --create-home --shell /bin/bash app \
    && mkdir -p /app/media/uploads /app/media/processed /app/staticfiles /app/data /app/data/fastembed \
    && chown -R app:app /app
USER app

EXPOSE 8000

# Run under ASGI (uvicorn workers) so the async rephrase preview view doesn't
# block other requests during slow upstream AI calls. Sync views still work.
#
# Migrations + collectstatic are NOT in this CMD: they run once in a
# dedicated `migrate` service in docker-compose so multiple web replicas
# don't race against each other on startup.
CMD ["sh", "-c", "gunicorn pdf_project.asgi:application --bind 0.0.0.0:8000 --workers ${GUNICORN_WORKERS:-3} --worker-class uvicorn.workers.UvicornWorker --timeout 60"]
