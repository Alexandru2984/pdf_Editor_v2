FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps: tesseract for OCR, libgl/libglib for Pillow image ops in some flows.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt && pip install gunicorn

COPY . .

RUN useradd --create-home --shell /bin/bash app \
    && mkdir -p /app/media/uploads /app/media/processed /app/staticfiles /app/data \
    && chown -R app:app /app
USER app

EXPOSE 8000

# collectstatic at runtime (needs SECRET_KEY); migrate then start gunicorn.
CMD ["sh", "-c", "python manage.py migrate --noinput && python manage.py collectstatic --noinput && gunicorn pdf_project.wsgi:application --bind 0.0.0.0:8000 --workers 3 --timeout 60"]
