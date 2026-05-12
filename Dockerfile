FROM python:3.14-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps: tesseract for OCR (eng + ron language packs), ghostscript for
# PDF/A conversion, libgl/libglib for Pillow image ops, libgomp1 for the
# ONNX Runtime backend that powers fastembed (RAG embeddings).
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-ron \
        ghostscript \
        libglib2.0-0 \
        libgomp1 \
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

# Run under ASGI (uvicorn workers) so the async rephrase preview view doesn't
# block other requests during slow upstream AI calls. Sync views still work.
CMD ["sh", "-c", "python manage.py migrate --noinput && python manage.py collectstatic --noinput && gunicorn pdf_project.asgi:application --bind 0.0.0.0:8000 --workers 3 --worker-class uvicorn.workers.UvicornWorker --timeout 60"]
