# PDF Editor v2

Django web app for editing PDFs: find/replace text in-place, split, merge, compress,
watermark, rotate, add page numbers, OCR to text, and AI-powered rephrase of
selected regions using either a local Ollama model or the Groq API.

Live demo: https://pdf.micutu.com

## What makes this non-trivial

Editing text *inside* an existing PDF is harder than generating a new one. PyMuPDF
operates on coordinate rectangles, not paragraphs, so replacing "lorem ipsum" in a
document means:

1. Finding the exact bounding box of the original span.
2. Detecting the paragraph and column it belongs to (line gap, font size, alignment).
3. Redacting that area and inserting replacement text at the same coordinates with a
   font that approximates the original (the FLOW mode also shifts all text below
   when the replacement changes the paragraph height, and extends the page if it
   overflows).

The logic for this lives in `pdfeditor/pdf_processor/` and is the most interesting
part of the codebase to read.

## Feature overview

| Area | Description |
|------|-------------|
| Find & replace | Document-wide, case-sensitive/insensitive, page-range filter |
| AI rephrase    | Select a region in the browser (PDF.js) → send text to Ollama or Groq → paste back with paragraph reflow |
| Split / Merge  | Arbitrary page ranges; merge multiple uploads in a chosen order |
| Compress       | JPEG re-encoding of embedded images, 3 quality presets |
| Watermark      | Text or image, 9 positions, opacity + rotation |
| Rotate         | 90 / 180 / 270° on selected pages |
| Page numbers   | Position + format + font size + start page |
| Extract / OCR  | Text-layer extraction or Tesseract OCR fallback |

## Architecture

```
pdf_project/            Django project (settings, urls, wsgi)
pdfeditor/
├── models.py           UploadedPDF, ProcessedPDF (scoped by session_key)
├── admin.py            Admin views with list_filter, search, raw_id_fields
├── forms.py            Form validators for each operation
├── ai_service.py       OllamaProvider + GroqProvider behind a common interface
├── pdf_processor/      PDF manipulation package
│   ├── _common.py      Paths, page-range parsing, font/color mapping
│   ├── ops.py          Split, merge, compress, watermark, rotate, page numbers
│   ├── extract.py      Text-layer + OCR extraction
│   ├── _layout.py      Span/Line/Block model, paragraph detection, block shifting
│   └── edit.py         SAFE + FLOW text replacement, find/replace, coordinate rephrase
├── views/              HTTP views grouped by concern
│   ├── _common.py      Session scoping, guarded media serving, output recording
│   ├── upload.py       Dashboard, upload (magic-byte + size validation), delete
│   ├── edit.py         Find/replace form + result
│   ├── basic_ops.py    Split, merge, compress
│   ├── layout_ops.py   Watermark, rotate, page numbers
│   ├── extract.py      AJAX text/OCR
│   └── rephrase.py     AI rephrase UI + preview (rate-limited)
├── templates/          Django templates (includes PDF.js viewer)
└── management/commands/
    └── cleanup_old_pdfs.py   Prune DB rows + files + orphaned files on disk
```

PDF operations live in a standalone package with no Django imports outside
`_common.settings.MEDIA_ROOT`, so they could be extracted into a separate library.

## Anonymous session scoping

There is no user auth. Each browser session gets a Django `session_key` and every
row in `UploadedPDF` / `ProcessedPDF` carries that key. Media files are served
through a guarded view that matches the requested path against the DB rows owned
by the current session; direct URL guessing returns 404.

## Security posture

`python manage.py check --deploy` reports zero warnings. In place:

- `SECRET_KEY`, `ALLOWED_HOSTS`, `GROQ_API_KEY` read from `.env` (chmod 600).
- `SECURE_SSL_REDIRECT`, HSTS, `SECURE_PROXY_SSL_HEADER`, Secure + HttpOnly +
  SameSite cookies (skipped in tests via a `TESTING` flag).
- Upload validation: `%PDF-` magic bytes, 10 MB size cap, `basename()` sanitation.
- `django-axes` — 5 failed logins per (IP, username) → 1h lockout.
- `django-ratelimit` on the two AI endpoints (20/h and 30/h per IP).
- CSRF on all POSTs including the AJAX preview endpoint.

## Running locally

```bash
git clone https://github.com/Alexandru2984/pdf_Editor_v2.git
cd pdf_Editor_v2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cat > .env <<'EOF'
DEBUG=True
SECRET_KEY='pick-something-long-and-random'
ALLOWED_HOSTS=localhost,127.0.0.1
SECURE_SSL=0
# Optional: GROQ_API_KEY=gsk_...
EOF
chmod 600 .env

python manage.py migrate
python manage.py test pdfeditor
python manage.py runserver
```

Then open <http://localhost:8000>. For AI rephrase you need either a running
Ollama instance (`OLLAMA_HOST` env var) or a Groq API key in `.env`. OCR needs
`tesseract` on the system `PATH`.

## Production deployment

The live instance runs behind nginx → gunicorn over a Unix socket, managed by
two systemd units:

- `pdfeditor.service` — gunicorn with 3 workers, `EnvironmentFile=.env`.
- `pdfeditor-cleanup.timer` — hourly run of `cleanup_old_pdfs`, which deletes DB
  rows and files older than `PDF_CLEANUP_HOURS` plus any orphan files on disk.

## Tech stack

Django 4.2, PyMuPDF (fitz), Pillow, pytesseract, django-axes, django-ratelimit,
python-dotenv, gunicorn, systemd. AI providers: Ollama (local) and Groq (cloud),
both behind an `AIProvider` interface so new providers slot in with a single class.

## Known limitations (FLOW mode)

- Bullet/list markers are detected to avoid being swallowed into the wrong
  paragraph, but the markers themselves aren't re-inserted when blocks shift.
- Multi-column tables and complex layouts may reflow incorrectly; SAFE mode is
  the fallback when paragraph detection fails.
