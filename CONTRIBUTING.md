# Contributing

Thanks for taking a look. This is a self-hosted PDF toolbox (Django 5.2 +
Celery + Postgres/pgvector). The bar for a change to land is simple: the
same gates CI runs must pass locally, and behaviour changes come with
tests.

## Dev setup

Docker is the least-surprising path — it brings up the whole stack
(Postgres+pgvector, Redis, PgBouncer, web, worker, nginx, observability):

```bash
git clone https://github.com/Alexandru2984/pdf_Editor_v2.git
cd pdf_Editor_v2
cat > .env <<EOF
SECRET_KEY=$(openssl rand -hex 32)
POSTGRES_PASSWORD=$(openssl rand -hex 16)
REDIS_PASSWORD=$(openssl rand -hex 16)
GRAFANA_ADMIN_PASSWORD=$(openssl rand -hex 16)
DEBUG=False
ALLOWED_HOSTS=localhost,127.0.0.1
EOF
docker compose up -d
```

Bare-metal (needs system libraries for OCR and PDF/A):

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
sudo apt-get install -y tesseract-ocr tesseract-ocr-ron ghostscript
python manage.py migrate
python manage.py runserver
```

The Django test suite needs Postgres **with the pgvector extension** (some
columns are `vector`-typed) — plain SQLite won't run it. The `pgvector/
pgvector:pg16` image or a local Postgres with `CREATE EXTENSION vector`
both work.

## Before you push

CI runs these exact checks (`.github/workflows/test.yml`); run them first:

```bash
# Lint + format (must be clean)
ruff check pdfeditor/ pdf_project/
ruff format --check pdfeditor/ pdf_project/

# Types (strict on the pure PDF library)
mypy pdfeditor/

# Security static analysis
bandit -r pdfeditor/ pdf_project/ -c pyproject.toml -x "tests*,*/tests*"

# Dependency CVEs
pip-audit --requirement requirements.txt --strict

# Tests — point DATABASE_URL at a pgvector Postgres; disable ClamAV locally
DATABASE_URL="postgres://USER:PASS@127.0.0.1:5432/DB" \
  CLAMAV_ENABLED=0 python manage.py test pdfeditor

# SDK unit tests (network mocked, no services needed)
python -m unittest discover sdk/tests
```

`pre-commit install` wires ruff + ruff-format + mypy + the hygiene hooks
(trailing whitespace, large-file guard, YAML check) so most of the above
runs on every commit. The Python matrix in CI is 3.10 / 3.11 / 3.12.

## Conventions

- **Match the surrounding code.** The `pdf_processor/` package is a pure
  library with no Django imports — keep it that way so it stays testable in
  isolation. Views live under `pdfeditor/views/` grouped by concern; API
  code under `pdfeditor/api/`.
- **User-facing strings are translated.** Wrap them in `{% trans %}` /
  `gettext`, then `python manage.py makemessages -l ro -l en --ignore=venv`
  and fill in `locale/ro/LC_MESSAGES/django.po`. Both catalogs ship at
  100%; don't regress that.
- **Behaviour changes ship with tests.** Add them next to the suite for the
  area you touched (`tests_*.py`). A new PDF operation needs both a
  `pdf_processor` test and a view/API test.
- **Migrations are one-shot in prod.** They run in a dedicated `migrate`
  init container, so don't add data migrations that assume a single web
  process.
- **Don't loosen security defaults** to make something work — the CSP,
  rate limits, path guards, and container hardening (`cap_drop: ALL`,
  read-only rootfs) are load-bearing. If a change needs one relaxed, call
  it out explicitly in the PR.

## Reporting security issues

Please don't open a public issue for vulnerabilities — see
[SECURITY.md](SECURITY.md) for the disclosure process.
