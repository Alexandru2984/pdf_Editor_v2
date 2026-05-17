# pdf-editor-sdk

Python client + CLI for the [PDF Editor](https://pdf.micutu.com) REST API.

```bash
pip install pdf-editor-sdk
```

## Quick start (library)

```python
from pdf_editor import PdfEditorClient

c = PdfEditorClient("https://pdf.micutu.com", api_key="...")

# Upload + sync ops
pdf = c.upload("report.pdf")
out = c.compress(pdf["id"], quality="medium")
c.download(out["id"], "report-small.pdf")

# Single-shot AI summary
print(c.summarize(pdf["id"], language="Romanian")["summary"])

# Batch one op across many PDFs (returns a Job)
submit = c.batch("compress", [pdf["id"]], params={"quality": "low"})
job = c.wait_for(submit, timeout=600)
print(job["status"], job["params"]["results"])

# Async ops (OCR, PDF/A, etc.) — wait_for polls until done
submit = c.ocr(pdf["id"], language="eng+ron")
job = c.wait_for(submit)
c.download(job["output_id"], "report-ocr.pdf")
```

## CLI

```bash
# One-time setup
export PDF_EDITOR_URL=https://pdf.micutu.com
export PDF_EDITOR_API_KEY=...

# or write ~/.config/pdf-editor/config.json:
#   { "url": "...", "api_key": "..." }

pdf-edit upload report.pdf
pdf-edit list pdfs
pdf-edit compress <pdf-id> --quality medium --out small.pdf
pdf-edit summarize <pdf-id> --language Romanian
pdf-edit ocr <pdf-id> --out searchable.pdf
pdf-edit batch compress id1 id2 id3 --params '{"quality":"low"}'
pdf-edit list jobs --status queued --status running
pdf-edit cancel <job-id>
```

## Errors

Non-2xx responses raise `pdf_editor.ApiError` with `.status_code` and
`.body` (parsed JSON if possible, otherwise raw text).

## Compatibility

Python ≥ 3.10. Depends only on `requests`.

## Development

```bash
cd sdk
python -m unittest discover -s tests
```
