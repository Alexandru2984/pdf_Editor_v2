"""``pdf-edit`` command-line tool.

Config order (highest precedence first):

1. CLI flags (``--url``, ``--api-key``)
2. Environment variables (``PDF_EDITOR_URL``, ``PDF_EDITOR_API_KEY``)
3. ``~/.config/pdf-editor/config.json`` (keys ``url``, ``api_key``)

Examples:
    pdf-edit upload report.pdf
    pdf-edit list
    pdf-edit compress <pdf-id> --quality medium --out small.pdf
    pdf-edit summarize <pdf-id>
    pdf-edit batch compress id1 id2 id3 --quality low
    pdf-edit ocr <pdf-id> --language eng+ron --out searchable.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .client import ApiError, PdfEditorClient

_CONFIG_PATH = Path.home() / ".config" / "pdf-editor" / "config.json"


def _load_config() -> dict:
    if not _CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(_CONFIG_PATH.read_text())
    except (OSError, ValueError):
        return {}


def _make_client(args) -> PdfEditorClient:
    config = _load_config()
    url = args.url or os.environ.get("PDF_EDITOR_URL") or config.get("url")
    api_key = args.api_key or os.environ.get("PDF_EDITOR_API_KEY") or config.get("api_key")
    if not url or not api_key:
        sys.stderr.write(
            "pdf-edit: missing --url and/or --api-key.\n"
            "Set them via flags, PDF_EDITOR_URL/PDF_EDITOR_API_KEY env vars,\n"
            "or ~/.config/pdf-editor/config.json.\n"
        )
        raise SystemExit(2)
    return PdfEditorClient(url, api_key)


def _print_json(data) -> None:
    json.dump(data, sys.stdout, indent=2, ensure_ascii=False, default=str)
    sys.stdout.write("\n")


def _cmd_upload(client: PdfEditorClient, args) -> int:
    result = client.upload(args.file)
    _print_json(result)
    return 0


def _cmd_list(client: PdfEditorClient, args) -> int:
    result = (client.list_pdfs() if args.what == "pdfs"
              else client.list_outputs() if args.what == "outputs"
              else client.list_jobs(status=args.status or None, kind=args.kind))
    _print_json(result)
    return 0


def _cmd_compress(client: PdfEditorClient, args) -> int:
    out = client.compress(args.pdf_id, quality=args.quality)
    _print_json(out)
    if args.out:
        client.download(out["id"], args.out)
        sys.stderr.write(f"saved: {args.out}\n")
    return 0


def _cmd_summarize(client: PdfEditorClient, args) -> int:
    result = client.summarize(args.pdf_id, language=args.language)
    if args.json:
        _print_json(result)
    else:
        sys.stdout.write(result["summary"] + "\n")
    return 0


def _cmd_batch(client: PdfEditorClient, args) -> int:
    params = json.loads(args.params) if args.params else {}
    submit = client.batch(args.op, args.pdf_ids, params=params)
    if args.no_wait:
        _print_json(submit)
        return 0
    sys.stderr.write(f"submitted job {submit['job_id']}, waiting…\n")
    job = client.wait_for(submit)
    _print_json(job)
    return 0 if job["status"] == "done" else 1


def _cmd_ocr(client: PdfEditorClient, args) -> int:
    submit = client.ocr(args.pdf_id, language=args.language, dpi=args.dpi)
    job = client.wait_for(submit)
    if job["status"] != "done":
        sys.stderr.write(f"job failed: {job.get('error_message', '?')}\n")
        return 1
    if args.out:
        client.download(job["output_id"], args.out)
        sys.stderr.write(f"saved: {args.out}\n")
    _print_json(job)
    return 0


def _cmd_cancel(client: PdfEditorClient, args) -> int:
    result = client.cancel_job(args.job_id)
    _print_json(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pdf-edit", description=__doc__.split("\n")[0])
    p.add_argument("--url", help="API base URL, e.g. https://pdf.micutu.com")
    p.add_argument("--api-key", help="X-API-Key issued from the profile page")
    sub = p.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upload", help="Upload a PDF.")
    up.add_argument("file")
    up.set_defaults(fn=_cmd_upload)

    ls = sub.add_parser("list", help="List PDFs, outputs, or jobs.")
    ls.add_argument("what", choices=["pdfs", "outputs", "jobs"])
    ls.add_argument("--status", action="append", help="filter jobs by status (repeatable)")
    ls.add_argument("--kind", help="filter jobs by kind")
    ls.set_defaults(fn=_cmd_list)

    co = sub.add_parser("compress", help="Compress a PDF.")
    co.add_argument("pdf_id")
    co.add_argument("--quality", default="medium", choices=["low", "medium", "high"])
    co.add_argument("--out", help="if set, also download the result to this path")
    co.set_defaults(fn=_cmd_compress)

    sm = sub.add_parser("summarize", help="Get a 200-400 word summary of a PDF.")
    sm.add_argument("pdf_id")
    sm.add_argument("--language", default="English")
    sm.add_argument("--json", action="store_true", help="emit full JSON instead of just the summary")
    sm.set_defaults(fn=_cmd_summarize)

    ba = sub.add_parser("batch", help="Apply one op to a list of PDFs.")
    ba.add_argument("op", help="op name (compress, rotate, watermark, …)")
    ba.add_argument("pdf_ids", nargs="+")
    ba.add_argument("--params", help="JSON-encoded params, e.g. '{\"quality\":\"low\"}'")
    ba.add_argument("--no-wait", action="store_true", help="return immediately with the job id")
    ba.set_defaults(fn=_cmd_batch)

    oc = sub.add_parser("ocr", help="Add OCR layer to a (scanned) PDF.")
    oc.add_argument("pdf_id")
    oc.add_argument("--language", default="eng+ron")
    oc.add_argument("--dpi", type=int, default=200)
    oc.add_argument("--out", help="download the result to this path")
    oc.set_defaults(fn=_cmd_ocr)

    cn = sub.add_parser("cancel", help="Cancel a running job.")
    cn.add_argument("job_id")
    cn.set_defaults(fn=_cmd_cancel)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        client = _make_client(args)
        return args.fn(client, args)
    except ApiError as exc:
        sys.stderr.write(f"API error {exc.status_code}: {exc.body}\n")
        return 1
    except FileNotFoundError as exc:
        sys.stderr.write(f"File not found: {exc.filename}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
