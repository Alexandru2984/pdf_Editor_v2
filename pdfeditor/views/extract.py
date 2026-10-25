"""Text extraction + OCR + AI summary + more-tools AJAX endpoints."""

import json

from django.contrib import messages
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.utils.translation import gettext as _

from ..pdf_processor import extract_text_from_pdf, ocr_pdf_to_text, summarize_pdf
from ..ratelimiting import auth_aware_ratelimit
from ._common import get_pdf_by_id, get_uploaded_pdfs

SUMMARY_LANGUAGES = (
    ("English", "English"),
    ("Romanian", "Română"),
    ("French", "Français"),
    ("German", "Deutsch"),
    ("Spanish", "Español"),
    ("Italian", "Italiano"),
)


def more_tools_view(request):
    uploaded_pdfs = get_uploaded_pdfs(request)
    if not uploaded_pdfs:
        messages.error(request, _("No PDF found. Please upload a PDF first."))
        return redirect("dashboard")

    return render(
        request,
        "pdfeditor/more_tools.html",
        {
            "uploaded_pdfs": uploaded_pdfs,
            "summary_languages": SUMMARY_LANGUAGES,
        },
    )


@auth_aware_ratelimit(anon_rate="15/h", user_rate="60/h", method="POST")
def summarize_view(request):
    uploaded_pdfs = get_uploaded_pdfs(request)
    if not uploaded_pdfs:
        messages.error(request, _("No PDF found. Please upload a PDF first."))
        return redirect("dashboard")

    pdf_id = request.GET.get("pdf") or request.POST.get("pdf_id")
    selected_pdf = get_pdf_by_id(request, pdf_id) if pdf_id else uploaded_pdfs[0]
    if not selected_pdf:
        messages.error(request, _("Selected PDF not found."))
        return redirect("dashboard")

    summary_result = None
    selected_language = "Romanian" if (request.LANGUAGE_CODE or "").startswith("ro") else "English"

    if request.method == "POST":
        allowed_langs = {code for code, _label in SUMMARY_LANGUAGES}
        posted_lang = (request.POST.get("language") or selected_language).strip()
        if posted_lang in allowed_langs:
            selected_language = posted_lang
        try:
            summary_result = summarize_pdf(selected_pdf.path, language=selected_language)
            request.session["extracted_text"] = summary_result["summary"]
            request.session["extracted_filename"] = selected_pdf.name.replace(".pdf", "_summary.txt")
        except ValueError as exc:
            messages.error(request, str(exc))
        except Exception:
            messages.error(request, _("Summary generation failed. Please try again."))

    return render(
        request,
        "pdfeditor/summarize.html",
        {
            "uploaded_pdfs": uploaded_pdfs,
            "selected_pdf": selected_pdf,
            "pdf_name": selected_pdf.name,
            "summary_languages": SUMMARY_LANGUAGES,
            "selected_language": selected_language,
            "summary_result": summary_result,
        },
    )


def _extract_to_session(request, pdf_id, extractor, suffix):
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Invalid method"})

    pdf = get_pdf_by_id(request, pdf_id)
    if not pdf:
        return JsonResponse({"success": False, "error": "PDF not found"})

    try:
        text = extractor(pdf.path)
        request.session["extracted_text"] = text
        request.session["extracted_filename"] = pdf.name.replace(".pdf", suffix)
        return JsonResponse(
            {
                "success": True,
                "text": text,
                "filename": pdf.name,
            }
        )
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})


@auth_aware_ratelimit(anon_rate="60/h", user_rate="240/h", method="POST")
def extract_text_ajax(request, pdf_id):
    return _extract_to_session(request, pdf_id, extract_text_from_pdf, "_extracted.txt")


# OCR is CPU-intensive (300dpi rasterise + tesseract per page), so even
# logged-in users get a tight quota.
@auth_aware_ratelimit(anon_rate="10/h", user_rate="40/h", method="POST")
def ocr_text_ajax(request, pdf_id):
    return _extract_to_session(request, pdf_id, ocr_pdf_to_text, "_ocr.txt")


@auth_aware_ratelimit(anon_rate="15/h", user_rate="60/h", method="POST")
def summarize_text_ajax(request, pdf_id):
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Invalid method"}, status=405)

    pdf = get_pdf_by_id(request, pdf_id)
    if not pdf:
        return JsonResponse({"success": False, "error": "PDF not found"}, status=404)

    language = "English"
    try:
        if request.body:
            payload = json.loads(request.body)
            if isinstance(payload, dict) and payload.get("language"):
                language = str(payload["language"])
    except (ValueError, TypeError):
        pass

    try:
        result = summarize_pdf(pdf.path, language=language)
        request.session["extracted_text"] = result["summary"]
        request.session["extracted_filename"] = pdf.name.replace(".pdf", "_summary.txt")
        return JsonResponse(
            {
                "success": True,
                "text": result["summary"],
                "truncated": result["truncated"],
                "chars_used": result["chars_used"],
                "filename": pdf.name,
            }
        )
    except ValueError as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=400)
    except Exception:
        return JsonResponse({"success": False, "error": "Summary generation failed."}, status=502)


def download_text_view(request):
    text = request.session.get("extracted_text")
    filename = request.session.get("extracted_filename", "extracted.txt")

    if not text:
        messages.error(request, _("No text found to download."))
        return redirect("dashboard")

    response = HttpResponse(text, content_type="text/plain")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response
