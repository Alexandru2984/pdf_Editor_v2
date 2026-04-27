"""AcroForm form-fill view."""

import os

from django.conf import settings
from django.contrib import messages
from django.http import Http404
from django.shortcuts import redirect, render

from ..models import ProcessedPDF
from ..pdf_processor import extract_form_fields, fill_form_fields, has_form_fields
from ._common import (
    attachment_response,
    get_pdf_by_id,
    get_uploaded_pdfs,
    owner_filter,
    record_output,
)


def _resolve_pdf(request):
    uploaded_pdfs = get_uploaded_pdfs(request)
    if not uploaded_pdfs:
        messages.error(request, "No PDF found. Please upload a PDF first.")
        return None, None, redirect("dashboard")

    pdf_id = request.GET.get("pdf")
    selected_pdf = get_pdf_by_id(request, pdf_id) if pdf_id else uploaded_pdfs[0]
    if not selected_pdf:
        messages.error(request, "Selected PDF not found.")
        return None, None, redirect("dashboard")
    return selected_pdf, uploaded_pdfs, None


def _fetch_output(request, session_key):
    output_id = request.session.get(session_key)
    if not output_id:
        return None
    return ProcessedPDF.objects.filter(owner_filter(request), id=output_id).first()


def form_fill_view(request):
    selected_pdf, uploaded_pdfs, early = _resolve_pdf(request)
    if early:
        return early

    if not has_form_fields(selected_pdf.path):
        messages.warning(
            request,
            f'"{selected_pdf.name}" has no fillable form fields. '
            "Use this tool with PDFs containing AcroForm widgets.",
        )
        return redirect("dashboard")

    if request.method == "POST":
        # Field names are submitted with the prefix "field_" so they don't
        # collide with framework fields (csrfmiddlewaretoken, flatten, etc).
        values = {
            key[len("field_") :]: value for key, value in request.POST.items() if key.startswith("field_")
        }
        flatten = request.POST.get("flatten") == "on"

        try:
            output_path, num_filled, warnings = fill_form_fields(
                selected_pdf.path,
                values,
                flatten=flatten,
            )
        except ValueError as e:
            messages.error(request, f"Error: {e}")
        except Exception as e:
            messages.error(request, f"Error filling form: {e}")
        else:
            output = record_output(
                request,
                kind=ProcessedPDF.KIND_FORM_FILL,
                path=output_path,
                source=selected_pdf,
            )
            request.session["form_fill_pdf_id"] = str(output.id)
            request.session["form_fill_count"] = num_filled
            request.session["form_fill_warnings"] = warnings
            messages.success(request, f"Filled {num_filled} field(s) successfully!")
            return redirect("form_fill_result")

    fields = extract_form_fields(selected_pdf.path)
    return render(
        request,
        "pdfeditor/form_fill.html",
        {
            "pdf_name": selected_pdf.name,
            "pdf_path_relative": os.path.relpath(selected_pdf.path, settings.MEDIA_ROOT),
            "uploaded_pdfs": uploaded_pdfs,
            "selected_pdf": selected_pdf,
            "fields": fields,
        },
    )


def form_fill_result_view(request):
    output = _fetch_output(request, "form_fill_pdf_id")
    if not output or not output.exists_on_disk():
        messages.error(request, "Filled file not found.")
        return redirect("dashboard")

    warnings = request.session.get("form_fill_warnings", [])
    return render(
        request,
        "pdfeditor/form_fill_result.html",
        {
            "filled_filename": output.name,
            "filled_size": output.size,
            "fields_filled": request.session.get("form_fill_count", 0),
            "warnings": warnings,
            "has_warnings": bool(warnings),
            "pdf_path_relative": os.path.relpath(output.path, settings.MEDIA_ROOT),
        },
    )


def download_filled_view(request):
    output = _fetch_output(request, "form_fill_pdf_id")
    if not output:
        messages.error(request, "File not found.")
        return redirect("dashboard")
    try:
        return attachment_response(output.path)
    except Http404:
        messages.error(request, "File not found.")
        return redirect("dashboard")
