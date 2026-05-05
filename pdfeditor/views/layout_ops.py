"""Watermark, rotate, page-numbers, reorder views."""

import os

import fitz
from django.conf import settings
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from django.http import Http404, HttpResponse
from django.shortcuts import redirect, render
from django.utils.translation import gettext as _
from django.views.decorators.http import require_GET

from ..forms import PageNumbersForm, ReorderPagesForm, RotatePagesForm, WatermarkForm
from ..models import ProcessedPDF, UploadedPDF
from ..pdf_processor import (
    add_page_numbers,
    add_watermark,
    render_page_thumbnail,
    reorder_pages,
    rotate_pages,
)
from ._common import (
    attachment_response,
    get_pdf_by_id,
    get_uploaded_pdfs,
    owner_filter,
    record_output,
)


def _require_pdf(request):
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


# ---------- Watermark ----------


def watermark_view(request):
    selected_pdf, uploaded_pdfs, early = _require_pdf(request)
    if early:
        return early

    pdf_path = selected_pdf.path

    if request.method == "POST":
        form = WatermarkForm(request.POST, request.FILES)
        if form.is_valid():
            options = {
                "position": form.cleaned_data["position"],
                "opacity": form.cleaned_data["opacity"],
                "rotation": form.cleaned_data["rotation"],
            }
            try:
                if form.cleaned_data["watermark_type"] == "text":
                    options["font_size"] = form.cleaned_data.get("font_size", 48)
                    output_path = add_watermark(pdf_path, "text", form.cleaned_data["text_content"], options)
                else:
                    uploaded_image = form.cleaned_data["watermark_image"]
                    fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, "temp"))
                    safe_image_name = os.path.basename(uploaded_image.name)
                    image_filename = fs.save(safe_image_name, uploaded_image)
                    image_path = fs.path(image_filename)
                    try:
                        output_path = add_watermark(pdf_path, "image", image_path, options)
                    finally:
                        if os.path.exists(image_path):
                            os.remove(image_path)

                output = record_output(
                    request,
                    kind=ProcessedPDF.KIND_WATERMARK,
                    path=output_path,
                    source=selected_pdf,
                )
                request.session["watermarked_pdf_id"] = str(output.id)
                messages.success(request, "Watermark added successfully!")
                return redirect("watermark_result")
            except Exception as e:
                messages.error(request, f"Error adding watermark: {e}")
    else:
        form = WatermarkForm()

    return render(
        request,
        "pdfeditor/watermark.html",
        {
            "form": form,
            "pdf_name": selected_pdf.name,
            "pdf_path_relative": os.path.relpath(pdf_path, settings.MEDIA_ROOT),
            "uploaded_pdfs": uploaded_pdfs,
            "selected_pdf": selected_pdf,
        },
    )


def watermark_result_view(request):
    output = _fetch_output(request, "watermarked_pdf_id")
    if not output or not output.exists_on_disk():
        messages.error(request, "Watermarked file not found.")
        return redirect("dashboard")

    return render(
        request,
        "pdfeditor/watermark_result.html",
        {
            "watermarked_filename": output.name,
            "watermarked_size": output.size,
            "pdf_path_relative": os.path.relpath(output.path, settings.MEDIA_ROOT),
        },
    )


def download_watermarked_view(request):
    output = _fetch_output(request, "watermarked_pdf_id")
    if not output:
        messages.error(request, "File not found.")
        return redirect("dashboard")
    try:
        return attachment_response(output.path)
    except Http404:
        messages.error(request, "File not found.")
        return redirect("dashboard")


# ---------- Rotate ----------


def rotate_view(request):
    selected_pdf, uploaded_pdfs, early = _require_pdf(request)
    if early:
        return early

    pdf_path = selected_pdf.path

    if request.method == "POST":
        form = RotatePagesForm(request.POST)
        if form.is_valid():
            rotation_angle = int(form.cleaned_data["rotation_angle"])
            page_range = form.cleaned_data.get("page_range", "").strip()
            try:
                output_path = rotate_pages(pdf_path, rotation_angle, page_range or None)
                output = record_output(
                    request,
                    kind=ProcessedPDF.KIND_ROTATE,
                    path=output_path,
                    source=selected_pdf,
                )
                request.session["rotated_pdf_id"] = str(output.id)
                request.session["rotation_angle"] = rotation_angle
                messages.success(request, f"Pages rotated {rotation_angle}° successfully!")
                return redirect("rotate_result")
            except ValueError as e:
                messages.error(request, f"Error: {e}")
            except Exception as e:
                messages.error(request, f"Error rotating pages: {e}")
    else:
        form = RotatePagesForm()

    return render(
        request,
        "pdfeditor/rotate.html",
        {
            "form": form,
            "pdf_name": selected_pdf.name,
            "pdf_path_relative": os.path.relpath(pdf_path, settings.MEDIA_ROOT),
            "uploaded_pdfs": uploaded_pdfs,
            "selected_pdf": selected_pdf,
        },
    )


def rotate_result_view(request):
    output = _fetch_output(request, "rotated_pdf_id")
    if not output or not output.exists_on_disk():
        messages.error(request, "Rotated file not found.")
        return redirect("dashboard")

    return render(
        request,
        "pdfeditor/rotate_result.html",
        {
            "rotated_filename": output.name,
            "rotated_size": output.size,
            "rotation_angle": request.session.get("rotation_angle", 0),
            "pdf_path_relative": os.path.relpath(output.path, settings.MEDIA_ROOT),
        },
    )


def download_rotated_view(request):
    output = _fetch_output(request, "rotated_pdf_id")
    if not output:
        messages.error(request, "File not found.")
        return redirect("dashboard")
    try:
        return attachment_response(output.path)
    except Http404:
        messages.error(request, "File not found.")
        return redirect("dashboard")


# ---------- Page numbers ----------


def page_numbers_view(request):
    selected_pdf, uploaded_pdfs, early = _require_pdf(request)
    if early:
        return early

    pdf_path = selected_pdf.path

    if request.method == "POST":
        form = PageNumbersForm(request.POST)
        if form.is_valid():
            options = {
                "position": form.cleaned_data["position"],
                "format": form.cleaned_data["format"],
                "font_size": form.cleaned_data["font_size"],
                "start_page": form.cleaned_data["start_page"],
            }
            try:
                output_path = add_page_numbers(pdf_path, options)
                output = record_output(
                    request,
                    kind=ProcessedPDF.KIND_PAGE_NUMBERS,
                    path=output_path,
                    source=selected_pdf,
                )
                request.session["numbered_pdf_id"] = str(output.id)
                messages.success(request, "Page numbers added successfully!")
                return redirect("page_numbers_result")
            except Exception as e:
                messages.error(request, f"Error adding page numbers: {e}")
    else:
        form = PageNumbersForm()

    return render(
        request,
        "pdfeditor/page_numbers.html",
        {
            "form": form,
            "pdf_name": selected_pdf.name,
            "pdf_path_relative": os.path.relpath(pdf_path, settings.MEDIA_ROOT),
            "uploaded_pdfs": uploaded_pdfs,
            "selected_pdf": selected_pdf,
        },
    )


def page_numbers_result_view(request):
    output = _fetch_output(request, "numbered_pdf_id")
    if not output or not output.exists_on_disk():
        messages.error(request, "Numbered file not found.")
        return redirect("dashboard")

    return render(
        request,
        "pdfeditor/page_numbers_result.html",
        {
            "numbered_filename": output.name,
            "numbered_size": output.size,
            "pdf_path_relative": os.path.relpath(output.path, settings.MEDIA_ROOT),
        },
    )


def download_numbered_view(request):
    output = _fetch_output(request, "numbered_pdf_id")
    if not output:
        messages.error(request, "File not found.")
        return redirect("dashboard")
    try:
        return attachment_response(output.path)
    except Http404:
        messages.error(request, "File not found.")
        return redirect("dashboard")


# ---------- Reorder / delete pages ----------


def reorder_view(request):
    selected_pdf, uploaded_pdfs, early = _require_pdf(request)
    if early:
        return early

    pdf_path = selected_pdf.path

    try:
        with fitz.open(pdf_path) as doc:
            page_count = len(doc)
    except Exception:
        messages.error(request, _("Could not open the selected PDF."))
        return redirect("dashboard")

    if request.method == "POST":
        form = ReorderPagesForm(request.POST)
        if form.is_valid():
            page_order = form.cleaned_data["page_order"]
            if any(n > page_count for n in page_order):
                messages.error(
                    request,
                    _("Page numbers must be between 1 and %(total)d.")
                    % {"total": page_count},
                )
            else:
                try:
                    output_path = reorder_pages(pdf_path, page_order)
                    output = record_output(
                        request,
                        kind=ProcessedPDF.KIND_REORDER,
                        path=output_path,
                        source=selected_pdf,
                    )
                    request.session["reordered_pdf_id"] = str(output.id)
                    request.session["reordered_original_count"] = page_count
                    request.session["reordered_kept_count"] = len(page_order)
                    messages.success(request, _("Pages reordered successfully!"))
                    return redirect("reorder_result")
                except ValueError as e:
                    messages.error(request, _("Error: %(err)s") % {"err": e})
                except Exception as e:
                    messages.error(
                        request,
                        _("Error reordering pages: %(err)s") % {"err": e},
                    )
    else:
        form = ReorderPagesForm()

    return render(
        request,
        "pdfeditor/reorder.html",
        {
            "form": form,
            "pdf_name": selected_pdf.name,
            "pdf_id": str(selected_pdf.id),
            "page_count": page_count,
            "page_numbers": list(range(1, page_count + 1)),
            "pdf_path_relative": os.path.relpath(pdf_path, settings.MEDIA_ROOT),
            "uploaded_pdfs": uploaded_pdfs,
            "selected_pdf": selected_pdf,
        },
    )


def reorder_result_view(request):
    output = _fetch_output(request, "reordered_pdf_id")
    if not output or not output.exists_on_disk():
        messages.error(request, _("Reordered file not found."))
        return redirect("dashboard")

    return render(
        request,
        "pdfeditor/reorder_result.html",
        {
            "reordered_filename": output.name,
            "reordered_size": output.size,
            "original_count": request.session.get("reordered_original_count"),
            "kept_count": request.session.get("reordered_kept_count"),
            "pdf_path_relative": os.path.relpath(output.path, settings.MEDIA_ROOT),
        },
    )


def download_reordered_view(request):
    output = _fetch_output(request, "reordered_pdf_id")
    if not output:
        messages.error(request, _("File not found."))
        return redirect("dashboard")
    try:
        return attachment_response(output.path)
    except Http404:
        messages.error(request, _("File not found."))
        return redirect("dashboard")


@require_GET
def page_thumbnail_view(request, pdf_id, page_number):
    """Serve a PNG thumbnail of a single page from an uploaded PDF."""
    pdf = UploadedPDF.objects.filter(owner_filter(request), id=pdf_id).first()
    if pdf is None or not pdf.exists_on_disk():
        raise Http404
    try:
        png = render_page_thumbnail(pdf.path, int(page_number), max_width=180)
    except Exception:
        raise Http404
    response = HttpResponse(png, content_type="image/png")
    response["Cache-Control"] = "private, max-age=3600"
    return response
