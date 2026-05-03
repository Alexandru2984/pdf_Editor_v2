"""Split, merge, compress views."""

import os

from django.conf import settings
from django.contrib import messages
from django.http import Http404
from django.shortcuts import redirect, render
from django.utils.translation import gettext as _

from ..forms import CompressPDFForm, MergePDFForm, SplitPDFForm
from ..models import ProcessedPDF
from ..pdf_processor import compress_pdf, merge_pdfs, split_pdf
from ._common import (
    attachment_response,
    get_pdf_by_id,
    get_uploaded_pdfs,
    owner_filter,
    record_output,
)


def _resolve_pdf_or_redirect(request):
    uploaded_pdfs = get_uploaded_pdfs(request)
    if not uploaded_pdfs:
        messages.error(request, _("No PDF found. Please upload a PDF first."))
        return None, None, redirect("dashboard")

    pdf_id = request.GET.get("pdf")
    selected_pdf = get_pdf_by_id(request, pdf_id) if pdf_id else uploaded_pdfs[0]
    if not selected_pdf:
        messages.error(request, _("Selected PDF not found."))
        return None, None, redirect("dashboard")
    return selected_pdf, uploaded_pdfs, None


def _fetch_output(request, session_key):
    output_id = request.session.get(session_key)
    if not output_id:
        return None
    return ProcessedPDF.objects.filter(owner_filter(request), id=output_id).first()


# ---------- Split ----------


def split_view(request):
    selected_pdf, uploaded_pdfs, early = _resolve_pdf_or_redirect(request)
    if early:
        return early

    if request.method == "POST":
        form = SplitPDFForm(request.POST)
        if form.is_valid():
            try:
                output_files = split_pdf(selected_pdf.path, form.cleaned_data["ranges"])
                outputs = [
                    record_output(request, kind=ProcessedPDF.KIND_SPLIT, path=p, source=selected_pdf)
                    for p in output_files
                ]
                request.session["split_ids"] = [str(o.id) for o in outputs]
                messages.success(
                    request,
                    _("PDF split successfully into %(count)s files!") % {"count": len(outputs)},
                )
                return redirect("split_result")
            except ValueError as e:
                messages.error(request, _("Error: %(err)s") % {"err": e})
            except Exception as e:
                messages.error(request, _("Error splitting PDF: %(err)s") % {"err": e})
    else:
        form = SplitPDFForm()

    return render(
        request,
        "pdfeditor/split.html",
        {
            "form": form,
            "pdf_name": selected_pdf.name,
            "pdf_path_relative": os.path.relpath(selected_pdf.path, settings.MEDIA_ROOT),
            "uploaded_pdfs": uploaded_pdfs,
            "selected_pdf": selected_pdf,
        },
    )


def _session_split_outputs(request):
    ids = request.session.get("split_ids", []) or []
    if not ids:
        return []
    outputs = ProcessedPDF.objects.filter(owner_filter(request), id__in=ids)
    ordered = {str(o.id): o for o in outputs}
    return [ordered[i] for i in ids if i in ordered]


def split_result_view(request):
    outputs = _session_split_outputs(request)
    if not outputs:
        messages.error(request, _("No split files found."))
        return redirect("dashboard")

    files_info = [
        {
            "name": o.name,
            "path": o.path,
            "path_relative": os.path.relpath(o.path, settings.MEDIA_ROOT),
            "size": o.size,
        }
        for o in outputs
        if o.exists_on_disk()
    ]

    return render(
        request,
        "pdfeditor/split_result.html",
        {
            "files_info": files_info,
            "split_count": len(files_info),
        },
    )


def download_split_file_view(request):
    file_index = request.GET.get("file")
    outputs = _session_split_outputs(request)

    if file_index is None or not outputs:
        raise Http404("File not found")

    try:
        idx = int(file_index)
    except (TypeError, ValueError) as exc:
        raise Http404("Invalid file index") from exc

    if idx < 0 or idx >= len(outputs):
        raise Http404("File index out of range")

    return attachment_response(outputs[idx].path, not_found_message="File not found on disk")


# ---------- Merge ----------


def merge_view(request):
    uploaded_pdfs = get_uploaded_pdfs(request)
    if len(uploaded_pdfs) < 2:
        messages.error(request, _("You need at least 2 PDFs to merge. Please upload more files."))
        return redirect("dashboard")

    if request.method == "POST":
        form = MergePDFForm(request.POST)
        if form.is_valid():
            try:
                pdf_paths = []
                for pdf_id in form.cleaned_data["selected_pdfs"]:
                    pdf = get_pdf_by_id(request, pdf_id)
                    if not pdf:
                        messages.error(
                            request,
                            _("PDF with ID %(pdf_id)s not found.") % {"pdf_id": pdf_id},
                        )
                        return redirect("merge")
                    pdf_paths.append(pdf.path)

                if len(pdf_paths) < 2:
                    messages.error(request, _("At least 2 PDFs are required for merging."))
                    return redirect("merge")

                merged_path = merge_pdfs(pdf_paths, form.cleaned_data.get("output_name"))
                output = record_output(request, kind=ProcessedPDF.KIND_MERGE, path=merged_path)
                request.session["merged_pdf_id"] = str(output.id)
                request.session["merged_pdf_count"] = len(pdf_paths)
                messages.success(
                    request,
                    _("Successfully merged %(count)s PDFs!") % {"count": len(pdf_paths)},
                )
                return redirect("merge_result")
            except ValueError as e:
                messages.error(request, _("Error: %(err)s") % {"err": e})
            except Exception as e:
                messages.error(request, _("Error merging PDFs: %(err)s") % {"err": e})
    else:
        form = MergePDFForm()

    return render(
        request,
        "pdfeditor/merge.html",
        {
            "form": form,
            "uploaded_pdfs": uploaded_pdfs,
        },
    )


def merge_result_view(request):
    output = _fetch_output(request, "merged_pdf_id")
    if not output or not output.exists_on_disk():
        messages.error(request, _("Merged file not found."))
        return redirect("dashboard")

    return render(
        request,
        "pdfeditor/merge_result.html",
        {
            "merged_filename": output.name,
            "merged_size": output.size,
            "merged_count": request.session.get("merged_pdf_count", 0),
            "pdf_path_relative": os.path.relpath(output.path, settings.MEDIA_ROOT),
        },
    )


def download_merged_view(request):
    output = _fetch_output(request, "merged_pdf_id")
    if not output:
        messages.error(request, _("File not found."))
        return redirect("dashboard")
    try:
        return attachment_response(output.path)
    except Http404:
        messages.error(request, _("File not found."))
        return redirect("dashboard")


# ---------- Compress ----------


def compress_view(request):
    selected_pdf, uploaded_pdfs, early = _resolve_pdf_or_redirect(request)
    if early:
        return early

    pdf_path = selected_pdf.path

    if request.method == "POST":
        form = CompressPDFForm(request.POST)
        if form.is_valid():
            try:
                output_path, original_size, compressed_size, ratio = compress_pdf(
                    pdf_path,
                    quality=form.cleaned_data["quality"],
                )
                output = record_output(
                    request,
                    kind=ProcessedPDF.KIND_COMPRESS,
                    path=output_path,
                    source=selected_pdf,
                )
                request.session["compressed_pdf_id"] = str(output.id)
                request.session["original_size"] = original_size
                request.session["compressed_size"] = compressed_size
                request.session["compression_ratio"] = ratio
                messages.success(
                    request,
                    _("PDF compressed successfully! Saved %(ratio).1f%% space.") % {"ratio": ratio},
                )
                return redirect("compress_result")
            except Exception as e:
                messages.error(request, _("Error compressing PDF: %(err)s") % {"err": e})
    else:
        form = CompressPDFForm()

    return render(
        request,
        "pdfeditor/compress.html",
        {
            "form": form,
            "pdf_name": selected_pdf.name,
            "pdf_path_relative": os.path.relpath(pdf_path, settings.MEDIA_ROOT),
            "uploaded_pdfs": uploaded_pdfs,
            "selected_pdf": selected_pdf,
            "original_size": os.path.getsize(pdf_path),
        },
    )


def compress_result_view(request):
    output = _fetch_output(request, "compressed_pdf_id")
    if not output or not output.exists_on_disk():
        messages.error(request, _("Compressed file not found."))
        return redirect("dashboard")

    original_size = request.session.get("original_size", 0)
    compressed_size = request.session.get("compressed_size", 0)
    return render(
        request,
        "pdfeditor/compress_result.html",
        {
            "compressed_filename": output.name,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": request.session.get("compression_ratio", 0),
            "saved_bytes": original_size - compressed_size,
            "pdf_path_relative": os.path.relpath(output.path, settings.MEDIA_ROOT),
        },
    )


def download_compressed_view(request):
    output = _fetch_output(request, "compressed_pdf_id")
    if not output:
        messages.error(request, _("File not found."))
        return redirect("dashboard")
    try:
        return attachment_response(output.path)
    except Http404:
        messages.error(request, _("File not found."))
        return redirect("dashboard")
