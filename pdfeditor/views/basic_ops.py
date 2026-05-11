"""Split, merge, compress, password-protect, digital-sign views."""

import os

from django.conf import settings
from django.contrib import messages
from django.http import Http404
from django.shortcuts import redirect, render
from django.utils.translation import gettext as _

from ..forms import (
    ComparePdfsForm,
    CompressPDFForm,
    ConvertToDocxForm,
    FlattenPDFForm,
    GenerateCertForm,
    ImagesToPdfForm,
    MakeSearchableForm,
    MergePDFForm,
    MetadataForm,
    PdfaForm,
    PdfToImagesForm,
    ProtectPDFForm,
    RedactPDFForm,
    SignPDFForm,
    SplitPDFForm,
    UnprotectPDFForm,
    VerifyPDFForm,
)
from ..models import ProcessedPDF, TrustAnchor
from ..pdf_processor import (
    compare_pdfs,
    compress_pdf,
    convert_images_to_pdf,
    convert_pdf_to_docx,
    convert_pdf_to_images,
    convert_to_pdfa,
    edit_pdf_metadata,
    flatten_pdf,
    make_pdf_searchable,
    merge_pdfs,
    protect_pdf,
    read_pdf_metadata,
    redact_text,
    remove_pdf_password,
    sign_pdf,
    split_pdf,
    verify_pdf_signatures,
)
from ..ratelimiting import auth_aware_ratelimit
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


# ---------- Password protect ----------


@auth_aware_ratelimit(anon_rate="20/h", user_rate="100/h", method="POST")
def protect_view(request):
    selected_pdf, uploaded_pdfs, early = _resolve_pdf_or_redirect(request)
    if early:
        return early

    pdf_path = selected_pdf.path

    if request.method == "POST":
        form = ProtectPDFForm(request.POST)
        if form.is_valid():
            try:
                output_path = protect_pdf(
                    pdf_path,
                    user_password=form.cleaned_data["user_password"],
                )
                output = record_output(
                    request,
                    kind=ProcessedPDF.KIND_PROTECT,
                    path=output_path,
                    source=selected_pdf,
                )
                request.session["protected_pdf_id"] = str(output.id)
                messages.success(request, _("PDF protected successfully!"))
                return redirect("protect_result")
            except Exception as e:
                messages.error(request, _("Error protecting PDF: %(err)s") % {"err": e})
    else:
        form = ProtectPDFForm()

    return render(
        request,
        "pdfeditor/protect.html",
        {
            "form": form,
            "pdf_name": selected_pdf.name,
            "pdf_path_relative": os.path.relpath(pdf_path, settings.MEDIA_ROOT),
            "uploaded_pdfs": uploaded_pdfs,
            "selected_pdf": selected_pdf,
        },
    )


def protect_result_view(request):
    output = _fetch_output(request, "protected_pdf_id")
    if not output or not output.exists_on_disk():
        messages.error(request, _("Protected file not found."))
        return redirect("dashboard")

    return render(
        request,
        "pdfeditor/protect_result.html",
        {
            "protected_filename": output.name,
            "size": os.path.getsize(output.path),
        },
    )


def download_protected_view(request):
    output = _fetch_output(request, "protected_pdf_id")
    if not output:
        messages.error(request, _("File not found."))
        return redirect("dashboard")
    try:
        return attachment_response(output.path)
    except Http404:
        messages.error(request, _("File not found."))
        return redirect("dashboard")


# ---------- Remove password ----------


@auth_aware_ratelimit(anon_rate="20/h", user_rate="100/h", method="POST")
def unprotect_view(request):
    selected_pdf, uploaded_pdfs, early = _resolve_pdf_or_redirect(request)
    if early:
        return early

    pdf_path = selected_pdf.path

    if request.method == "POST":
        form = UnprotectPDFForm(request.POST)
        if form.is_valid():
            try:
                output_path = remove_pdf_password(pdf_path, password=form.cleaned_data["password"])
                output = record_output(
                    request,
                    kind=ProcessedPDF.KIND_UNPROTECT,
                    path=output_path,
                    source=selected_pdf,
                )
                request.session["unprotected_pdf_id"] = str(output.id)
                messages.success(request, _("Password removed successfully!"))
                return redirect("unprotect_result")
            except ValueError as e:
                messages.error(request, _("Error: %(err)s") % {"err": e})
            except Exception as e:
                messages.error(request, _("Error removing password: %(err)s") % {"err": e})
    else:
        form = UnprotectPDFForm()

    return render(
        request,
        "pdfeditor/unprotect.html",
        {
            "form": form,
            "pdf_name": selected_pdf.name,
            "pdf_path_relative": os.path.relpath(pdf_path, settings.MEDIA_ROOT),
            "uploaded_pdfs": uploaded_pdfs,
            "selected_pdf": selected_pdf,
        },
    )


def unprotect_result_view(request):
    output = _fetch_output(request, "unprotected_pdf_id")
    if not output or not output.exists_on_disk():
        messages.error(request, _("Unprotected file not found."))
        return redirect("dashboard")

    return render(
        request,
        "pdfeditor/unprotect_result.html",
        {
            "pdf_filename": output.name,
            "size": os.path.getsize(output.path),
        },
    )


def download_unprotected_view(request):
    output = _fetch_output(request, "unprotected_pdf_id")
    if not output:
        messages.error(request, _("File not found."))
        return redirect("dashboard")
    try:
        return attachment_response(output.path)
    except Http404:
        messages.error(request, _("File not found."))
        return redirect("dashboard")


# ---------- Flatten ----------


@auth_aware_ratelimit(anon_rate="20/h", user_rate="100/h", method="POST")
def flatten_view(request):
    selected_pdf, uploaded_pdfs, early = _resolve_pdf_or_redirect(request)
    if early:
        return early

    pdf_path = selected_pdf.path

    if request.method == "POST":
        form = FlattenPDFForm(request.POST)
        if form.is_valid():
            try:
                output_path = flatten_pdf(
                    pdf_path,
                    flatten_annotations=form.cleaned_data.get("flatten_annotations", False),
                    flatten_forms=form.cleaned_data.get("flatten_forms", False),
                )
                output = record_output(
                    request,
                    kind=ProcessedPDF.KIND_FLATTEN,
                    path=output_path,
                    source=selected_pdf,
                )
                request.session["flattened_pdf_id"] = str(output.id)
                messages.success(request, _("PDF flattened successfully!"))
                return redirect("flatten_result")
            except ValueError as e:
                messages.error(request, _("Error: %(err)s") % {"err": e})
            except Exception as e:
                messages.error(request, _("Error flattening PDF: %(err)s") % {"err": e})
    else:
        form = FlattenPDFForm()

    return render(
        request,
        "pdfeditor/flatten.html",
        {
            "form": form,
            "pdf_name": selected_pdf.name,
            "pdf_path_relative": os.path.relpath(pdf_path, settings.MEDIA_ROOT),
            "uploaded_pdfs": uploaded_pdfs,
            "selected_pdf": selected_pdf,
        },
    )


def flatten_result_view(request):
    output = _fetch_output(request, "flattened_pdf_id")
    if not output or not output.exists_on_disk():
        messages.error(request, _("Flattened file not found."))
        return redirect("dashboard")

    return render(
        request,
        "pdfeditor/flatten_result.html",
        {
            "pdf_filename": output.name,
            "size": os.path.getsize(output.path),
        },
    )


def download_flattened_view(request):
    output = _fetch_output(request, "flattened_pdf_id")
    if not output:
        messages.error(request, _("File not found."))
        return redirect("dashboard")
    try:
        return attachment_response(output.path)
    except Http404:
        messages.error(request, _("File not found."))
        return redirect("dashboard")


# ---------- Redact ----------


@auth_aware_ratelimit(anon_rate="20/h", user_rate="100/h", method="POST")
def redact_view(request):
    selected_pdf, uploaded_pdfs, early = _resolve_pdf_or_redirect(request)
    if early:
        return early

    pdf_path = selected_pdf.path

    if request.method == "POST":
        form = RedactPDFForm(request.POST)
        if form.is_valid():
            try:
                output_path, match_count = redact_text(
                    pdf_path,
                    search_terms=form.cleaned_data["search_terms"],
                    page_range=form.cleaned_data.get("page_range") or None,
                )
                output = record_output(
                    request,
                    kind=ProcessedPDF.KIND_REDACT,
                    path=output_path,
                    source=selected_pdf,
                )
                request.session["redacted_pdf_id"] = str(output.id)
                request.session["redacted_match_count"] = match_count
                if match_count == 0:
                    messages.warning(
                        request,
                        _("No matches were found — output is identical to the source."),
                    )
                else:
                    messages.success(
                        request,
                        _("Redacted %(count)s match(es) successfully!") % {"count": match_count},
                    )
                return redirect("redact_result")
            except ValueError as e:
                messages.error(request, _("Error: %(err)s") % {"err": e})
            except Exception as e:
                messages.error(request, _("Error redacting PDF: %(err)s") % {"err": e})
    else:
        form = RedactPDFForm()

    return render(
        request,
        "pdfeditor/redact.html",
        {
            "form": form,
            "pdf_name": selected_pdf.name,
            "pdf_path_relative": os.path.relpath(pdf_path, settings.MEDIA_ROOT),
            "uploaded_pdfs": uploaded_pdfs,
            "selected_pdf": selected_pdf,
        },
    )


def redact_result_view(request):
    output = _fetch_output(request, "redacted_pdf_id")
    if not output or not output.exists_on_disk():
        messages.error(request, _("Redacted file not found."))
        return redirect("dashboard")

    return render(
        request,
        "pdfeditor/redact_result.html",
        {
            "pdf_filename": output.name,
            "size": os.path.getsize(output.path),
            "match_count": request.session.get("redacted_match_count", 0),
        },
    )


def download_redacted_view(request):
    output = _fetch_output(request, "redacted_pdf_id")
    if not output:
        messages.error(request, _("File not found."))
        return redirect("dashboard")
    try:
        return attachment_response(output.path)
    except Http404:
        messages.error(request, _("File not found."))
        return redirect("dashboard")


# ---------- Searchable PDF (OCR layer) ----------


@auth_aware_ratelimit(anon_rate="10/h", user_rate="40/h", method="POST")
def searchable_view(request):
    selected_pdf, uploaded_pdfs, early = _resolve_pdf_or_redirect(request)
    if early:
        return early

    pdf_path = selected_pdf.path

    if request.method == "POST":
        form = MakeSearchableForm(request.POST)
        if form.is_valid():
            try:
                output_path, pages_ocrd = make_pdf_searchable(
                    pdf_path,
                    language=form.cleaned_data["language"],
                    dpi=form.cleaned_data["dpi"],
                )
                output = record_output(
                    request,
                    kind=ProcessedPDF.KIND_OCR_LAYER,
                    path=output_path,
                    source=selected_pdf,
                )
                request.session["searchable_pdf_id"] = str(output.id)
                request.session["searchable_pages_ocrd"] = pages_ocrd
                if pages_ocrd == 0:
                    messages.info(
                        request,
                        _("All pages already had a text layer — nothing to OCR."),
                    )
                else:
                    messages.success(
                        request,
                        _("Added OCR text layer on %(count)s page(s).") % {"count": pages_ocrd},
                    )
                return redirect("searchable_result")
            except ValueError as e:
                messages.error(request, _("Error: %(err)s") % {"err": e})
            except Exception as e:
                messages.error(request, _("Error running OCR: %(err)s") % {"err": e})
    else:
        form = MakeSearchableForm()

    return render(
        request,
        "pdfeditor/searchable.html",
        {
            "form": form,
            "pdf_name": selected_pdf.name,
            "pdf_path_relative": os.path.relpath(pdf_path, settings.MEDIA_ROOT),
            "uploaded_pdfs": uploaded_pdfs,
            "selected_pdf": selected_pdf,
        },
    )


def searchable_result_view(request):
    output = _fetch_output(request, "searchable_pdf_id")
    if not output or not output.exists_on_disk():
        messages.error(request, _("Searchable file not found."))
        return redirect("dashboard")

    return render(
        request,
        "pdfeditor/searchable_result.html",
        {
            "pdf_filename": output.name,
            "size": os.path.getsize(output.path),
            "pages_ocrd": request.session.get("searchable_pages_ocrd", 0),
        },
    )


def download_searchable_view(request):
    output = _fetch_output(request, "searchable_pdf_id")
    if not output:
        messages.error(request, _("File not found."))
        return redirect("dashboard")
    try:
        return attachment_response(output.path)
    except Http404:
        messages.error(request, _("File not found."))
        return redirect("dashboard")


# ---------- PDF/A conversion ----------


@auth_aware_ratelimit(anon_rate="10/h", user_rate="40/h", method="POST")
def pdfa_view(request):
    selected_pdf, uploaded_pdfs, early = _resolve_pdf_or_redirect(request)
    if early:
        return early

    pdf_path = selected_pdf.path

    if request.method == "POST":
        form = PdfaForm(request.POST)
        if form.is_valid():
            try:
                output_path, version = convert_to_pdfa(
                    pdf_path,
                    version=form.cleaned_data["version"],
                )
                output = record_output(
                    request,
                    kind=ProcessedPDF.KIND_PDFA,
                    path=output_path,
                    source=selected_pdf,
                )
                request.session["pdfa_pdf_id"] = str(output.id)
                request.session["pdfa_version"] = version
                messages.success(
                    request,
                    _("PDF converted to PDF/A-%(ver)s.") % {"ver": version},
                )
                return redirect("pdfa_result")
            except ValueError as e:
                messages.error(request, _("Error: %(err)s") % {"err": e})
            except Exception as e:
                messages.error(request, _("Error converting to PDF/A: %(err)s") % {"err": e})
    else:
        form = PdfaForm()

    return render(
        request,
        "pdfeditor/pdfa.html",
        {
            "form": form,
            "pdf_name": selected_pdf.name,
            "pdf_path_relative": os.path.relpath(pdf_path, settings.MEDIA_ROOT),
            "uploaded_pdfs": uploaded_pdfs,
            "selected_pdf": selected_pdf,
        },
    )


def pdfa_result_view(request):
    output = _fetch_output(request, "pdfa_pdf_id")
    if not output or not output.exists_on_disk():
        messages.error(request, _("PDF/A file not found."))
        return redirect("dashboard")

    return render(
        request,
        "pdfeditor/pdfa_result.html",
        {
            "pdf_filename": output.name,
            "size": os.path.getsize(output.path),
            "version": request.session.get("pdfa_version", ""),
        },
    )


def download_pdfa_view(request):
    output = _fetch_output(request, "pdfa_pdf_id")
    if not output:
        messages.error(request, _("File not found."))
        return redirect("dashboard")
    try:
        return attachment_response(output.path)
    except Http404:
        messages.error(request, _("File not found."))
        return redirect("dashboard")


# ---------- Compare PDFs ----------


@auth_aware_ratelimit(anon_rate="20/h", user_rate="100/h", method="POST")
def compare_view(request):
    selected_pdf, uploaded_pdfs, early = _resolve_pdf_or_redirect(request)
    if early:
        return early

    others = [p for p in uploaded_pdfs if p.id != selected_pdf.id]
    if not others:
        messages.error(
            request,
            _("You need at least 2 uploaded PDFs to run a comparison."),
        )
        return redirect("dashboard")

    pdf_choices = [(str(p.id), f"{p.name} • {p.size} bytes") for p in others]

    if request.method == "POST":
        form = ComparePdfsForm(request.POST, pdf_choices=pdf_choices)
        if form.is_valid():
            second = get_pdf_by_id(request, form.cleaned_data["second_pdf"])
            if not second:
                messages.error(request, _("Selected PDF not found."))
                return redirect("compare")
            try:
                output_path, stats = compare_pdfs(selected_pdf.path, second.path)
                output = record_output(
                    request,
                    kind=ProcessedPDF.KIND_COMPARE,
                    path=output_path,
                    source=selected_pdf,
                )
                request.session["compare_pdf_id"] = str(output.id)
                request.session["compare_stats"] = stats
                request.session["compare_second_name"] = second.name
                total_changes = stats["changed"] + stats["added"] + stats["removed"]
                if total_changes == 0:
                    messages.info(request, _("No textual differences found between the two PDFs."))
                else:
                    messages.success(
                        request,
                        _("Comparison complete — %(n)s differing page(s).") % {"n": total_changes},
                    )
                return redirect("compare_result")
            except ValueError as e:
                messages.error(request, _("Error: %(err)s") % {"err": e})
            except Exception as e:
                messages.error(request, _("Error comparing PDFs: %(err)s") % {"err": e})
    else:
        form = ComparePdfsForm(pdf_choices=pdf_choices)

    return render(
        request,
        "pdfeditor/compare.html",
        {
            "form": form,
            "pdf_name": selected_pdf.name,
            "pdf_path_relative": os.path.relpath(selected_pdf.path, settings.MEDIA_ROOT),
            "uploaded_pdfs": uploaded_pdfs,
            "selected_pdf": selected_pdf,
        },
    )


def compare_result_view(request):
    output = _fetch_output(request, "compare_pdf_id")
    if not output or not output.exists_on_disk():
        messages.error(request, _("Comparison report not found."))
        return redirect("dashboard")

    return render(
        request,
        "pdfeditor/compare_result.html",
        {
            "pdf_filename": output.name,
            "size": os.path.getsize(output.path),
            "stats": request.session.get("compare_stats", {}),
            "second_name": request.session.get("compare_second_name", ""),
        },
    )


def download_compare_view(request):
    output = _fetch_output(request, "compare_pdf_id")
    if not output:
        messages.error(request, _("File not found."))
        return redirect("dashboard")
    try:
        return attachment_response(output.path)
    except Http404:
        messages.error(request, _("File not found."))
        return redirect("dashboard")


# ---------- Digital signature ----------


@auth_aware_ratelimit(anon_rate="10/h", user_rate="40/h", method="POST")
def sign_view(request):
    selected_pdf, uploaded_pdfs, early = _resolve_pdf_or_redirect(request)
    if early:
        return early

    pdf_path = selected_pdf.path

    if request.method == "POST":
        form = SignPDFForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                p12_file = form.cleaned_data["p12_file"]
                p12_bytes = p12_file.read()
                tsa_url = settings.PDF_SIGN_TSA_URL if form.cleaned_data.get("add_timestamp") else None
                output_path = sign_pdf(
                    pdf_path,
                    p12_bytes=p12_bytes,
                    p12_password=form.cleaned_data["p12_password"],
                    page=form.cleaned_data["page"],
                    position=form.cleaned_data["position"],
                    reason=form.cleaned_data.get("reason", ""),
                    location=form.cleaned_data.get("location", ""),
                    tsa_url=tsa_url,
                    embed_validation_info=form.cleaned_data.get("embed_validation_info", False),
                    add_doc_timestamp=form.cleaned_data.get("add_doc_timestamp", False),
                )
                output = record_output(
                    request,
                    kind=ProcessedPDF.KIND_SIGN,
                    path=output_path,
                    source=selected_pdf,
                )
                request.session["signed_pdf_id"] = str(output.id)
                messages.success(request, _("PDF signed successfully!"))
                return redirect("sign_result")
            except ValueError as e:
                messages.error(request, _("Error signing PDF: %(err)s") % {"err": e})
            except Exception as e:
                messages.error(request, _("Error signing PDF: %(err)s") % {"err": e})
    else:
        form = SignPDFForm()

    return render(
        request,
        "pdfeditor/sign.html",
        {
            "form": form,
            "pdf_name": selected_pdf.name,
            "pdf_path_relative": os.path.relpath(pdf_path, settings.MEDIA_ROOT),
            "uploaded_pdfs": uploaded_pdfs,
            "selected_pdf": selected_pdf,
        },
    )


def sign_result_view(request):
    output = _fetch_output(request, "signed_pdf_id")
    if not output or not output.exists_on_disk():
        messages.error(request, _("Signed file not found."))
        return redirect("dashboard")

    return render(
        request,
        "pdfeditor/sign_result.html",
        {
            "signed_filename": output.name,
            "size": os.path.getsize(output.path),
        },
    )


def download_signed_view(request):
    output = _fetch_output(request, "signed_pdf_id")
    if not output:
        messages.error(request, _("File not found."))
        return redirect("dashboard")
    try:
        return attachment_response(output.path)
    except Http404:
        messages.error(request, _("File not found."))
        return redirect("dashboard")


# ---------- Self-signed certificate generator ----------


def _build_self_signed_p12(common_name: str, passphrase: str) -> bytes:
    """Generate a self-signed PKCS#12 archive (test/demo use only)."""
    from datetime import datetime, timedelta, timezone

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.serialization import pkcs12
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=365))
        .sign(key, hashes.SHA256())
    )
    return pkcs12.serialize_key_and_certificates(
        common_name.encode("utf-8")[:64] or b"signer",
        key,
        cert,
        [],
        serialization.BestAvailableEncryption(passphrase.encode("utf-8")),
    )


@auth_aware_ratelimit(anon_rate="5/h", user_rate="20/h", method="POST")
def generate_cert_view(request):
    """Generate a self-signed PKCS#12 archive and return it as a download."""
    if request.method == "POST":
        form = GenerateCertForm(request.POST)
        if form.is_valid():
            try:
                pfx = _build_self_signed_p12(
                    common_name=form.cleaned_data["common_name"],
                    passphrase=form.cleaned_data["passphrase"],
                )
            except Exception as e:
                messages.error(request, _("Error generating certificate: %(err)s") % {"err": e})
            else:
                from django.http import HttpResponse

                resp = HttpResponse(pfx, content_type="application/x-pkcs12")
                resp["Content-Disposition"] = 'attachment; filename="signer.p12"'
                return resp
    else:
        form = GenerateCertForm()

    return render(request, "pdfeditor/generate_cert.html", {"form": form})


# ---------- Convert to DOCX ----------


@auth_aware_ratelimit(anon_rate="10/h", user_rate="40/h", method="POST")
def convert_view(request):
    selected_pdf, uploaded_pdfs, early = _resolve_pdf_or_redirect(request)
    if early:
        return early

    pdf_path = selected_pdf.path

    if request.method == "POST":
        form = ConvertToDocxForm(request.POST)
        if form.is_valid():
            try:
                output_path, has_text = convert_pdf_to_docx(pdf_path)
                output = record_output(
                    request,
                    kind=ProcessedPDF.KIND_CONVERT,
                    path=output_path,
                    source=selected_pdf,
                )
                request.session["converted_docx_id"] = str(output.id)
                request.session["converted_has_text"] = has_text
                if has_text:
                    messages.success(request, _("PDF converted to Word successfully!"))
                else:
                    messages.warning(
                        request,
                        _(
                            "Conversion done, but the source PDF appears to be image-only "
                            "(scan). The .docx will be near-empty — run OCR first for better results."
                        ),
                    )
                return redirect("convert_result")
            except ValueError as e:
                messages.error(request, _("Error: %(err)s") % {"err": e})
            except Exception as e:
                messages.error(request, _("Error converting PDF: %(err)s") % {"err": e})
    else:
        form = ConvertToDocxForm()

    return render(
        request,
        "pdfeditor/convert.html",
        {
            "form": form,
            "pdf_name": selected_pdf.name,
            "pdf_path_relative": os.path.relpath(pdf_path, settings.MEDIA_ROOT),
            "uploaded_pdfs": uploaded_pdfs,
            "selected_pdf": selected_pdf,
        },
    )


def convert_result_view(request):
    output = _fetch_output(request, "converted_docx_id")
    if not output or not output.exists_on_disk():
        messages.error(request, _("Converted file not found."))
        return redirect("dashboard")

    return render(
        request,
        "pdfeditor/convert_result.html",
        {
            "converted_filename": output.name,
            "size": os.path.getsize(output.path),
            "has_text": request.session.get("converted_has_text", True),
        },
    )


def download_converted_view(request):
    output = _fetch_output(request, "converted_docx_id")
    if not output:
        messages.error(request, _("File not found."))
        return redirect("dashboard")
    try:
        return attachment_response(output.path)
    except Http404:
        messages.error(request, _("File not found."))
        return redirect("dashboard")


# ---------- PDF → images ----------


@auth_aware_ratelimit(anon_rate="20/h", user_rate="100/h", method="POST")
def to_images_view(request):
    selected_pdf, uploaded_pdfs, early = _resolve_pdf_or_redirect(request)
    if early:
        return early

    pdf_path = selected_pdf.path

    if request.method == "POST":
        form = PdfToImagesForm(request.POST)
        if form.is_valid():
            try:
                output_path, page_count = convert_pdf_to_images(
                    pdf_path,
                    fmt=form.cleaned_data["fmt"],
                    dpi=form.cleaned_data["dpi"],
                )
                output = record_output(
                    request,
                    kind=ProcessedPDF.KIND_TO_IMAGES,
                    path=output_path,
                    source=selected_pdf,
                )
                request.session["to_images_id"] = str(output.id)
                request.session["to_images_count"] = page_count
                request.session["to_images_fmt"] = form.cleaned_data["fmt"]
                messages.success(
                    request,
                    _("PDF exported to %(count)s image(s)!") % {"count": page_count},
                )
                return redirect("to_images_result")
            except ValueError as e:
                messages.error(request, _("Error: %(err)s") % {"err": e})
            except Exception as e:
                messages.error(request, _("Error exporting PDF to images: %(err)s") % {"err": e})
    else:
        form = PdfToImagesForm()

    return render(
        request,
        "pdfeditor/to_images.html",
        {
            "form": form,
            "pdf_name": selected_pdf.name,
            "pdf_path_relative": os.path.relpath(pdf_path, settings.MEDIA_ROOT),
            "uploaded_pdfs": uploaded_pdfs,
            "selected_pdf": selected_pdf,
        },
    )


def to_images_result_view(request):
    output = _fetch_output(request, "to_images_id")
    if not output or not output.exists_on_disk():
        messages.error(request, _("Image bundle not found."))
        return redirect("dashboard")

    return render(
        request,
        "pdfeditor/to_images_result.html",
        {
            "zip_filename": output.name,
            "size": os.path.getsize(output.path),
            "page_count": request.session.get("to_images_count", 0),
            "fmt": request.session.get("to_images_fmt", "png"),
        },
    )


def download_images_view(request):
    output = _fetch_output(request, "to_images_id")
    if not output:
        messages.error(request, _("File not found."))
        return redirect("dashboard")
    try:
        return attachment_response(output.path)
    except Http404:
        messages.error(request, _("File not found."))
        return redirect("dashboard")


# ---------- Metadata editor ----------


@auth_aware_ratelimit(anon_rate="20/h", user_rate="100/h", method="POST")
def metadata_view(request):
    selected_pdf, uploaded_pdfs, early = _resolve_pdf_or_redirect(request)
    if early:
        return early

    pdf_path = selected_pdf.path

    try:
        current_meta = read_pdf_metadata(pdf_path)
    except Exception:
        current_meta = {
            "title": "",
            "author": "",
            "subject": "",
            "keywords": "",
            "creator": "",
            "producer": "",
            "creation_date": None,
            "mod_date": None,
        }

    if request.method == "POST":
        form = MetadataForm(request.POST)
        if form.is_valid():
            try:
                output_path = edit_pdf_metadata(
                    pdf_path,
                    metadata={
                        "title": form.cleaned_data["title"],
                        "author": form.cleaned_data["author"],
                        "subject": form.cleaned_data["subject"],
                        "keywords": form.cleaned_data["keywords"],
                        "creator": form.cleaned_data["creator"],
                        "producer": form.cleaned_data["producer"],
                    },
                    clear_dates=form.cleaned_data.get("clear_dates", False),
                )
                output = record_output(
                    request,
                    kind=ProcessedPDF.KIND_METADATA,
                    path=output_path,
                    source=selected_pdf,
                )
                request.session["metadata_pdf_id"] = str(output.id)
                messages.success(request, _("Metadata updated successfully!"))
                return redirect("metadata_result")
            except ValueError as e:
                messages.error(request, _("Error: %(err)s") % {"err": e})
            except Exception as e:
                messages.error(request, _("Error updating metadata: %(err)s") % {"err": e})
    else:
        form = MetadataForm(
            initial={
                "title": current_meta["title"],
                "author": current_meta["author"],
                "subject": current_meta["subject"],
                "keywords": current_meta["keywords"],
                "creator": current_meta["creator"],
                "producer": current_meta["producer"],
            }
        )

    return render(
        request,
        "pdfeditor/metadata.html",
        {
            "form": form,
            "pdf_name": selected_pdf.name,
            "pdf_path_relative": os.path.relpath(pdf_path, settings.MEDIA_ROOT),
            "uploaded_pdfs": uploaded_pdfs,
            "selected_pdf": selected_pdf,
            "current_meta": current_meta,
        },
    )


def metadata_result_view(request):
    output = _fetch_output(request, "metadata_pdf_id")
    if not output or not output.exists_on_disk():
        messages.error(request, _("Updated PDF not found."))
        return redirect("dashboard")

    try:
        new_meta = read_pdf_metadata(output.path)
    except Exception:
        new_meta = None

    return render(
        request,
        "pdfeditor/metadata_result.html",
        {
            "pdf_filename": output.name,
            "size": os.path.getsize(output.path),
            "new_meta": new_meta,
        },
    )


def download_metadata_view(request):
    output = _fetch_output(request, "metadata_pdf_id")
    if not output:
        messages.error(request, _("File not found."))
        return redirect("dashboard")
    try:
        return attachment_response(output.path)
    except Http404:
        messages.error(request, _("File not found."))
        return redirect("dashboard")


# ---------- Images → PDF ----------


_IMAGES_TO_PDF_MAX_FILES = 50
_IMAGES_TO_PDF_MAX_BYTES_PER_FILE = 15 * 1024 * 1024  # 15 MB
_IMAGES_TO_PDF_ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


@auth_aware_ratelimit(anon_rate="20/h", user_rate="100/h", method="POST")
def images_to_pdf_view(request):
    if request.method == "POST":
        form = ImagesToPdfForm(request.POST)
        uploaded = request.FILES.getlist("images")
        if not uploaded:
            messages.error(request, _("Please select at least one image."))
        elif len(uploaded) > _IMAGES_TO_PDF_MAX_FILES:
            messages.error(
                request,
                _("Too many images — limit is %(max)d per export.") % {"max": _IMAGES_TO_PDF_MAX_FILES},
            )
        elif form.is_valid():
            valid_files: list = []
            for f in uploaded:
                ext = os.path.splitext(f.name)[1].lower()
                if ext not in _IMAGES_TO_PDF_ALLOWED_EXT:
                    messages.warning(
                        request,
                        _('Skipped "%(name)s" — unsupported image format.') % {"name": f.name},
                    )
                    continue
                if f.size > _IMAGES_TO_PDF_MAX_BYTES_PER_FILE:
                    messages.warning(
                        request,
                        _('Skipped "%(name)s" — exceeds %(mb)d MB limit.')
                        % {
                            "name": f.name,
                            "mb": _IMAGES_TO_PDF_MAX_BYTES_PER_FILE // (1024 * 1024),
                        },
                    )
                    continue
                valid_files.append(f)

            if not valid_files:
                messages.error(request, _("No valid images were uploaded."))
            else:
                # Apply user-supplied order (CSV of original indices); fall back
                # to upload order if the order is missing or partial.
                order = form.cleaned_data.get("images_order") or []
                ordered_files = []
                if order and all(0 <= i < len(uploaded) for i in order):
                    # Translate indices: ``order`` references the original
                    # ``uploaded`` list, not the post-validation ``valid_files``.
                    valid_set = {id(f) for f in valid_files}
                    for i in order:
                        f = uploaded[i]
                        if id(f) in valid_set:
                            ordered_files.append(f)
                if not ordered_files:
                    ordered_files = valid_files

                import tempfile

                staging_dir = tempfile.mkdtemp(prefix="img2pdf_")
                staged_paths: list[str] = []
                try:
                    for idx, f in enumerate(ordered_files):
                        # Preserve extension to help PIL identify the format.
                        ext = os.path.splitext(f.name)[1].lower()
                        staged_path = os.path.join(staging_dir, f"{idx:03d}{ext}")
                        with open(staged_path, "wb") as out:
                            for chunk in f.chunks():
                                out.write(chunk)
                        staged_paths.append(staged_path)

                    try:
                        output_path, count = convert_images_to_pdf(
                            staged_paths,
                            page_size=form.cleaned_data["page_size"],
                            fit_mode=form.cleaned_data["fit_mode"],
                        )
                    except ValueError as e:
                        messages.error(request, _("Error: %(err)s") % {"err": e})
                    except Exception as e:
                        messages.error(
                            request,
                            _("Error building PDF from images: %(err)s") % {"err": e},
                        )
                    else:
                        output = record_output(
                            request,
                            kind=ProcessedPDF.KIND_IMAGES_TO_PDF,
                            path=output_path,
                            source=None,
                        )
                        request.session["images_to_pdf_id"] = str(output.id)
                        request.session["images_to_pdf_count"] = count
                        messages.success(
                            request,
                            _("PDF built from %(count)s image(s)!") % {"count": count},
                        )
                        return redirect("images_to_pdf_result")
                finally:
                    import shutil as _shutil

                    _shutil.rmtree(staging_dir, ignore_errors=True)
    else:
        form = ImagesToPdfForm()

    return render(
        request,
        "pdfeditor/images_to_pdf.html",
        {
            "form": form,
            "max_files": _IMAGES_TO_PDF_MAX_FILES,
            "max_mb": _IMAGES_TO_PDF_MAX_BYTES_PER_FILE // (1024 * 1024),
        },
    )


def images_to_pdf_result_view(request):
    output = _fetch_output(request, "images_to_pdf_id")
    if not output or not output.exists_on_disk():
        messages.error(request, _("Built PDF not found."))
        return redirect("dashboard")

    return render(
        request,
        "pdfeditor/images_to_pdf_result.html",
        {
            "pdf_filename": output.name,
            "size": os.path.getsize(output.path),
            "image_count": request.session.get("images_to_pdf_count", 0),
        },
    )


def download_images_to_pdf_view(request):
    output = _fetch_output(request, "images_to_pdf_id")
    if not output:
        messages.error(request, _("File not found."))
        return redirect("dashboard")
    try:
        return attachment_response(output.path)
    except Http404:
        messages.error(request, _("File not found."))
        return redirect("dashboard")


# ---------- Verify signatures ----------


@auth_aware_ratelimit(anon_rate="20/h", user_rate="100/h", method="POST")
def verify_signature_view(request):
    """Upload a signed PDF and report on its embedded signatures."""
    reports = None
    if request.method == "POST":
        form = VerifyPDFForm(request.POST, request.FILES)
        if form.is_valid():
            pdf_file = form.cleaned_data["pdf_file"]
            trust_file = form.cleaned_data.get("trust_certs")
            extra_certs: list[bytes] = []
            if trust_file:
                extra_certs.append(trust_file.read())
            # Merge in every globally-active trust anchor stored by an admin.
            for anchor in TrustAnchor.objects.filter(is_active=True):
                extra_certs.append(anchor.cert_pem.encode("utf-8"))

            import tempfile

            fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
            try:
                with os.fdopen(fd, "wb") as out:
                    for chunk in pdf_file.chunks():
                        out.write(chunk)
                try:
                    reports = verify_pdf_signatures(tmp_path, extra_trust_certs=extra_certs)
                except Exception as e:
                    messages.error(request, _("Error verifying signatures: %(err)s") % {"err": e})
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            if reports is not None and not reports:
                messages.warning(request, _("No signatures found in this PDF."))
    else:
        form = VerifyPDFForm()

    return render(
        request,
        "pdfeditor/verify_signature.html",
        {
            "form": form,
            "reports": reports,
        },
    )
