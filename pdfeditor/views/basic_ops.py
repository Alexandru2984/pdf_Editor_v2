"""Split, merge, compress, password-protect, digital-sign views."""

import os

from django.conf import settings
from django.contrib import messages
from django.http import Http404
from django.shortcuts import redirect, render
from django.utils.translation import gettext as _

from ..forms import (
    CompressPDFForm,
    GenerateCertForm,
    MergePDFForm,
    ProtectPDFForm,
    SignPDFForm,
    SplitPDFForm,
    VerifyPDFForm,
)
from ..models import ProcessedPDF
from ..pdf_processor import (
    compress_pdf,
    merge_pdfs,
    protect_pdf,
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
