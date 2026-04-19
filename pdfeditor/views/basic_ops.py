"""Split, merge, compress views."""
import os

from django.conf import settings
from django.contrib import messages
from django.http import FileResponse, Http404
from django.shortcuts import redirect, render

from ..forms import CompressPDFForm, MergePDFForm, SplitPDFForm
from ..pdf_processor import compress_pdf, merge_pdfs, split_pdf
from ._common import attachment_response, get_pdf_by_id, get_uploaded_pdfs


# ---------- Split ----------

def split_view(request):
    uploaded_pdfs = get_uploaded_pdfs(request)
    if not uploaded_pdfs:
        messages.error(request, 'No PDF found. Please upload a PDF first.')
        return redirect('dashboard')

    pdf_id = request.GET.get('pdf')
    selected_pdf = get_pdf_by_id(request, pdf_id) if pdf_id else uploaded_pdfs[0]
    if not selected_pdf:
        messages.error(request, 'Selected PDF not found.')
        return redirect('dashboard')

    if request.method == 'POST':
        form = SplitPDFForm(request.POST)
        if form.is_valid():
            try:
                output_files = split_pdf(selected_pdf['path'], form.cleaned_data['ranges'])
                request.session['split_files'] = output_files
                request.session['split_count'] = len(output_files)
                messages.success(request, f'PDF split successfully into {len(output_files)} files!')
                return redirect('split_result')
            except ValueError as e:
                messages.error(request, f'Error: {e}')
            except Exception as e:
                messages.error(request, f'Error splitting PDF: {e}')
    else:
        form = SplitPDFForm()

    return render(request, 'pdfeditor/split.html', {
        'form': form,
        'pdf_name': selected_pdf['name'],
        'pdf_path_relative': os.path.relpath(selected_pdf['path'], settings.MEDIA_ROOT),
        'uploaded_pdfs': uploaded_pdfs,
        'selected_pdf': selected_pdf,
    })


def split_result_view(request):
    split_files = request.session.get('split_files', [])
    if not split_files:
        messages.error(request, 'No split files found.')
        return redirect('dashboard')

    files_info = [
        {
            'name': os.path.basename(p),
            'path': p,
            'path_relative': os.path.relpath(p, settings.MEDIA_ROOT),
            'size': os.path.getsize(p),
        }
        for p in split_files if os.path.exists(p)
    ]

    return render(request, 'pdfeditor/split_result.html', {
        'files_info': files_info,
        'split_count': request.session.get('split_count', 0),
    })


def download_split_file_view(request):
    file_index = request.GET.get('file')
    split_files = request.session.get('split_files', [])

    if file_index is None or not split_files:
        raise Http404('File not found')

    try:
        idx = int(file_index)
    except (TypeError, ValueError):
        raise Http404('Invalid file index')

    if idx < 0 or idx >= len(split_files):
        raise Http404('File index out of range')

    file_path = split_files[idx]
    if not os.path.exists(file_path):
        raise Http404('File not found on disk')

    response = FileResponse(open(file_path, 'rb'), content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
    return response


# ---------- Merge ----------

def merge_view(request):
    uploaded_pdfs = get_uploaded_pdfs(request)
    if len(uploaded_pdfs) < 2:
        messages.error(request, 'You need at least 2 PDFs to merge. Please upload more files.')
        return redirect('dashboard')

    if request.method == 'POST':
        form = MergePDFForm(request.POST)
        if form.is_valid():
            try:
                pdf_paths = []
                for pdf_id in form.cleaned_data['selected_pdfs']:
                    pdf = get_pdf_by_id(request, pdf_id)
                    if not pdf:
                        messages.error(request, f'PDF with ID {pdf_id} not found.')
                        return redirect('merge')
                    pdf_paths.append(pdf['path'])

                if len(pdf_paths) < 2:
                    messages.error(request, 'At least 2 PDFs are required for merging.')
                    return redirect('merge')

                merged_path = merge_pdfs(pdf_paths, form.cleaned_data.get('output_name'))
                request.session['merged_pdf_path'] = merged_path
                request.session['merged_pdf_count'] = len(pdf_paths)
                messages.success(request, f'Successfully merged {len(pdf_paths)} PDFs!')
                return redirect('merge_result')
            except ValueError as e:
                messages.error(request, f'Error: {e}')
            except Exception as e:
                messages.error(request, f'Error merging PDFs: {e}')
    else:
        form = MergePDFForm()

    return render(request, 'pdfeditor/merge.html', {
        'form': form,
        'uploaded_pdfs': uploaded_pdfs,
    })


def merge_result_view(request):
    merged_path = request.session.get('merged_pdf_path')
    if not merged_path or not os.path.exists(merged_path):
        messages.error(request, 'Merged file not found.')
        return redirect('dashboard')

    return render(request, 'pdfeditor/merge_result.html', {
        'merged_filename': os.path.basename(merged_path),
        'merged_size': os.path.getsize(merged_path),
        'merged_count': request.session.get('merged_pdf_count', 0),
        'pdf_path_relative': os.path.relpath(merged_path, settings.MEDIA_ROOT),
    })


def download_merged_view(request):
    try:
        return attachment_response(request.session.get('merged_pdf_path'))
    except Http404:
        messages.error(request, 'File not found.')
        return redirect('dashboard')


# ---------- Compress ----------

def compress_view(request):
    uploaded_pdfs = get_uploaded_pdfs(request)
    if not uploaded_pdfs:
        messages.error(request, 'No PDF found. Please upload a PDF first.')
        return redirect('dashboard')

    pdf_id = request.GET.get('pdf')
    selected_pdf = get_pdf_by_id(request, pdf_id) if pdf_id else uploaded_pdfs[0]
    if not selected_pdf:
        messages.error(request, 'Selected PDF not found.')
        return redirect('dashboard')

    pdf_path = selected_pdf['path']

    if request.method == 'POST':
        form = CompressPDFForm(request.POST)
        if form.is_valid():
            try:
                output_path, original_size, compressed_size, ratio = compress_pdf(
                    pdf_path, quality=form.cleaned_data['quality'],
                )
                request.session['compressed_pdf_path'] = output_path
                request.session['original_size'] = original_size
                request.session['compressed_size'] = compressed_size
                request.session['compression_ratio'] = ratio
                messages.success(request, f'PDF compressed successfully! Saved {ratio:.1f}% space.')
                return redirect('compress_result')
            except Exception as e:
                messages.error(request, f'Error compressing PDF: {e}')
    else:
        form = CompressPDFForm()

    return render(request, 'pdfeditor/compress.html', {
        'form': form,
        'pdf_name': selected_pdf['name'],
        'pdf_path_relative': os.path.relpath(pdf_path, settings.MEDIA_ROOT),
        'uploaded_pdfs': uploaded_pdfs,
        'selected_pdf': selected_pdf,
        'original_size': os.path.getsize(pdf_path),
    })


def compress_result_view(request):
    compressed_path = request.session.get('compressed_pdf_path')
    if not compressed_path or not os.path.exists(compressed_path):
        messages.error(request, 'Compressed file not found.')
        return redirect('dashboard')

    original_size = request.session.get('original_size', 0)
    compressed_size = request.session.get('compressed_size', 0)
    return render(request, 'pdfeditor/compress_result.html', {
        'compressed_filename': os.path.basename(compressed_path),
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': request.session.get('compression_ratio', 0),
        'saved_bytes': original_size - compressed_size,
        'pdf_path_relative': os.path.relpath(compressed_path, settings.MEDIA_ROOT),
    })


def download_compressed_view(request):
    try:
        return attachment_response(request.session.get('compressed_pdf_path'))
    except Http404:
        messages.error(request, 'File not found.')
        return redirect('dashboard')
