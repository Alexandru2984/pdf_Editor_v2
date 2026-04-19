"""Watermark, rotate, page-numbers views."""
import os

from django.conf import settings
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from django.http import Http404
from django.shortcuts import redirect, render

from ..forms import PageNumbersForm, RotatePagesForm, WatermarkForm
from ..pdf_processor import add_page_numbers, add_watermark, rotate_pages
from ._common import attachment_response, get_pdf_by_id, get_uploaded_pdfs


def _require_pdf(request):
    uploaded_pdfs = get_uploaded_pdfs(request)
    if not uploaded_pdfs:
        messages.error(request, 'No PDF found. Please upload a PDF first.')
        return None, None, redirect('dashboard')

    pdf_id = request.GET.get('pdf')
    selected_pdf = get_pdf_by_id(request, pdf_id) if pdf_id else uploaded_pdfs[0]
    if not selected_pdf:
        messages.error(request, 'Selected PDF not found.')
        return None, None, redirect('dashboard')

    return selected_pdf, uploaded_pdfs, None


# ---------- Watermark ----------

def watermark_view(request):
    selected_pdf, uploaded_pdfs, early = _require_pdf(request)
    if early:
        return early

    pdf_path = selected_pdf['path']

    if request.method == 'POST':
        form = WatermarkForm(request.POST, request.FILES)
        if form.is_valid():
            options = {
                'position': form.cleaned_data['position'],
                'opacity': form.cleaned_data['opacity'],
                'rotation': form.cleaned_data['rotation'],
            }
            try:
                if form.cleaned_data['watermark_type'] == 'text':
                    options['font_size'] = form.cleaned_data.get('font_size', 48)
                    output_path = add_watermark(pdf_path, 'text', form.cleaned_data['text_content'], options)
                else:
                    uploaded_image = form.cleaned_data['watermark_image']
                    fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'temp'))
                    safe_image_name = os.path.basename(uploaded_image.name)
                    image_filename = fs.save(safe_image_name, uploaded_image)
                    image_path = fs.path(image_filename)
                    try:
                        output_path = add_watermark(pdf_path, 'image', image_path, options)
                    finally:
                        if os.path.exists(image_path):
                            os.remove(image_path)

                request.session['watermarked_pdf_path'] = output_path
                messages.success(request, 'Watermark added successfully!')
                return redirect('watermark_result')
            except Exception as e:
                messages.error(request, f'Error adding watermark: {e}')
    else:
        form = WatermarkForm()

    return render(request, 'pdfeditor/watermark.html', {
        'form': form,
        'pdf_name': selected_pdf['name'],
        'pdf_path_relative': os.path.relpath(pdf_path, settings.MEDIA_ROOT),
        'uploaded_pdfs': uploaded_pdfs,
        'selected_pdf': selected_pdf,
    })


def watermark_result_view(request):
    watermarked_path = request.session.get('watermarked_pdf_path')
    if not watermarked_path or not os.path.exists(watermarked_path):
        messages.error(request, 'Watermarked file not found.')
        return redirect('dashboard')

    return render(request, 'pdfeditor/watermark_result.html', {
        'watermarked_filename': os.path.basename(watermarked_path),
        'watermarked_size': os.path.getsize(watermarked_path),
        'pdf_path_relative': os.path.relpath(watermarked_path, settings.MEDIA_ROOT),
    })


def download_watermarked_view(request):
    try:
        return attachment_response(request.session.get('watermarked_pdf_path'))
    except Http404:
        messages.error(request, 'File not found.')
        return redirect('dashboard')


# ---------- Rotate ----------

def rotate_view(request):
    selected_pdf, uploaded_pdfs, early = _require_pdf(request)
    if early:
        return early

    pdf_path = selected_pdf['path']

    if request.method == 'POST':
        form = RotatePagesForm(request.POST)
        if form.is_valid():
            rotation_angle = int(form.cleaned_data['rotation_angle'])
            page_range = form.cleaned_data.get('page_range', '').strip()
            try:
                output_path = rotate_pages(pdf_path, rotation_angle, page_range or None)
                request.session['rotated_pdf_path'] = output_path
                request.session['rotation_angle'] = rotation_angle
                messages.success(request, f'Pages rotated {rotation_angle}° successfully!')
                return redirect('rotate_result')
            except ValueError as e:
                messages.error(request, f'Error: {e}')
            except Exception as e:
                messages.error(request, f'Error rotating pages: {e}')
    else:
        form = RotatePagesForm()

    return render(request, 'pdfeditor/rotate.html', {
        'form': form,
        'pdf_name': selected_pdf['name'],
        'pdf_path_relative': os.path.relpath(pdf_path, settings.MEDIA_ROOT),
        'uploaded_pdfs': uploaded_pdfs,
        'selected_pdf': selected_pdf,
    })


def rotate_result_view(request):
    rotated_path = request.session.get('rotated_pdf_path')
    if not rotated_path or not os.path.exists(rotated_path):
        messages.error(request, 'Rotated file not found.')
        return redirect('dashboard')

    return render(request, 'pdfeditor/rotate_result.html', {
        'rotated_filename': os.path.basename(rotated_path),
        'rotated_size': os.path.getsize(rotated_path),
        'rotation_angle': request.session.get('rotation_angle', 0),
        'pdf_path_relative': os.path.relpath(rotated_path, settings.MEDIA_ROOT),
    })


def download_rotated_view(request):
    try:
        return attachment_response(request.session.get('rotated_pdf_path'))
    except Http404:
        messages.error(request, 'File not found.')
        return redirect('dashboard')


# ---------- Page numbers ----------

def page_numbers_view(request):
    selected_pdf, uploaded_pdfs, early = _require_pdf(request)
    if early:
        return early

    pdf_path = selected_pdf['path']

    if request.method == 'POST':
        form = PageNumbersForm(request.POST)
        if form.is_valid():
            options = {
                'position': form.cleaned_data['position'],
                'format': form.cleaned_data['format'],
                'font_size': form.cleaned_data['font_size'],
                'start_page': form.cleaned_data['start_page'],
            }
            try:
                output_path = add_page_numbers(pdf_path, options)
                request.session['numbered_pdf_path'] = output_path
                messages.success(request, 'Page numbers added successfully!')
                return redirect('page_numbers_result')
            except Exception as e:
                messages.error(request, f'Error adding page numbers: {e}')
    else:
        form = PageNumbersForm()

    return render(request, 'pdfeditor/page_numbers.html', {
        'form': form,
        'pdf_name': selected_pdf['name'],
        'pdf_path_relative': os.path.relpath(pdf_path, settings.MEDIA_ROOT),
        'uploaded_pdfs': uploaded_pdfs,
        'selected_pdf': selected_pdf,
    })


def page_numbers_result_view(request):
    numbered_path = request.session.get('numbered_pdf_path')
    if not numbered_path or not os.path.exists(numbered_path):
        messages.error(request, 'Numbered file not found.')
        return redirect('dashboard')

    return render(request, 'pdfeditor/page_numbers_result.html', {
        'numbered_filename': os.path.basename(numbered_path),
        'numbered_size': os.path.getsize(numbered_path),
        'pdf_path_relative': os.path.relpath(numbered_path, settings.MEDIA_ROOT),
    })


def download_numbered_view(request):
    try:
        return attachment_response(request.session.get('numbered_pdf_path'))
    except Http404:
        messages.error(request, 'File not found.')
        return redirect('dashboard')
