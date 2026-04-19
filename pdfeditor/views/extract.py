"""Text extraction + OCR + more-tools AJAX endpoints."""
from django.contrib import messages
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render

from ..pdf_processor import extract_text_from_pdf, ocr_pdf_to_text
from ._common import get_pdf_by_id, get_uploaded_pdfs


def more_tools_view(request):
    uploaded_pdfs = get_uploaded_pdfs(request)
    if not uploaded_pdfs:
        messages.error(request, 'No PDF found. Please upload a PDF first.')
        return redirect('dashboard')

    return render(request, 'pdfeditor/more_tools.html', {'uploaded_pdfs': uploaded_pdfs})


def _extract_to_session(request, pdf_id, extractor, suffix):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Invalid method'})

    pdf = get_pdf_by_id(request, pdf_id)
    if not pdf:
        return JsonResponse({'success': False, 'error': 'PDF not found'})

    try:
        text = extractor(pdf['path'])
        request.session['extracted_text'] = text
        request.session['extracted_filename'] = pdf['name'].replace('.pdf', suffix)
        return JsonResponse({
            'success': True,
            'text': text,
            'filename': pdf['name'],
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


def extract_text_ajax(request, pdf_id):
    return _extract_to_session(request, pdf_id, extract_text_from_pdf, '_extracted.txt')


def ocr_text_ajax(request, pdf_id):
    return _extract_to_session(request, pdf_id, ocr_pdf_to_text, '_ocr.txt')


def download_text_view(request):
    text = request.session.get('extracted_text')
    filename = request.session.get('extracted_filename', 'extracted.txt')

    if not text:
        messages.error(request, 'No text found to download.')
        return redirect('dashboard')

    response = HttpResponse(text, content_type='text/plain')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response
