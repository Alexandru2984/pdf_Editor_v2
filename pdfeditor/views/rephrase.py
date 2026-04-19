"""AI-powered rephrase views (Ollama + Groq)."""
import json
import os

from django.conf import settings
from django.contrib import messages
from django.http import Http404, HttpResponse
from django.shortcuts import redirect, render
from django_ratelimit.decorators import ratelimit

from ..ai_service import get_all_models, get_provider
from ..models import ProcessedPDF
from ..pdf_processor import rephrase_with_coordinates
from ._common import (
    attachment_response,
    ensure_session_key,
    get_pdf_by_id,
    get_uploaded_pdfs,
    record_output,
)


def _json_response(payload, status=200):
    return HttpResponse(json.dumps(payload), content_type='application/json', status=status)


def _fetch_output(request, session_key):
    output_id = request.session.get(session_key)
    if not output_id:
        return None
    return ProcessedPDF.objects.filter(
        session_key=ensure_session_key(request),
        id=output_id,
    ).first()


@ratelimit(key='ip', rate='20/h', method='POST', block=True)
def rephrase_view(request):
    uploaded_pdfs = get_uploaded_pdfs(request)
    if not uploaded_pdfs:
        messages.error(request, 'No PDF found. Please upload a PDF first.')
        return redirect('dashboard')

    pdf_id = request.GET.get('pdf')
    selected_pdf = get_pdf_by_id(request, pdf_id) if pdf_id else uploaded_pdfs[0]
    if not selected_pdf:
        messages.error(request, 'Selected PDF not found.')
        return redirect('dashboard')

    all_models = get_all_models()
    ollama_models = all_models.get('ollama', [])
    groq_models = all_models.get('groq', [])

    if request.method == 'POST':
        _handle_rephrase_post(request, selected_pdf, ollama_models, groq_models)
        if request.session.get('rephrased_pdf_id'):
            return redirect('rephrase_result')

    return render(request, 'pdfeditor/rephrase.html', {
        'pdf_name': selected_pdf.name,
        'pdf_path_relative': selected_pdf.path.replace(f"{settings.MEDIA_ROOT}/", ''),
        'uploaded_pdfs': uploaded_pdfs,
        'selected_pdf': selected_pdf,
        'ollama_models': ollama_models,
        'groq_models': groq_models,
        'ollama_connected': bool(ollama_models),
    })


def _handle_rephrase_post(request, selected_pdf, ollama_models, groq_models):
    selected_text = request.POST.get('selected_text', '').strip()
    rephrase_style = request.POST.get('rephrase_style', 'formal')
    model = request.POST.get('model') or (ollama_models[0] if ollama_models else None)

    try:
        page_number = int(request.POST.get('page_number', 0))
        bbox = {
            'x0': float(request.POST.get('bbox_x0', 0)),
            'y0': float(request.POST.get('bbox_y0', 0)),
            'x1': float(request.POST.get('bbox_x1', 0)),
            'y1': float(request.POST.get('bbox_y1', 0)),
        }
    except (TypeError, ValueError):
        messages.error(request, 'Invalid selection coordinates.')
        return

    if not selected_text:
        messages.error(request, 'Please select text from the PDF first.')
        return
    if all(v == 0 for v in bbox.values()):
        messages.error(request, 'Missing selection coordinates. Please capture the selection again.')
        return
    if not model:
        messages.error(request, 'Please select an AI model.')
        return

    provider_name = 'groq' if model in groq_models else 'ollama'
    provider = get_provider(provider_name)

    try:
        rephrased_text, success, error = provider.rephrase(
            text=selected_text, style=rephrase_style, model=model,
        )
        if not success:
            messages.error(request, f'AI Error ({provider_name}): {error}')
            return

        output_path, replacement_count, warnings = rephrase_with_coordinates(
            pdf_path=selected_pdf.path,
            page_number=page_number,
            bounding_box_bl=bbox,
            replace_text=rephrased_text,
            original_text=selected_text,
        )

        output = record_output(
            request,
            kind=ProcessedPDF.KIND_REPHRASE,
            path=output_path,
            source=selected_pdf,
        )
        request.session['rephrased_pdf_id'] = str(output.id)
        request.session['rephrase_original_text'] = selected_text
        request.session['rephrase_new_text'] = rephrased_text
        request.session['rephrase_count'] = replacement_count
        request.session['rephrase_warnings'] = warnings
        request.session['rephrase_style'] = rephrase_style
        request.session['rephrase_model'] = model
    except ValueError as e:
        messages.error(request, f'Error: {e}')
    except Exception as e:
        messages.error(request, f'Error processing PDF: {e}')


@ratelimit(key='ip', rate='30/h', method='POST', block=True)
def rephrase_preview_ajax(request):
    """Preview rephrased text without applying to the PDF."""
    if request.method != 'POST':
        return _json_response({'success': False, 'error': 'Only POST allowed'}, status=405)

    try:
        text = request.POST.get('text', '').strip()
        style = request.POST.get('style', 'formal')
        model = request.POST.get('model', '')

        if not text:
            return _json_response({'success': False, 'error': 'No text provided'}, status=400)

        groq_models = get_all_models().get('groq', [])
        provider_name = 'groq' if model in groq_models else 'ollama'
        provider = get_provider(provider_name)

        rephrased, success, error = provider.rephrase(text, style, model)
        if not success:
            return _json_response({'success': False, 'error': error})

        return _json_response({
            'success': True,
            'original_text': text,
            'rephrased_text': rephrased,
            'model': model,
            'style': style,
        })
    except Exception as e:
        return _json_response({'success': False, 'error': str(e)}, status=500)


def rephrase_result_view(request):
    output = _fetch_output(request, 'rephrased_pdf_id')
    if not output or not output.exists_on_disk():
        messages.error(request, 'Rephrased PDF not found.')
        return redirect('dashboard')

    warnings = request.session.get('rephrase_warnings', [])
    return render(request, 'pdfeditor/rephrase_result.html', {
        'rephrased_filename': output.name,
        'rephrased_size': output.size,
        'original_text': request.session.get('rephrase_original_text', ''),
        'new_text': request.session.get('rephrase_new_text', ''),
        'replacement_count': request.session.get('rephrase_count', 0),
        'warnings': warnings,
        'has_warnings': bool(warnings),
        'style': request.session.get('rephrase_style', ''),
        'model': request.session.get('rephrase_model', ''),
        'pdf_path_relative': os.path.relpath(output.path, settings.MEDIA_ROOT),
    })


def download_rephrased_view(request):
    output = _fetch_output(request, 'rephrased_pdf_id')
    if not output:
        messages.error(request, 'File not found.')
        return redirect('dashboard')
    try:
        return attachment_response(output.path)
    except Http404:
        messages.error(request, 'File not found.')
        return redirect('dashboard')
