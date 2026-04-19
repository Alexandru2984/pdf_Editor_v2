"""Find & replace views + preview/result/download."""
import os

from django.conf import settings
from django.contrib import messages
from django.http import FileResponse, Http404
from django.shortcuts import redirect, render

from ..forms import FindReplaceForm
from ..pdf_processor import find_and_replace_text
from ._common import get_pdf_by_id, get_uploaded_pdfs


def _resolve_selected_pdf(request):
    """Return (selected_pdf, uploaded_pdfs) or a redirect response."""
    uploaded_pdfs = get_uploaded_pdfs(request)
    if not uploaded_pdfs:
        messages.error(request, 'No PDF found. Please upload a PDF first.')
        return None, None, redirect('dashboard')

    pdf_id = request.GET.get('pdf')
    if pdf_id:
        selected_pdf = get_pdf_by_id(request, pdf_id)
        if not selected_pdf:
            messages.error(request, 'Selected PDF not found.')
            return None, None, redirect('dashboard')
    else:
        selected_pdf = uploaded_pdfs[0]

    return selected_pdf, uploaded_pdfs, None


def edit_view(request):
    selected_pdf, uploaded_pdfs, early = _resolve_selected_pdf(request)
    if early:
        return early

    pdf_path = selected_pdf['path']

    if request.method == 'POST':
        form = FindReplaceForm(request.POST)
        if form.is_valid():
            page_range = form.cleaned_data.get('page_range', '').strip()
            try:
                output_path, count, warnings = find_and_replace_text(
                    pdf_path=pdf_path,
                    search_text=form.cleaned_data['search_text'],
                    replace_text=form.cleaned_data['replace_text'],
                    case_sensitive=form.cleaned_data['case_sensitive'],
                    page_range=page_range or None,
                )
                request.session['processed_pdf_path'] = output_path
                request.session['replacement_count'] = count
                request.session['warnings'] = warnings
                return redirect('result')
            except ValueError as e:
                messages.error(request, f'Error: {e}')
            except Exception as e:
                messages.error(request, f'Error processing PDF: {e}')
    else:
        form = FindReplaceForm()

    return render(request, 'pdfeditor/edit.html', {
        'form': form,
        'pdf_name': selected_pdf['name'],
        'pdf_path_relative': os.path.relpath(pdf_path, settings.MEDIA_ROOT),
        'uploaded_pdfs': uploaded_pdfs,
        'selected_pdf': selected_pdf,
    })


def result_view(request):
    processed_pdf_path = request.session.get('processed_pdf_path')
    if not processed_pdf_path or not os.path.exists(processed_pdf_path):
        messages.error(request, 'Processed file not found.')
        return redirect('dashboard')

    warnings = request.session.get('warnings', [])
    return render(request, 'pdfeditor/result.html', {
        'replacement_count': request.session.get('replacement_count', 0),
        'warnings': warnings,
        'has_warnings': bool(warnings),
        'pdf_path_relative': os.path.relpath(processed_pdf_path, settings.MEDIA_ROOT),
    })


def download_view(request):
    processed_pdf_path = request.session.get('processed_pdf_path')
    if not processed_pdf_path or not os.path.exists(processed_pdf_path):
        raise Http404('PDF-ul nu a fost găsit.')

    response = FileResponse(open(processed_pdf_path, 'rb'), content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{os.path.basename(processed_pdf_path)}"'
    return response


def preview_view(request):
    pdf_type = request.GET.get('type', 'uploaded')

    if pdf_type == 'processed':
        pdf_path = request.session.get('processed_pdf_path')
        pdf_name = 'Modified PDF'
    else:
        pdf_path = request.session.get('uploaded_pdf_path')
        pdf_name = request.session.get('uploaded_pdf_name', 'Uploaded PDF')

    if not pdf_path or not os.path.exists(pdf_path):
        messages.error(request, 'PDF not found for preview.')
        return redirect('dashboard')

    return render(request, 'pdfeditor/preview.html', {
        'pdf_name': pdf_name,
        'pdf_url': f"/media/{os.path.relpath(pdf_path, settings.MEDIA_ROOT)}",
        'pdf_type': pdf_type,
    })
