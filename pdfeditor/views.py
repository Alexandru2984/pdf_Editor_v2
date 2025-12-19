"""
Views pentru aplicația PDF Editor.
"""
import os
import uuid
from datetime import datetime
from django.shortcuts import render, redirect
from django.http import FileResponse, Http404, HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.contrib import messages

from .forms import FindReplaceForm, SplitPDFForm, MergePDFForm, CompressPDFForm, WatermarkForm, RotatePagesForm, PageNumbersForm
from .pdf_processor import find_and_replace_text, check_pdf_has_text, split_pdf, merge_pdfs, compress_pdf, add_watermark, rotate_pages, add_page_numbers, extract_text_from_pdf, ocr_pdf_to_text


def get_uploaded_pdfs(request):
    """Get list of uploaded PDFs from session, filtering out deleted files."""
    uploaded_pdfs = request.session.get('uploaded_pdfs', [])
    valid_pdfs = [pdf for pdf in uploaded_pdfs if os.path.exists(pdf['path'])]
    
    # Update session if any were filtered out
    if len(valid_pdfs) != len(uploaded_pdfs):
        request.session['uploaded_pdfs'] = valid_pdfs
    
    return valid_pdfs


def get_pdf_by_id(request, pdf_id):
    """Get specific PDF by ID from session."""
    uploaded_pdfs = get_uploaded_pdfs(request)
    for pdf in uploaded_pdfs:
        if pdf['id'] == pdf_id:
            return pdf
    return None


def dashboard_view(request):
    """View for main dashboard - central hub."""
    uploaded_pdfs = get_uploaded_pdfs(request)
    
    context = {
        'uploaded_pdfs': uploaded_pdfs
    }
    return render(request, 'pdfeditor/dashboard.html', context)


def upload_view(request):
    """View for uploading one or more PDF files."""
    if request.method == 'POST':
        uploaded_files = request.FILES.getlist('pdf_file')
        
        if not uploaded_files:
            messages.error(request, 'Please select at least one PDF file.')
            return render(request, 'pdfeditor/upload.html')
        
        # Get or initialize PDF list in session
        uploaded_pdfs = request.session.get('uploaded_pdfs', [])
        
        uploaded_count = 0
        for uploaded_file in uploaded_files:
            if not uploaded_file.name.lower().endswith('.pdf'):
                messages.warning(request, f'Skipped "{uploaded_file.name}" - only PDF files are accepted.')
                continue
            
            # Save file
            fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploads'))
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_path = fs.path(filename)
            
            # Check for text
            has_text, message = check_pdf_has_text(file_path)
            if not has_text:
                messages.warning(request, f'{uploaded_file.name}: {message}')
            
            # Generate unique ID
            
            
            pdf_data = {
                'id': str(uuid.uuid4()),
                'path': file_path,
                'name': uploaded_file.name,
                'size': uploaded_file.size,
                'uploaded_at': datetime.now().isoformat()
            }
            
            uploaded_pdfs.append(pdf_data)
            uploaded_count += 1
        
        # Save to session
        request.session['uploaded_pdfs'] = uploaded_pdfs
        
        if uploaded_count > 0:
            if uploaded_count == 1:
                messages.success(request, f'PDF "{uploaded_pdfs[-1]["name"]}" uploaded successfully! Choose an operation below.')
            else:
                messages.success(request, f'{uploaded_count} PDFs uploaded successfully! Choose an operation below.')
            return redirect('dashboard')
        else:
            messages.error(request, 'No valid PDF files were uploaded.')
    
    return render(request, 'pdfeditor/upload.html')


def edit_view(request):
    """View for find & replace form."""
    # Get uploaded PDFs
    uploaded_pdfs = get_uploaded_pdfs(request)
    
    if not uploaded_pdfs:
        messages.error(request, 'No PDF found. Please upload a PDF first.')
        return redirect('dashboard')
    
    # Get PDF ID from query params or use first/only PDF
    pdf_id = request.GET.get('pdf')
    if pdf_id:
        selected_pdf = get_pdf_by_id(request, pdf_id)
        if not selected_pdf:
            messages.error(request, 'Selected PDF not found.')
            return redirect('dashboard')
    else:
        selected_pdf = uploaded_pdfs[0]  # Use first PDF if not specified
    
    pdf_path = selected_pdf['path']
    pdf_name = selected_pdf['name']
    
    if request.method == 'POST':
        form = FindReplaceForm(request.POST)
        if form.is_valid():
            search_text = form.cleaned_data['search_text']
            replace_text = form.cleaned_data['replace_text']
            case_sensitive = form.cleaned_data['case_sensitive']
            page_range = form.cleaned_data.get('page_range', '').strip()
            
            try:
                # Process PDF
                output_path, replacement_count, warnings = find_and_replace_text(
                    pdf_path=pdf_path,
                    search_text=search_text,
                    replace_text=replace_text,
                    case_sensitive=case_sensitive,
                    page_range=page_range if page_range else None
                )
                
                # Store result in session
                request.session['processed_pdf_path'] = output_path
                request.session['replacement_count'] = replacement_count
                request.session['warnings'] = warnings
                
                return redirect('result')
                
            except ValueError as e:
                messages.error(request, f'Error: {str(e)}')
            except Exception as e:
                messages.error(request, f'Error processing PDF: {str(e)}')
    else:
        form = FindReplaceForm()
    
    context = {
        'form': form,
        'pdf_name': pdf_name,
        'pdf_path_relative': os.path.relpath(pdf_path, settings.MEDIA_ROOT),
        'uploaded_pdfs': uploaded_pdfs,
        'selected_pdf': selected_pdf
    }
    return render(request, 'pdfeditor/edit.html', context)


def result_view(request):
    """View pentru afișarea rezultatului și link de download."""
    processed_pdf_path = request.session.get('processed_pdf_path')
    replacement_count = request.session.get('replacement_count', 0)
    warnings = request.session.get('warnings', [])
    
    if not processed_pdf_path or not os.path.exists(processed_pdf_path):
        messages.error(request, 'Processed file not found.')
        return redirect('dashboard')
    
    context = {
        'replacement_count': replacement_count,
        'warnings': warnings,
        'has_warnings': len(warnings) > 0,
        'pdf_path_relative': os.path.relpath(processed_pdf_path, settings.MEDIA_ROOT)
    }
    return render(request, 'pdfeditor/result.html', context)


def download_view(request):
    """View pentru descărcarea PDF-ului modificat."""
    processed_pdf_path = request.session.get('processed_pdf_path')
    
    if not processed_pdf_path or not os.path.exists(processed_pdf_path):
        raise Http404('PDF-ul nu a fost găsit.')
    
    # Serve file
    response = FileResponse(
        open(processed_pdf_path, 'rb'),
        content_type='application/pdf'
    )
    response['Content-Disposition'] = f'attachment; filename="{os.path.basename(processed_pdf_path)}"'
    
    return response


def preview_view(request):
    """View pentru preview PDF cu PDF.js."""
    # Can preview either uploaded or processed PDF
    pdf_type = request.GET.get('type', 'uploaded')  # 'uploaded' or 'processed'
    
    if pdf_type == 'processed':
        pdf_path = request.session.get('processed_pdf_path')
        pdf_name = 'Modified PDF'
    else:
        pdf_path = request.session.get('uploaded_pdf_path')
        pdf_name = request.session.get('uploaded_pdf_name', 'Uploaded PDF')
    
    if not pdf_path or not os.path.exists(pdf_path):
        messages.error(request, 'PDF not found for preview.')
        return redirect('dashboard')
    
    # Generate URL for PDF access
    # We'll serve it through a separate endpoint
    pdf_url = f"/media/{os.path.relpath(pdf_path, settings.MEDIA_ROOT)}"
    
    context = {
        'pdf_name': pdf_name,
        'pdf_url': pdf_url,
        'pdf_type': pdf_type
    }
    return render(request, 'pdfeditor/preview.html', context)


def split_view(request):
    """View for splitting PDF."""
    # Get uploaded PDFs
    uploaded_pdfs = get_uploaded_pdfs(request)
    
    if not uploaded_pdfs:
        messages.error(request, 'No PDF found. Please upload a PDF first.')
        return redirect('dashboard')
    
    # Get PDF ID from query params or use first/only PDF
    pdf_id = request.GET.get('pdf')
    if pdf_id:
        selected_pdf = get_pdf_by_id(request, pdf_id)
        if not selected_pdf:
            messages.error(request, 'Selected PDF not found.')
            return redirect('dashboard')
    else:
        selected_pdf = uploaded_pdfs[0]  # Use first PDF if not specified
    
    pdf_path = selected_pdf['path']
    pdf_name = selected_pdf['name']
    
    if request.method == 'POST':
        form = SplitPDFForm(request.POST)
        if form.is_valid():
            ranges = form.cleaned_data['ranges']
            
            try:
                # Split PDF
                output_files = split_pdf(pdf_path, ranges)
                
                # Store results in session
                request.session['split_files'] = output_files
                request.session['split_count'] = len(output_files)
                
                messages.success(request, f'PDF split successfully into {len(output_files)} files!')
                return redirect('split_result')
                
            except ValueError as e:
                messages.error(request, f'Error: {str(e)}')
            except Exception as e:
                messages.error(request, f'Error splitting PDF: {str(e)}')
    else:
        form = SplitPDFForm()
    
    context = {
        'form': form,
        'pdf_name': pdf_name,
        'pdf_path_relative': os.path.relpath(pdf_path, settings.MEDIA_ROOT),
        'uploaded_pdfs': uploaded_pdfs,
        'selected_pdf': selected_pdf
    }
    return render(request, 'pdfeditor/split.html', context)


def split_result_view(request):
    """View pentru rezultatele split PDF."""
    split_files = request.session.get('split_files', [])
    split_count = request.session.get('split_count', 0)
    
    if not split_files:
        messages.error(request, 'No split files found.')
        return redirect('dashboard')
    
    # Prepare file info for display
    files_info = []
    for file_path in split_files:
        if os.path.exists(file_path):
            files_info.append({
                'name': os.path.basename(file_path),
                'path': file_path,
                'path_relative': os.path.relpath(file_path, settings.MEDIA_ROOT),
                'size': os.path.getsize(file_path)
            })
    
    context = {
        'files_info': files_info,
        'split_count': split_count
    }
    return render(request, 'pdfeditor/split_result.html', context)


def download_split_file_view(request):
    """Download individual split file."""
    file_index = request.GET.get('file')
    split_files = request.session.get('split_files', [])
    
    if file_index is None or not split_files:
        raise Http404('File not found')
    
    try:
        file_index = int(file_index)
        if file_index < 0 or file_index >= len(split_files):
            raise Http404('File index out of range')
        
        file_path = split_files[file_index]
        
        if not os.path.exists(file_path):
            raise Http404('File not found on disk')
        
        response = FileResponse(
            open(file_path, 'rb'),
            content_type='application/pdf'
        )
        response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
        
        return response
    except (ValueError, IndexError):
        raise Http404('Invalid file index')


def merge_view(request):
    """View for merging multiple PDFs."""
    # Get uploaded PDFs
    uploaded_pdfs = get_uploaded_pdfs(request)
    
    if len(uploaded_pdfs) < 2:
        messages.error(request, 'You need at least 2 PDFs to merge. Please upload more files.')
        return redirect('dashboard')
    
    if request.method == 'POST':
        form = MergePDFForm(request.POST)
        if form.is_valid():
            selected_pdf_ids = form.cleaned_data['selected_pdfs']
            output_name = form.cleaned_data.get('output_name')
            
            try:
                # Get PDF paths in selected order
                pdf_paths = []
                for pdf_id in selected_pdf_ids:
                    pdf = get_pdf_by_id(request, pdf_id)
                    if pdf:
                        pdf_paths.append(pdf['path'])
                    else:
                        messages.error(request, f'PDF with ID {pdf_id} not found.')
                        return redirect('merge')
                
                if len(pdf_paths) < 2:
                    messages.error(request, 'At least 2 PDFs are required for merging.')
                    return redirect('merge')
                
                # Merge PDFs                
                merged_path = merge_pdfs(pdf_paths, output_name)
                
                # Store result in session
                request.session['merged_pdf_path'] = merged_path
                request.session['merged_pdf_count'] = len(pdf_paths)
                
                messages.success(request, f'Successfully merged {len(pdf_paths)} PDFs!')
                return redirect('merge_result')
                
            except ValueError as e:
                messages.error(request, f'Error: {str(e)}')
            except Exception as e:
                messages.error(request, f'Error merging PDFs: {str(e)}')
    else:
        form = MergePDFForm()
    
    context = {
        'form': form,
        'uploaded_pdfs': uploaded_pdfs
    }
    return render(request, 'pdfeditor/merge.html', context)


def merge_result_view(request):
    """View for displaying merge result."""
    merged_path = request.session.get('merged_pdf_path')
    merged_count = request.session.get('merged_pdf_count', 0)
    
    if not merged_path or not os.path.exists(merged_path):
        messages.error(request, 'Merged file not found.')
        return redirect('dashboard')
    
    context = {
        'merged_filename': os.path.basename(merged_path),
        'merged_size': os.path.getsize(merged_path),
        'merged_count': merged_count,
        'pdf_path_relative': os.path.relpath(merged_path, settings.MEDIA_ROOT)
    }
    return render(request, 'pdfeditor/merge_result.html', context)


def download_merged_view(request):
    """Download the merged PDF file."""
    merged_path = request.session.get('merged_pdf_path')
    
    if not merged_path or not os.path.exists(merged_path):
        messages.error(request, 'File not found.')
        return redirect('dashboard')
    
    try:
        return FileResponse(
            open(merged_path, 'rb'),
            as_attachment=True,
            filename=os.path.basename(merged_path)
        )
    except Exception as e:
        messages.error(request, f'Error downloading file: {str(e)}')
        return redirect('dashboard')


def compress_view(request):
    """View for compressing PDF."""
    # Get uploaded PDFs
    uploaded_pdfs = get_uploaded_pdfs(request)
    
    if not uploaded_pdfs:
        messages.error(request, 'No PDF found. Please upload a PDF first.')
        return redirect('dashboard')
    
    # Get PDF ID from query params or use first/only PDF
    pdf_id = request.GET.get('pdf')
    if pdf_id:
        selected_pdf = get_pdf_by_id(request, pdf_id)
        if not selected_pdf:
            messages.error(request, 'Selected PDF not found.')
            return redirect('dashboard')
    else:
        selected_pdf = uploaded_pdfs[0]
    
    pdf_path = selected_pdf['path']
    pdf_name = selected_pdf['name']
    
    if request.method == 'POST':
        form = CompressPDFForm(request.POST)
        if form.is_valid():
            quality = form.cleaned_data['quality']
            
            try:
                # Compress PDF
                output_path, original_size, compressed_size, compression_ratio = compress_pdf(
                    pdf_path, 
                    quality=quality
                )
                
                # Store results in session
                request.session['compressed_pdf_path'] = output_path
                request.session['original_size'] = original_size
                request.session['compressed_size'] = compressed_size
                request.session['compression_ratio'] = compression_ratio
                
                messages.success(request, f'PDF compressed successfully! Saved {compression_ratio:.1f}% space.')
                return redirect('compress_result')
                
            except Exception as e:
                messages.error(request, f'Error compressing PDF: {str(e)}')
    else:
        form = CompressPDFForm()
    
    context = {
        'form': form,
        'pdf_name': pdf_name,
        'pdf_path_relative': os.path.relpath(pdf_path, settings.MEDIA_ROOT),
        'uploaded_pdfs': uploaded_pdfs,
        'selected_pdf': selected_pdf,
        'original_size': os.path.getsize(pdf_path)
    }
    return render(request, 'pdfeditor/compress.html', context)


def compress_result_view(request):
    """View for displaying compression result."""
    compressed_path = request.session.get('compressed_pdf_path')
    original_size = request.session.get('original_size', 0)
    compressed_size = request.session.get('compressed_size', 0)
    compression_ratio = request.session.get('compression_ratio', 0)
    
    if not compressed_path or not os.path.exists(compressed_path):
        messages.error(request, 'Compressed file not found.')
        return redirect('dashboard')
    
    context = {
        'compressed_filename': os.path.basename(compressed_path),
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': compression_ratio,
        'saved_bytes': original_size - compressed_size,
        'pdf_path_relative': os.path.relpath(compressed_path, settings.MEDIA_ROOT)
    }
    return render(request, 'pdfeditor/compress_result.html', context)


def download_compressed_view(request):
    """Download the compressed PDF file."""
    compressed_path = request.session.get('compressed_pdf_path')
    
    if not compressed_path or not os.path.exists(compressed_path):
        messages.error(request, 'File not found.')
        return redirect('dashboard')
    
    try:
        return FileResponse(
            open(compressed_path, 'rb'),
            as_attachment=True,
            filename=os.path.basename(compressed_path)
        )
    except Exception as e:
        messages.error(request, f'Error downloading file: {str(e)}')
        return redirect('dashboard')


def watermark_view(request):
    """View for adding watermark to PDF."""
    # Get uploaded PDFs
    uploaded_pdfs = get_uploaded_pdfs(request)
    
    if not uploaded_pdfs:
        messages.error(request, 'No PDF found. Please upload a PDF first.')
        return redirect('dashboard')
    
    # Get PDF ID from query params or use first/only PDF
    pdf_id = request.GET.get('pdf')
    if pdf_id:
        selected_pdf = get_pdf_by_id(request, pdf_id)
        if not selected_pdf:
            messages.error(request, 'Selected PDF not found.')
            return redirect('dashboard')
    else:
        selected_pdf = uploaded_pdfs[0]
    
    pdf_path = selected_pdf['path']
    pdf_name = selected_pdf['name']
    
    if request.method == 'POST':
        form = WatermarkForm(request.POST, request.FILES)
        if form.is_valid():
            watermark_type = form.cleaned_data['watermark_type']
            position = form.cleaned_data['position']
            opacity = form.cleaned_data['opacity']
            rotation = form.cleaned_data['rotation']
            
            try:
                if watermark_type == 'text':
                    text_content = form.cleaned_data['text_content']
                    font_size = form.cleaned_data.get('font_size', 48)
                    
                    options = {
                        'position': position,
                        'opacity': opacity,
                        'rotation': rotation,
                        'font_size': font_size
                    }
                    
                    output_path = add_watermark(pdf_path, 'text', text_content, options)
                    
                elif watermark_type == 'image':
                    uploaded_image = form.cleaned_data['watermark_image']
                    
                    # Save uploaded image temporarily
                    fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'temp'))
                    image_filename = fs.save(uploaded_image.name, uploaded_image)
                    image_path = fs.path(image_filename)
                    
                    options = {
                        'position': position,
                        'opacity': opacity,
                        'rotation': rotation
                    }
                    
                    output_path = add_watermark(pdf_path, 'image', image_path, options)
                    
                    # Clean up temp image
                    if os.path.exists(image_path):
                        os.remove(image_path)
                
                # Store result in session
                request.session['watermarked_pdf_path'] = output_path
                
                messages.success(request, 'Watermark added successfully!')
                return redirect('watermark_result')
                
            except Exception as e:
                messages.error(request, f'Error adding watermark: {str(e)}')
    else:
        form = WatermarkForm()
    
    context = {
        'form': form,
        'pdf_name': pdf_name,
        'pdf_path_relative': os.path.relpath(pdf_path, settings.MEDIA_ROOT),
        'uploaded_pdfs': uploaded_pdfs,
        'selected_pdf': selected_pdf
    }
    return render(request, 'pdfeditor/watermark.html', context)


def watermark_result_view(request):
    """View for displaying watermark result."""
    watermarked_path = request.session.get('watermarked_pdf_path')
    
    if not watermarked_path or not os.path.exists(watermarked_path):
        messages.error(request, 'Watermarked file not found.')
        return redirect('dashboard')
    
    context = {
        'watermarked_filename': os.path.basename(watermarked_path),
        'watermarked_size': os.path.getsize(watermarked_path),
        'pdf_path_relative': os.path.relpath(watermarked_path, settings.MEDIA_ROOT)
    }
    return render(request, 'pdfeditor/watermark_result.html', context)


def download_watermarked_view(request):
    """Download the watermarked PDF file."""
    watermarked_path = request.session.get('watermarked_pdf_path')
    
    if not watermarked_path or not os.path.exists(watermarked_path):
        messages.error(request, 'File not found.')
        return redirect('dashboard')
    
    try:
        return FileResponse(
            open(watermarked_path, 'rb'),
            as_attachment=True,
            filename=os.path.basename(watermarked_path)
        )
    except Exception as e:
        messages.error(request, f'Error downloading file: {str(e)}')
        return redirect('dashboard')


def rotate_view(request):
    """View for rotating PDF pages."""
    uploaded_pdfs = get_uploaded_pdfs(request)
    
    if not uploaded_pdfs:
        messages.error(request, 'No PDF found. Please upload a PDF first.')
        return redirect('dashboard')
    
    pdf_id = request.GET.get('pdf')
    if pdf_id:
        selected_pdf = get_pdf_by_id(request, pdf_id)
        if not selected_pdf:
            messages.error(request, 'Selected PDF not found.')
            return redirect('dashboard')
    else:
        selected_pdf = uploaded_pdfs[0]
    
    pdf_path = selected_pdf['path']
    pdf_name = selected_pdf['name']
    
    if request.method == 'POST':
        form = RotatePagesForm(request.POST)
        if form.is_valid():
            rotation_angle = int(form.cleaned_data['rotation_angle'])
            page_range = form.cleaned_data.get('page_range', '').strip()
            
            try:
                output_path = rotate_pages(
                    pdf_path,
                    rotation_angle,
                    page_range if page_range else None
                )
                
                request.session['rotated_pdf_path'] = output_path
                request.session['rotation_angle'] = rotation_angle
                
                messages.success(request, f'Pages rotated {rotation_angle}° successfully!')
                return redirect('rotate_result')
                
            except ValueError as e:
                messages.error(request, f'Error: {str(e)}')
            except Exception as e:
                messages.error(request, f'Error rotating pages: {str(e)}')
    else:
        form = RotatePagesForm()
    
    context = {
        'form': form,
        'pdf_name': pdf_name,
        'pdf_path_relative': os.path.relpath(pdf_path, settings.MEDIA_ROOT),
        'uploaded_pdfs': uploaded_pdfs,
        'selected_pdf': selected_pdf
    }
    return render(request, 'pdfeditor/rotate.html', context)


def rotate_result_view(request):
    """View for displaying rotation result."""
    rotated_path = request.session.get('rotated_pdf_path')
    rotation_angle = request.session.get('rotation_angle', 0)
    
    if not rotated_path or not os.path.exists(rotated_path):
        messages.error(request, 'Rotated file not found.')
        return redirect('dashboard')
    
    context = {
        'rotated_filename': os.path.basename(rotated_path),
        'rotated_size': os.path.getsize(rotated_path),
        'rotation_angle': rotation_angle,
        'pdf_path_relative': os.path.relpath(rotated_path, settings.MEDIA_ROOT)
    }
    return render(request, 'pdfeditor/rotate_result.html', context)


def download_rotated_view(request):
    """Download the rotated PDF file."""
    rotated_path = request.session.get('rotated_pdf_path')
    
    if not rotated_path or not os.path.exists(rotated_path):
        messages.error(request, 'File not found.')
        return redirect('dashboard')
    
    try:
        return FileResponse(
            open(rotated_path, 'rb'),
            as_attachment=True,
            filename=os.path.basename(rotated_path)
        )
    except Exception as e:
        messages.error(request, f'Error downloading file: {str(e)}')
        return redirect('dashboard')


def page_numbers_view(request):
    """View for adding page numbers to PDF."""
    uploaded_pdfs = get_uploaded_pdfs(request)
    
    if not uploaded_pdfs:
        messages.error(request, 'No PDF found. Please upload a PDF first.')
        return redirect('dashboard')
    
    pdf_id = request.GET.get('pdf')
    if pdf_id:
        selected_pdf = get_pdf_by_id(request, pdf_id)
        if not selected_pdf:
            messages.error(request, 'Selected PDF not found.')
            return redirect('dashboard')
    else:
        selected_pdf = uploaded_pdfs[0]
    
    pdf_path = selected_pdf['path']
    pdf_name = selected_pdf['name']
    
    if request.method == 'POST':
        form = PageNumbersForm(request.POST)
        if form.is_valid():
            position = form.cleaned_data['position']
            format_type = form.cleaned_data['format']
            font_size = form.cleaned_data['font_size']
            start_page = form.cleaned_data['start_page']
            
            try:
                options = {
                    'position': position,
                    'format': format_type,
                    'font_size': font_size,
                    'start_page': start_page
                }
                
                output_path = add_page_numbers(pdf_path, options)
                
                request.session['numbered_pdf_path'] = output_path
                
                messages.success(request, 'Page numbers added successfully!')
                return redirect('page_numbers_result')
                
            except Exception as e:
                messages.error(request, f'Error adding page numbers: {str(e)}')
    else:
        form = PageNumbersForm()
    
    context = {
        'form': form,
        'pdf_name': pdf_name,
        'pdf_path_relative': os.path.relpath(pdf_path, settings.MEDIA_ROOT),
        'uploaded_pdfs': uploaded_pdfs,
        'selected_pdf': selected_pdf
    }
    return render(request, 'pdfeditor/page_numbers.html', context)


def page_numbers_result_view(request):
    """View for displaying page numbers result."""
    numbered_path = request.session.get('numbered_pdf_path')
    
    if not numbered_path or not os.path.exists(numbered_path):
        messages.error(request, 'Numbered file not found.')
        return redirect('dashboard')
    
    context = {
        'numbered_filename': os.path.basename(numbered_path),
        'numbered_size': os.path.getsize(numbered_path),
        'pdf_path_relative': os.path.relpath(numbered_path, settings.MEDIA_ROOT)
    }
    return render(request, 'pdfeditor/page_numbers_result.html', context)


def download_numbered_view(request):
    """Download the numbered PDF file."""
    numbered_path = request.session.get('numbered_pdf_path')
    
    if not numbered_path or not os.path.exists(numbered_path):
        messages.error(request, 'File not found.')
        return redirect('dashboard')
    
    try:
        return FileResponse(
            open(numbered_path, 'rb'),
            as_attachment=True,
            filename=os.path.basename(numbered_path)
        )
    except Exception as e:
        messages.error(request, f'Error downloading file: {str(e)}')
        return redirect('dashboard')


def more_tools_view(request):
    """View for More Tools modal with advanced functions."""
    uploaded_pdfs = get_uploaded_pdfs(request)
    
    if not uploaded_pdfs:
        messages.error(request, 'No PDF found. Please upload a PDF first.')
        return redirect('dashboard')
    
    context = {
        'uploaded_pdfs': uploaded_pdfs
    }
    return render(request, 'pdfeditor/more_tools.html', context)


def extract_text_ajax(request, pdf_id):
    """AJAX endpoint for extracting text from PDF."""
    from django.http import JsonResponse
    
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Invalid method'})
    
    pdf = get_pdf_by_id(request, pdf_id)
    if not pdf:
        return JsonResponse({'success': False, 'error': 'PDF not found'})
    
    try:
        text = extract_text_from_pdf(pdf['path'])
        
        # Store in session for download
        request.session['extracted_text'] = text
        request.session['extracted_filename'] = pdf['name'].replace('.pdf', '_extracted.txt')
        
        return JsonResponse({
            'success': True,
            'text': text,
            'filename': pdf['name']
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


def ocr_text_ajax(request, pdf_id):
    """AJAX endpoint for OCR text extraction from PDF."""
    from django.http import JsonResponse
    
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Invalid method'})
    
    pdf = get_pdf_by_id(request, pdf_id)
    if not pdf:
        return JsonResponse({'success': False, 'error': 'PDF not found'})
    
    try:
        text = ocr_pdf_to_text(pdf['path'])
        
        # Store in session for download
        request.session['extracted_text'] = text
        request.session['extracted_filename'] = pdf['name'].replace('.pdf', '_ocr.txt')
        
        return JsonResponse({
            'success': True,
            'text': text,
            'filename': pdf['name']
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


def download_text_view(request):
    """Download extracted text as .txt file."""
    from django.http import HttpResponse
    
    text = request.session.get('extracted_text')
    filename = request.session.get('extracted_filename', 'extracted.txt')
    
    if not text:
        messages.error(request, 'No text found to download.')
        return redirect('dashboard')
    
    response = HttpResponse(text, content_type='text/plain')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    
    return response


def delete_pdf_view(request, pdf_id):
    """Delete PDF from session."""
    uploaded_pdfs = request.session.get('uploaded_pdfs', [])
    
    # Find and remove PDF
    updated_pdfs = [pdf for pdf in uploaded_pdfs if pdf['id'] != pdf_id]
    
    if len(updated_pdfs) < len(uploaded_pdfs):
        request.session['uploaded_pdfs'] = updated_pdfs
        messages.success(request, 'PDF removed successfully.')
    else:
        messages.error(request, 'PDF not found.')
    
    return redirect('dashboard')


# ==========================================
# AI Rephrase Views
# ==========================================

def rephrase_view(request):
    """View for AI-powered text rephrasing in PDF."""
    from .forms import RephraseForm
    from .ollama_service import get_available_models, check_ollama_connection, rephrase_text
    
    # Get uploaded PDFs
    uploaded_pdfs = get_uploaded_pdfs(request)
    
    if not uploaded_pdfs:
        messages.error(request, 'No PDF found. Please upload a PDF first.')
        return redirect('dashboard')
    
    # Get PDF ID from query params or use first/only PDF
    pdf_id = request.GET.get('pdf')
    if pdf_id:
        selected_pdf = get_pdf_by_id(request, pdf_id)
        if not selected_pdf:
            messages.error(request, 'Selected PDF not found.')
            return redirect('dashboard')
    else:
        selected_pdf = uploaded_pdfs[0]
    
    pdf_path = selected_pdf['path']
    pdf_name = selected_pdf['name']
    
    # Check Ollama connection
    ollama_connected, ollama_message = check_ollama_connection()
    
    # Get available models
    available_models = get_available_models() if ollama_connected else []
    
    # Get available models from both providers
    from .ai_service import get_all_models, get_provider
    
    all_models = get_all_models()
    ollama_models = all_models.get('ollama', [])
    groq_models = all_models.get('groq', [])

    if request.method == 'POST':
        selected_text = request.POST.get('selected_text', '').strip()
        rephrase_style = request.POST.get('rephrase_style', 'formal')
        model = request.POST.get('model') or (ollama_models[0] if ollama_models else None)
        
        # Get coordinates from PDF.js selection
        page_number = int(request.POST.get('page_number', 0))
        bbox_x0 = float(request.POST.get('bbox_x0', 0))
        bbox_y0 = float(request.POST.get('bbox_y0', 0))
        bbox_x1 = float(request.POST.get('bbox_x1', 0))
        bbox_y1 = float(request.POST.get('bbox_y1', 0))
        
        # DEBUG: Print received coordinates
        print(f"🔍 Received coordinates from frontend:")
        print(f"   Page: {page_number}")
        print(f"   BBox: x0={bbox_x0:.2f}, y0={bbox_y0:.2f}, x1={bbox_x1:.2f}, y1={bbox_y1:.2f}")
        
        if not selected_text:
            messages.error(request, 'Please select text from the PDF first.')
        elif bbox_x0 == 0 and bbox_y0 == 0 and bbox_x1 == 0 and bbox_y1 == 0:
            # All coordinates are 0 - likely not set
            messages.error(request, 'Missing selection coordinates. Please capture the selection again.')
        elif not model:
            messages.error(request, 'Please select an AI model.')
        else:
            try:
                # Determine provider
                provider_name = 'groq' if model in groq_models else 'ollama'
                provider = get_provider(provider_name)
                
                # Rephrase
                rephrased_text, success, error_message = provider.rephrase(
                    text=selected_text,
                    style=rephrase_style,
                    model=model
                )
                
                if not success:
                    messages.error(request, f'AI Error ({provider_name}): {error_message}')
                else:
                    # Import coordinate-based rephrase function
                    from .pdf_processor import rephrase_with_coordinates
                    
                    # Build bounding box
                    bounding_box = {
                        'x0': float(bbox_x0),
                        'y0': float(bbox_y0),
                        'x1': float(bbox_x1),
                        'y1': float(bbox_y1)
                    }
                    
                    # Apply replacement using exact coordinates
                    output_path, replacement_count, warnings = rephrase_with_coordinates(
                        pdf_path=pdf_path,
                        page_number=int(page_number),
                        bounding_box_bl=bounding_box,
                        replace_text=rephrased_text,
                        original_text=selected_text
                    )
                    
                    # Store result in session
                    request.session['rephrased_pdf_path'] = output_path
                    request.session['rephrase_original_text'] = selected_text
                    request.session['rephrase_new_text'] = rephrased_text
                    request.session['rephrase_count'] = replacement_count
                    request.session['rephrase_warnings'] = warnings
                    request.session['rephrase_style'] = rephrase_style
                    request.session['rephrase_model'] = model
                    
                    return redirect('rephrase_result')
                    
            except ValueError as e:
                messages.error(request, f'Error: {str(e)}')
            except Exception as e:
                messages.error(request, f'Error processing PDF: {str(e)}')
    
    context = {
        'pdf_name': pdf_name,
        'pdf_path_relative': pdf_path.replace(settings.MEDIA_ROOT + '/', ''),
        'uploaded_pdfs': uploaded_pdfs,
        'selected_pdf': selected_pdf,
        'ollama_models': ollama_models,
        'groq_models': groq_models,
        'ollama_connected': bool(ollama_models), # Simple check
    }

    return render(request, 'pdfeditor/rephrase.html', context)


from django.views.decorators.csrf import csrf_exempt

@csrf_exempt  # Temporary for debugging CSRF issues
def rephrase_preview_ajax(request):
    """AJAX endpoint for previewing rephrased text without applying to PDF."""
    import json
    from .ollama_service import rephrase_text, check_ollama_connection, get_available_models
    
    if request.method != 'POST':
        return HttpResponse(
            json.dumps({'success': False, 'error': 'Only POST allowed'}),
            content_type='application/json'
        )
    
    try:
        # Get data from form submission (x-www-form-urlencoded)
        text = request.POST.get('text', '').strip()
        style = request.POST.get('style', 'formal')
        model = request.POST.get('model', '')
        
        if not text:
            return HttpResponse(
                json.dumps({'success': False, 'error': 'No text provided'}),
                content_type='application/json'
            )

        # Determine provider
        from .ai_service import get_all_models, get_provider
        all_models = get_all_models()
        groq_models = all_models.get('groq', [])
        
        provider_name = 'groq' if model in groq_models else 'ollama'
        provider = get_provider(provider_name)
        
        # Rephrase
        rephrased, success, error = provider.rephrase(text, style, model)
        
        if success:
            return HttpResponse(
                json.dumps({
                    'success': True,
                    'original_text': text,
                    'rephrased_text': rephrased,
                    'model': model,
                    'style': style
                }),
                content_type='application/json'
            )

        else:
            return HttpResponse(
                json.dumps({'success': False, 'error': error}),
                content_type='application/json'
            )
            
    except Exception as e:
        return HttpResponse(
            json.dumps({'success': False, 'error': str(e)}),
            content_type='application/json'
        )


def rephrase_result_view(request):
    """View for displaying rephrase result."""
    rephrased_path = request.session.get('rephrased_pdf_path')
    original_text = request.session.get('rephrase_original_text', '')
    new_text = request.session.get('rephrase_new_text', '')
    replacement_count = request.session.get('rephrase_count', 0)
    warnings = request.session.get('rephrase_warnings', [])
    style = request.session.get('rephrase_style', '')
    model = request.session.get('rephrase_model', '')
    
    if not rephrased_path or not os.path.exists(rephrased_path):
        messages.error(request, 'Rephrased PDF not found.')
        return redirect('dashboard')
    
    context = {
        'rephrased_filename': os.path.basename(rephrased_path),
        'rephrased_size': os.path.getsize(rephrased_path),
        'original_text': original_text,
        'new_text': new_text,
        'replacement_count': replacement_count,
        'warnings': warnings,
        'has_warnings': len(warnings) > 0,
        'style': style,
        'model': model,
        'pdf_path_relative': os.path.relpath(rephrased_path, settings.MEDIA_ROOT)
    }
    return render(request, 'pdfeditor/rephrase_result.html', context)


def download_rephrased_view(request):
    """Download the rephrased PDF file."""
    rephrased_path = request.session.get('rephrased_pdf_path')
    
    if not rephrased_path or not os.path.exists(rephrased_path):
        messages.error(request, 'File not found.')
        return redirect('dashboard')
    
    try:
        return FileResponse(
            open(rephrased_path, 'rb'),
            as_attachment=True,
            filename=os.path.basename(rephrased_path)
        )
    except Exception as e:
        messages.error(request, f'Error downloading file: {str(e)}')
        return redirect('dashboard')
