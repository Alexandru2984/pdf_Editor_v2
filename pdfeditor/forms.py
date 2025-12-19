from django import forms


class FindReplaceForm(forms.Form):
    """Form pentru căutare și înlocuire text în PDF."""
    
    search_text = forms.CharField(
        max_length=500,
        required=True,
        label="Text de căutat",
        widget=forms.TextInput(attrs={
            'placeholder': 'ex: test',
            'class': 'form-input'
        })
    )
    
    replace_text = forms.CharField(
        max_length=500,
        required=True,
        label="Text nou",
        widget=forms.TextInput(attrs={
            'placeholder': 'ex: exemplu',
            'class': 'form-input'
        })
    )
    
    case_sensitive = forms.BooleanField(
        required=False,
        initial=True,
        label="Case sensitive",
        widget=forms.CheckboxInput(attrs={'class': 'form-checkbox'})
    )
    
    page_range = forms.CharField(
        max_length=100,
        required=False,
        label="Interval de pagini (opțional)",
        help_text="Ex: 1-3,5,7-10 sau lasă gol pentru toate paginile",
        widget=forms.TextInput(attrs={
            'placeholder': '1-3,5',
            'class': 'form-input'
        })
    )


class SplitPDFForm(forms.Form):
    """Form pentru split PDF."""
    
    ranges = forms.CharField(
        max_length=500,
        required=True,
        label="Intervale de pagini",
        help_text="Introdu intervale separate prin virgulă. Ex: 1-5,10-15 va crea 2 fișiere",
        widget=forms.TextInput(attrs={
            'placeholder': '1-5, 10-15, 20',
            'class': 'form-input'
        })
    )
    
    def clean_ranges(self):
        """Parse și validează ranges."""
        ranges_str = self.cleaned_data['ranges']
        ranges = []
        
        parts = ranges_str.replace(' ', '').split(',')
        for part in parts:
            if '-' in part:
                try:
                    start, end = part.split('-')
                    start, end = int(start), int(end)
                    if start < 1 or start > end:
                        raise forms.ValidationError(f"Range invalid: {part}")
                    ranges.append((start, end))
                except ValueError:
                    raise forms.ValidationError(f"Format invalid: {part}")
            else:
                try:
                    page = int(part)
                    if page < 1:
                        raise forms.ValidationError(f"Pagina trebuie să fie >= 1")
                    ranges.append((page, page))  # Single page
                except ValueError:
                    raise forms.ValidationError(f"Număr invalid: {part}")
        
        if not ranges:
            raise forms.ValidationError("Specifică cel puțin un interval")
        
        return ranges


class MergePDFForm(forms.Form):
    """Form for merging multiple PDFs."""
    selected_pdfs = forms.CharField(
        widget=forms.HiddenInput(),
        help_text='Comma-separated PDF IDs in merge order'
    )
    
    output_name = forms.CharField(
        required=False,
        max_length=200,
        label='Output Filename (optional)',
        help_text='Custom name for merged PDF (without .pdf extension)',
        widget=forms.TextInput(attrs={
            'class': 'form-input',
            'placeholder': 'e.g., My_Merged_Document'
        })
    )
    
    def clean_selected_pdfs(self):
        """Validate selected PDFs."""
        selected = self.cleaned_data['selected_pdfs'].strip()
        
        if not selected:
            raise forms.ValidationError('Please select at least 2 PDFs to merge.')
        
        # Split by comma and validate
        pdf_ids = [pid.strip() for pid in selected.split(',') if pid.strip()]
        
        if len(pdf_ids) < 2:
            raise forms.ValidationError('At least 2 PDFs are required for merging.')
        
        return pdf_ids
    
    def clean_output_name(self):
        """Validate and clean output name."""
        name = self.cleaned_data.get('output_name', '').strip()
        
        if name:
            # Remove .pdf extension if user added it
            if name.lower().endswith('.pdf'):
                name = name[:-4]
            
            # Sanitize filename (remove special characters)
            import re
            name = re.sub(r'[^\w\s-]', '', name)
            name = re.sub(r'[-\s]+', '_', name)
        
        return name if name else None


class CompressPDFForm(forms.Form):
    """Form for compressing PDF with quality selection."""
    QUALITY_CHOICES = [
        ('low', 'Maximum Compression - Smallest file size (may affect image quality)'),
        ('medium', 'Balanced - Good balance between size and quality (Recommended)'),
        ('high', 'Minimal Compression - Preserves quality with slight size reduction'),
    ]
    
    quality = forms.ChoiceField(
        choices=QUALITY_CHOICES,
        initial='medium',
        widget=forms.RadioSelect,
        label='Compression Level'
    )


class WatermarkForm(forms.Form):
    """Form for adding watermark to PDF."""
    WATERMARK_TYPE_CHOICES = [
        ('text', 'Text Watermark'),
        ('image', 'Image Watermark'),
    ]
    
    POSITION_CHOICES = [
        ('center', 'Center'),
        ('top-left', 'Top Left'),
        ('top-center', 'Top Center'),
        ('top-right', 'Top Right'),
        ('center-left', 'Center Left'),
        ('center-right', 'Center Right'),
        ('bottom-left', 'Bottom Left'),
        ('bottom-center', 'Bottom Center'),
        ('bottom-right', 'Bottom Right'),
    ]
    
    watermark_type = forms.ChoiceField(
        choices=WATERMARK_TYPE_CHOICES,
        initial='text',
        widget=forms.RadioSelect,
        label='Watermark Type'
    )
    
    # Text watermark fields
    text_content = forms.CharField(
        required=False,
        max_length=200,
        label='Watermark Text',
        widget=forms.TextInput(attrs={
            'class': 'form-input',
            'placeholder': 'e.g., CONFIDENTIAL'
        })
    )
    
    font_size = forms.IntegerField(
        required=False,
        initial=48,
        min_value=12,
        max_value=200,
        label='Font Size',
        widget=forms.NumberInput(attrs={'class': 'form-input'})
    )
    
    # Image watermark field
    watermark_image = forms.ImageField(
        required=False,
        label='Watermark Image',
        help_text='Upload PNG/JPG (max 5MB)'
    )
    
    # Common options
    position = forms.ChoiceField(
        choices=POSITION_CHOICES,
        initial='center',
        label='Position'
    )
    
    opacity = forms.FloatField(
        initial=0.3,
        min_value=0.1,
        max_value=1.0,
        label='Opacity',
        widget=forms.NumberInput(attrs={
            'class': 'form-input',
            'step': '0.1',
            'type': 'range'
        })
    )
    
    rotation = forms.IntegerField(
        initial=45,
        min_value=-90,
        max_value=90,
        label='Rotation (degrees)',
        widget=forms.NumberInput(attrs={
            'class': 'form-input',
            'type': 'range'
        })
    )
    
    def clean(self):
        cleaned_data = super().clean()
        watermark_type = cleaned_data.get('watermark_type')
        
        if watermark_type == 'text':
            if not cleaned_data.get('text_content'):
                raise forms.ValidationError('Text content is required for text watermark.')
        elif watermark_type == 'image':
            if not cleaned_data.get('watermark_image'):
                raise forms.ValidationError('Image file is required for image watermark.')
        
        return cleaned_data


class RotatePagesForm(forms.Form):
    """Form for rotating PDF pages."""
    ROTATION_CHOICES = [
        (90, '90° Clockwise'),
        (180, '180° (Upside Down)'),
        (270, '270° Clockwise (90° Counter-Clockwise)'),
    ]
    
    rotation_angle = forms.ChoiceField(
        choices=ROTATION_CHOICES,
        initial=90,
        widget=forms.RadioSelect,
        label='Rotation Angle'
    )
    
    page_range = forms.CharField(
        required=False,
        max_length=200,
        label='Page Range (Optional)',
        help_text='Leave empty to rotate all pages, or specify like: 1-3,5,7-9',
        widget=forms.TextInput(attrs={
            'class': 'form-input',
            'placeholder': 'e.g., 1-3,5,7-9 or leave empty for all'
        })
    )


class PageNumbersForm(forms.Form):
    """Form for adding page numbers to PDF."""
    POSITION_CHOICES = [
        ('bottom-center', 'Bottom Center'),
        ('bottom-left', 'Bottom Left'),
        ('bottom-right', 'Bottom Right'),
        ('top-center', 'Top Center'),
        ('top-left', 'Top Left'),
        ('top-right', 'Top Right'),
    ]
    
    FORMAT_CHOICES = [
        ('number', 'Simple Number (1, 2, 3...)'),
        ('page_number', 'Page Number (Page 1, Page 2...)'),
        ('of_total', 'Of Total (1 of 10, 2 of 10...)'),
    ]
    
    position = forms.ChoiceField(
        choices=POSITION_CHOICES,
        initial='bottom-center',
        widget=forms.RadioSelect,
        label='Position'
    )
    
    format = forms.ChoiceField(
        choices=FORMAT_CHOICES,
        initial='number',
        widget=forms.RadioSelect,
        label='Format'
    )
    
    font_size = forms.IntegerField(
        initial=12,
        min_value=8,
        max_value=24,
        label='Font Size',
        widget=forms.NumberInput(attrs={'class': 'form-input'})
    )
    
    start_page = forms.IntegerField(
        initial=1,
        min_value=1,
        label='Start from Page',
        help_text='First page to add numbers to (default: 1)',
        widget=forms.NumberInput(attrs={'class': 'form-input'})
    )


class RephraseForm(forms.Form):
    """Form for AI-powered text rephrasing in PDF."""
    
    STYLE_CHOICES = [
        ('formal', 'Formal/Professional'),
        ('casual', 'Casual/Conversational'),
        ('simplified', 'Simplified'),
        ('concise', 'Concise'),
        ('expanded', 'Expanded/Detailed'),
    ]
    
    search_text = forms.CharField(
        max_length=2000,
        required=True,
        label="Text to Rephrase",
        widget=forms.Textarea(attrs={
            'placeholder': 'Enter the text you want to find and rephrase...',
            'class': 'form-input',
            'rows': 4
        })
    )
    
    rephrase_style = forms.ChoiceField(
        choices=STYLE_CHOICES,
        initial='formal',
        label='Rephrase Style',
        widget=forms.Select(attrs={'class': 'form-input'})
    )
    
    model = forms.ChoiceField(
        choices=[],  # Dynamically populated
        required=False,
        label='AI Model',
        widget=forms.Select(attrs={'class': 'form-input'})
    )
    
    case_sensitive = forms.BooleanField(
        required=False,
        initial=False,
        label='Case Sensitive Search',
        widget=forms.CheckboxInput(attrs={'class': 'form-checkbox'})
    )
    
    page_range = forms.CharField(
        max_length=100,
        required=False,
        label='Page Range (Optional)',
        help_text='e.g., 1-3,5,7-10 or leave blank for all pages',
        widget=forms.TextInput(attrs={
            'placeholder': '1-3,5',
            'class': 'form-input'
        })
    )
    
    def __init__(self, *args, **kwargs):
        model_choices = kwargs.pop('model_choices', None)
        super().__init__(*args, **kwargs)
        
        if model_choices:
            self.fields['model'].choices = model_choices
        else:
            # Default fallback
            self.fields['model'].choices = [('', 'Default Model')]

