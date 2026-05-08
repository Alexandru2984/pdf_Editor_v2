from django import forms
from django.utils.translation import gettext_lazy as _


class FindReplaceForm(forms.Form):
    """Form pentru căutare și înlocuire text în PDF."""

    search_text = forms.CharField(
        max_length=500,
        required=True,
        label="Text de căutat",
        widget=forms.TextInput(attrs={"placeholder": "ex: test", "class": "form-input"}),
    )

    replace_text = forms.CharField(
        max_length=500,
        required=True,
        label="Text nou",
        widget=forms.TextInput(attrs={"placeholder": "ex: exemplu", "class": "form-input"}),
    )

    case_sensitive = forms.BooleanField(
        required=False,
        initial=True,
        label="Case sensitive",
        widget=forms.CheckboxInput(attrs={"class": "form-checkbox"}),
    )

    page_range = forms.CharField(
        max_length=100,
        required=False,
        label="Interval de pagini (opțional)",
        help_text="Ex: 1-3,5,7-10 sau lasă gol pentru toate paginile",
        widget=forms.TextInput(attrs={"placeholder": "1-3,5", "class": "form-input"}),
    )


class SplitPDFForm(forms.Form):
    """Form pentru split PDF."""

    ranges = forms.CharField(
        max_length=500,
        required=True,
        label="Intervale de pagini",
        help_text="Introdu intervale separate prin virgulă. Ex: 1-5,10-15 va crea 2 fișiere",
        widget=forms.TextInput(attrs={"placeholder": "1-5, 10-15, 20", "class": "form-input"}),
    )

    def clean_ranges(self):
        """Parse și validează ranges."""
        ranges_str = self.cleaned_data["ranges"]
        ranges = []

        parts = ranges_str.replace(" ", "").split(",")
        for part in parts:
            if "-" in part:
                try:
                    start, end = part.split("-")
                    start, end = int(start), int(end)
                    if start < 1 or start > end:
                        raise forms.ValidationError(f"Range invalid: {part}")
                    ranges.append((start, end))
                except ValueError as exc:
                    raise forms.ValidationError(f"Format invalid: {part}") from exc
            else:
                try:
                    page = int(part)
                    if page < 1:
                        raise forms.ValidationError("Pagina trebuie să fie >= 1")
                    ranges.append((page, page))  # Single page
                except ValueError as exc:
                    raise forms.ValidationError(f"Număr invalid: {part}") from exc

        if not ranges:
            raise forms.ValidationError("Specifică cel puțin un interval")

        return ranges


class MergePDFForm(forms.Form):
    """Form for merging multiple PDFs."""

    selected_pdfs = forms.CharField(
        widget=forms.HiddenInput(), help_text="Comma-separated PDF IDs in merge order"
    )

    output_name = forms.CharField(
        required=False,
        max_length=200,
        label="Output Filename (optional)",
        help_text="Custom name for merged PDF (without .pdf extension)",
        widget=forms.TextInput(attrs={"class": "form-input", "placeholder": "e.g., My_Merged_Document"}),
    )

    def clean_selected_pdfs(self):
        """Validate selected PDFs."""
        selected = self.cleaned_data["selected_pdfs"].strip()

        if not selected:
            raise forms.ValidationError("Please select at least 2 PDFs to merge.")

        # Split by comma and validate
        pdf_ids = [pid.strip() for pid in selected.split(",") if pid.strip()]

        if len(pdf_ids) < 2:
            raise forms.ValidationError("At least 2 PDFs are required for merging.")

        return pdf_ids

    def clean_output_name(self):
        """Validate and clean output name."""
        name = self.cleaned_data.get("output_name", "").strip()

        if name:
            # Remove .pdf extension if user added it
            if name.lower().endswith(".pdf"):
                name = name[:-4]

            # Sanitize filename (remove special characters)
            import re

            name = re.sub(r"[^\w\s-]", "", name)
            name = re.sub(r"[-\s]+", "_", name)

        return name if name else None


class CompressPDFForm(forms.Form):
    """Form for compressing PDF with quality selection."""

    QUALITY_CHOICES = [
        ("low", "Maximum Compression - Smallest file size (may affect image quality)"),
        ("medium", "Balanced - Good balance between size and quality (Recommended)"),
        ("high", "Minimal Compression - Preserves quality with slight size reduction"),
    ]

    quality = forms.ChoiceField(
        choices=QUALITY_CHOICES, initial="medium", widget=forms.RadioSelect, label="Compression Level"
    )


class ProtectPDFForm(forms.Form):
    """Form for password-protecting a PDF (AES-256)."""

    user_password = forms.CharField(
        min_length=4,
        max_length=128,
        required=True,
        label="Password",
        widget=forms.PasswordInput(attrs={"autocomplete": "new-password"}),
    )
    user_password_confirm = forms.CharField(
        max_length=128,
        required=True,
        label="Confirm password",
        widget=forms.PasswordInput(attrs={"autocomplete": "new-password"}),
    )

    def clean(self) -> dict:
        cleaned = super().clean()
        pw = cleaned.get("user_password") or ""
        confirm = cleaned.get("user_password_confirm") or ""
        if pw and confirm and pw != confirm:
            self.add_error("user_password_confirm", "Passwords do not match.")
        return cleaned


class UnprotectPDFForm(forms.Form):
    """Form for removing the password from an encrypted PDF."""

    password = forms.CharField(
        max_length=128,
        required=True,
        label=_("Password"),
        widget=forms.PasswordInput(attrs={"autocomplete": "current-password"}),
    )


class SignPDFForm(forms.Form):
    """Form for signing a PDF with a user-supplied PKCS#12 certificate."""

    POSITION_CHOICES = [
        ("bottom-right", "Bottom right"),
        ("bottom-left", "Bottom left"),
        ("top-right", "Top right"),
        ("top-left", "Top left"),
        ("center", "Center"),
    ]

    p12_file = forms.FileField(
        required=True,
        label="Certificate (.p12 / .pfx)",
        help_text="Your PKCS#12 archive containing the private key + certificate.",
    )
    p12_password = forms.CharField(
        required=True,
        max_length=256,
        label="Certificate password",
        widget=forms.PasswordInput(attrs={"autocomplete": "off"}),
    )
    page = forms.IntegerField(
        required=True,
        min_value=1,
        initial=1,
        label="Page",
        help_text="Page number where the visible signature will appear.",
    )
    position = forms.ChoiceField(
        choices=POSITION_CHOICES,
        initial="bottom-right",
        label="Position",
    )
    reason = forms.CharField(
        required=False,
        max_length=256,
        label="Reason (optional)",
    )
    location = forms.CharField(
        required=False,
        max_length=256,
        label="Location (optional)",
    )
    add_timestamp = forms.BooleanField(
        required=False,
        initial=True,
        label=_("Embed trusted timestamp (RFC 3161)"),
        help_text=_("Adds a verifiable signing time fetched from a public TSA."),
    )
    embed_validation_info = forms.BooleanField(
        required=False,
        initial=False,
        label=_("Embed long-term validation info (PAdES B-LT)"),
        help_text=_(
            "Fetches OCSP/CRL responses for the signing chain and embeds them in "
            "the PDF, so the signature stays verifiable even if the CA goes offline. "
            "Only works with certificates issued by a public CA — self-signed "
            "certificates will fail."
        ),
    )
    add_doc_timestamp = forms.BooleanField(
        required=False,
        initial=False,
        label=_("Append archival timestamp (PAdES B-LTA)"),
        help_text=_(
            "Adds a document-level RFC 3161 timestamp after the signature so the "
            "PDF can be re-validated even after the signing certificate expires. "
            "Requires the trusted timestamp option above."
        ),
    )

    def clean(self) -> dict:
        cleaned = super().clean()
        if cleaned.get("add_doc_timestamp") and not cleaned.get("add_timestamp"):
            self.add_error(
                "add_doc_timestamp",
                "Archival timestamp requires the trusted timestamp option to be on.",
            )
        return cleaned

    def clean_p12_file(self):
        f = self.cleaned_data.get("p12_file")
        if f and f.size > 1024 * 1024:  # 1 MB cap is plenty for a .p12
            raise forms.ValidationError("Certificate file is too large (max 1 MB).")
        return f


class VerifyPDFForm(forms.Form):
    """Form for verifying signatures on an uploaded PDF."""

    pdf_file = forms.FileField(
        required=True,
        label="Signed PDF",
    )
    trust_certs = forms.FileField(
        required=False,
        label="Additional trust anchors (optional)",
        help_text=(
            "PEM or DER certificate(s) to treat as trusted. Use this for self-signed "
            "or test certificates that aren't issued by a public CA."
        ),
    )

    def clean_pdf_file(self):
        f = self.cleaned_data.get("pdf_file")
        if f and f.size > 50 * 1024 * 1024:
            raise forms.ValidationError("PDF too large (max 50 MB).")
        return f

    def clean_trust_certs(self):
        f = self.cleaned_data.get("trust_certs")
        if f and f.size > 256 * 1024:
            raise forms.ValidationError("Trust anchor file too large (max 256 KB).")
        return f


class GenerateCertForm(forms.Form):
    """Form for generating a self-signed PKCS#12 archive (test/demo use only)."""

    common_name = forms.CharField(
        max_length=128,
        required=True,
        label="Common name (CN)",
        help_text="The name printed on the certificate, e.g. your full name.",
    )
    passphrase = forms.CharField(
        min_length=4,
        max_length=128,
        required=True,
        label="Passphrase",
        widget=forms.PasswordInput(attrs={"autocomplete": "new-password"}),
        help_text="Encrypts the private key inside the .p12 archive.",
    )
    passphrase_confirm = forms.CharField(
        max_length=128,
        required=True,
        label="Confirm passphrase",
        widget=forms.PasswordInput(attrs={"autocomplete": "new-password"}),
    )

    def clean(self) -> dict:
        cleaned = super().clean()
        p = cleaned.get("passphrase") or ""
        c = cleaned.get("passphrase_confirm") or ""
        if p and c and p != c:
            self.add_error("passphrase_confirm", "Passphrases do not match.")
        return cleaned


class ConvertToDocxForm(forms.Form):
    """No-fields form for the PDF → DOCX conversion (CSRF-only)."""


class MetadataForm(forms.Form):
    """Edit the standard PDF metadata fields (Title, Author, Subject, ...)."""

    title = forms.CharField(
        required=False,
        max_length=500,
        widget=forms.TextInput(attrs={"placeholder": _("e.g. Quarterly Report")}),
    )
    author = forms.CharField(
        required=False,
        max_length=500,
        widget=forms.TextInput(attrs={"placeholder": _("e.g. Jane Doe")}),
    )
    subject = forms.CharField(
        required=False,
        max_length=500,
        widget=forms.TextInput(attrs={"placeholder": _("e.g. Sales analysis")}),
    )
    keywords = forms.CharField(
        required=False,
        max_length=1000,
        widget=forms.TextInput(attrs={"placeholder": _("comma-separated, e.g. report, sales, q4")}),
    )
    creator = forms.CharField(
        required=False,
        max_length=500,
        widget=forms.TextInput(attrs={"placeholder": _("e.g. Microsoft Word")}),
    )
    producer = forms.CharField(
        required=False,
        max_length=500,
        widget=forms.TextInput(attrs={"placeholder": _("e.g. Adobe PDF Library")}),
    )
    clear_dates = forms.BooleanField(
        required=False,
        label=_("Clear creation and modification dates"),
        help_text=_("Remove the embedded creation/modification timestamps."),
    )


class ImagesToPdfForm(forms.Form):
    """Form for assembling a PDF from one or more uploaded images.

    The actual files come from ``request.FILES.getlist("images")`` because
    Django's ``FileField`` does not natively bind multiple uploads. This form
    holds only the layout options.
    """

    PAGE_SIZE_CHOICES = [
        ("auto", _("Auto (page matches image size)")),
        ("a4", _("A4 (210 × 297 mm)")),
        ("letter", _("Letter (8.5 × 11 in)")),
    ]

    FIT_CHOICES = [
        ("fit", _("Fit (preserve aspect ratio, white letterbox)")),
        ("fill", _("Fill (stretch to fill page, may distort)")),
    ]

    page_size = forms.ChoiceField(
        choices=PAGE_SIZE_CHOICES,
        initial="auto",
        widget=forms.RadioSelect,
        label=_("Page size"),
    )
    fit_mode = forms.ChoiceField(
        choices=FIT_CHOICES,
        initial="fit",
        widget=forms.RadioSelect,
        label=_("Image fit"),
    )
    images_order = forms.CharField(
        widget=forms.HiddenInput(),
        required=False,
        help_text="Comma-separated list of original upload indices in display order.",
    )

    def clean_images_order(self):
        raw = (self.cleaned_data.get("images_order") or "").strip()
        if not raw:
            return []
        try:
            order = [int(x.strip()) for x in raw.split(",") if x.strip() != ""]
        except ValueError as exc:
            raise forms.ValidationError(_("Invalid image order.")) from exc
        if any(n < 0 for n in order):
            raise forms.ValidationError(_("Invalid image order."))
        if len(set(order)) != len(order):
            raise forms.ValidationError(_("Image order cannot contain duplicates."))
        return order


class PdfToImagesForm(forms.Form):
    """Form for the PDF → images export (one image per page, packaged as ZIP)."""

    FORMAT_CHOICES = [
        ("png", _("PNG (lossless, larger files)")),
        ("jpg", _("JPG (smaller files, good for photos/scans)")),
    ]

    DPI_CHOICES = [
        (72, _("Low (72 DPI — screen)")),
        (150, _("Medium (150 DPI — recommended)")),
        (300, _("High (300 DPI — print quality)")),
    ]

    fmt = forms.ChoiceField(
        choices=FORMAT_CHOICES,
        initial="png",
        widget=forms.RadioSelect,
        label=_("Image format"),
    )
    dpi = forms.TypedChoiceField(
        choices=DPI_CHOICES,
        coerce=int,
        initial=150,
        widget=forms.RadioSelect,
        label=_("Resolution"),
    )


class ReorderPagesForm(forms.Form):
    """Form for reordering / deleting PDF pages.

    ``page_order`` is a comma-separated list of 1-indexed page numbers in
    the desired final order. Pages omitted from the list are deleted.
    """

    page_order = forms.CharField(widget=forms.HiddenInput(), required=True)

    def clean_page_order(self):
        raw = self.cleaned_data["page_order"].strip()
        if not raw:
            raise forms.ValidationError(_("You must keep at least one page."))
        try:
            order = [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError as err:
            raise forms.ValidationError(_("Invalid page order format.")) from err
        if not order:
            raise forms.ValidationError(_("You must keep at least one page."))
        if any(n < 1 for n in order):
            raise forms.ValidationError(_("Page numbers must be 1 or greater."))
        if len(set(order)) != len(order):
            raise forms.ValidationError(_("Page order cannot contain duplicates."))
        return order


class WatermarkForm(forms.Form):
    """Form for adding watermark to PDF."""

    WATERMARK_TYPE_CHOICES = [
        ("text", "Text Watermark"),
        ("image", "Image Watermark"),
    ]

    POSITION_CHOICES = [
        ("center", "Center"),
        ("top-left", "Top Left"),
        ("top-center", "Top Center"),
        ("top-right", "Top Right"),
        ("center-left", "Center Left"),
        ("center-right", "Center Right"),
        ("bottom-left", "Bottom Left"),
        ("bottom-center", "Bottom Center"),
        ("bottom-right", "Bottom Right"),
    ]

    watermark_type = forms.ChoiceField(
        choices=WATERMARK_TYPE_CHOICES, initial="text", widget=forms.RadioSelect, label="Watermark Type"
    )

    # Text watermark fields
    text_content = forms.CharField(
        required=False,
        max_length=200,
        label="Watermark Text",
        widget=forms.TextInput(attrs={"class": "form-input", "placeholder": "e.g., CONFIDENTIAL"}),
    )

    font_size = forms.IntegerField(
        required=False,
        initial=48,
        min_value=12,
        max_value=200,
        label="Font Size",
        widget=forms.NumberInput(attrs={"class": "form-input"}),
    )

    # Image watermark field
    watermark_image = forms.ImageField(
        required=False, label="Watermark Image", help_text="Upload PNG/JPG (max 5MB)"
    )

    # Common options
    position = forms.ChoiceField(choices=POSITION_CHOICES, initial="center", label="Position")

    opacity = forms.FloatField(
        initial=0.3,
        min_value=0.1,
        max_value=1.0,
        label="Opacity",
        widget=forms.NumberInput(attrs={"class": "form-input", "step": "0.1", "type": "range"}),
    )

    rotation = forms.IntegerField(
        initial=45,
        min_value=-90,
        max_value=90,
        label="Rotation (degrees)",
        widget=forms.NumberInput(attrs={"class": "form-input", "type": "range"}),
    )

    def clean(self):
        cleaned_data = super().clean()
        watermark_type = cleaned_data.get("watermark_type")

        if watermark_type == "text" and not cleaned_data.get("text_content"):
            raise forms.ValidationError("Text content is required for text watermark.")
        if watermark_type == "image" and not cleaned_data.get("watermark_image"):
            raise forms.ValidationError("Image file is required for image watermark.")

        return cleaned_data


class RotatePagesForm(forms.Form):
    """Form for rotating PDF pages."""

    ROTATION_CHOICES = [
        (90, "90° Clockwise"),
        (180, "180° (Upside Down)"),
        (270, "270° Clockwise (90° Counter-Clockwise)"),
    ]

    rotation_angle = forms.ChoiceField(
        choices=ROTATION_CHOICES, initial=90, widget=forms.RadioSelect, label="Rotation Angle"
    )

    page_range = forms.CharField(
        required=False,
        max_length=200,
        label="Page Range (Optional)",
        help_text="Leave empty to rotate all pages, or specify like: 1-3,5,7-9",
        widget=forms.TextInput(
            attrs={"class": "form-input", "placeholder": "e.g., 1-3,5,7-9 or leave empty for all"}
        ),
    )


class PageNumbersForm(forms.Form):
    """Form for adding page numbers to PDF."""

    POSITION_CHOICES = [
        ("bottom-center", "Bottom Center"),
        ("bottom-left", "Bottom Left"),
        ("bottom-right", "Bottom Right"),
        ("top-center", "Top Center"),
        ("top-left", "Top Left"),
        ("top-right", "Top Right"),
    ]

    FORMAT_CHOICES = [
        ("number", "Simple Number (1, 2, 3...)"),
        ("page_number", "Page Number (Page 1, Page 2...)"),
        ("of_total", "Of Total (1 of 10, 2 of 10...)"),
    ]

    position = forms.ChoiceField(
        choices=POSITION_CHOICES, initial="bottom-center", widget=forms.RadioSelect, label="Position"
    )

    format = forms.ChoiceField(
        choices=FORMAT_CHOICES, initial="number", widget=forms.RadioSelect, label="Format"
    )

    font_size = forms.IntegerField(
        initial=12,
        min_value=8,
        max_value=24,
        label="Font Size",
        widget=forms.NumberInput(attrs={"class": "form-input"}),
    )

    start_page = forms.IntegerField(
        initial=1,
        min_value=1,
        label="Start from Page",
        help_text="First page to add numbers to (default: 1)",
        widget=forms.NumberInput(attrs={"class": "form-input"}),
    )


class RephraseForm(forms.Form):
    """Form for AI-powered text rephrasing in PDF."""

    STYLE_CHOICES = [
        ("formal", "Formal/Professional"),
        ("casual", "Casual/Conversational"),
        ("simplified", "Simplified"),
        ("concise", "Concise"),
        ("expanded", "Expanded/Detailed"),
    ]

    search_text = forms.CharField(
        max_length=2000,
        required=True,
        label="Text to Rephrase",
        widget=forms.Textarea(
            attrs={
                "placeholder": "Enter the text you want to find and rephrase...",
                "class": "form-input",
                "rows": 4,
            }
        ),
    )

    rephrase_style = forms.ChoiceField(
        choices=STYLE_CHOICES,
        initial="formal",
        label="Rephrase Style",
        widget=forms.Select(attrs={"class": "form-input"}),
    )

    model = forms.ChoiceField(
        choices=[],  # Dynamically populated
        required=False,
        label="AI Model",
        widget=forms.Select(attrs={"class": "form-input"}),
    )

    case_sensitive = forms.BooleanField(
        required=False,
        initial=False,
        label="Case Sensitive Search",
        widget=forms.CheckboxInput(attrs={"class": "form-checkbox"}),
    )

    page_range = forms.CharField(
        max_length=100,
        required=False,
        label="Page Range (Optional)",
        help_text="e.g., 1-3,5,7-10 or leave blank for all pages",
        widget=forms.TextInput(attrs={"placeholder": "1-3,5", "class": "form-input"}),
    )

    def __init__(self, *args, **kwargs):
        model_choices = kwargs.pop("model_choices", None)
        super().__init__(*args, **kwargs)

        if model_choices:
            self.fields["model"].choices = model_choices
        else:
            # Default fallback
            self.fields["model"].choices = [("", "Default Model")]
