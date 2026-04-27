"""AcroForm field detection + filling.

PyMuPDF exposes form widgets through ``page.first_widget`` / ``widget.next``.
We support the field types most users actually see on real-world forms:
text (single + multi-line), checkbox, radio button, combobox, listbox.
Signature widgets are exposed but never filled — they need a cert.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Optional, Tuple

import fitz

from ._common import processed_dir, safe_basename, timestamp


# Map fitz widget type constants → human-readable + UI hint.
_FIELD_TYPE_NAMES: Dict[int, str] = {
    fitz.PDF_WIDGET_TYPE_TEXT: "text",
    fitz.PDF_WIDGET_TYPE_CHECKBOX: "checkbox",
    fitz.PDF_WIDGET_TYPE_RADIOBUTTON: "radio",
    fitz.PDF_WIDGET_TYPE_COMBOBOX: "combobox",
    fitz.PDF_WIDGET_TYPE_LISTBOX: "listbox",
    fitz.PDF_WIDGET_TYPE_SIGNATURE: "signature",
    fitz.PDF_WIDGET_TYPE_BUTTON: "button",
}


@dataclass
class FormField:
    name: str
    field_type: str
    page_number: int  # 0-indexed
    current_value: str
    options: List[str]   # For combobox/listbox/radio
    max_length: Optional[int]
    is_readonly: bool
    is_required: bool


def _widget_type_name(widget_type: int) -> str:
    return _FIELD_TYPE_NAMES.get(widget_type, "unknown")


def _iter_widgets(doc: "fitz.Document") -> Iterator[Tuple[int, "fitz.Widget"]]:
    for pno, page in enumerate(doc):
        widget = page.first_widget
        while widget is not None:
            yield pno, widget
            widget = widget.next


def extract_form_fields(pdf_path: str) -> List[FormField]:
    """Return every fillable field in `pdf_path`. Empty list if the PDF has no AcroForms."""
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    fields: List[FormField] = []
    with fitz.open(pdf_path) as doc:
        for pno, widget in _iter_widgets(doc):
            name = widget.field_name or f"field_{len(fields)}"
            options_raw = widget.choice_values or []
            options = [str(o) for o in options_raw] if options_raw else []
            fields.append(FormField(
                name=name,
                field_type=_widget_type_name(widget.field_type),
                page_number=pno,
                current_value=str(widget.field_value or ""),
                options=options,
                max_length=widget.text_maxlen if widget.text_maxlen else None,
                is_readonly=bool(widget.field_flags & 1),  # ReadOnly bit
                is_required=bool(widget.field_flags & 2),  # Required bit
            ))
    return fields


def has_form_fields(pdf_path: str) -> bool:
    """Cheap check: does this PDF contain at least one fillable widget?"""
    if not os.path.exists(pdf_path):
        return False
    with fitz.open(pdf_path) as doc:
        for _pno, _widget in _iter_widgets(doc):
            return True
    return False


def fill_form_fields(
    pdf_path: str,
    values: Mapping[str, str],
    flatten: bool = False,
) -> Tuple[str, int, List[str]]:
    """Fill widgets matching `values` (keyed by field name) and save a copy.

    Returns ``(output_path, num_filled, warnings)``.

    ``flatten=True`` bakes the values into the page content stream so
    downstream readers can't edit them.
    """
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    out_dir = processed_dir()
    out_path = os.path.join(
        out_dir, f"{safe_basename(pdf_path)}_filled_{timestamp()}.pdf",
    )
    warnings: List[str] = []
    num_filled = 0

    with fitz.open(pdf_path) as doc:
        for _pno, widget in _iter_widgets(doc):
            name = widget.field_name
            if not name or name not in values:
                continue
            if widget.field_flags & 1:  # ReadOnly
                warnings.append(f"Skipped read-only field '{name}'.")
                continue
            if widget.field_type == fitz.PDF_WIDGET_TYPE_SIGNATURE:
                warnings.append(f"Skipped signature field '{name}' (cannot fill without a cert).")
                continue

            new_value = values[name]

            try:
                if widget.field_type == fitz.PDF_WIDGET_TYPE_CHECKBOX:
                    # Accept "yes"/"true"/"on"/"1" as checked.
                    truthy = str(new_value).strip().lower() in ("yes", "true", "on", "1", "checked")
                    widget.field_value = True if truthy else False
                else:
                    widget.field_value = str(new_value)
                widget.update()
                num_filled += 1
            except Exception as exc:
                warnings.append(f"Could not fill field '{name}': {exc}")

        if flatten:
            # PyMuPDF flattens by saving without the form widgets dict.
            # Easiest: bake widgets into pages then strip the AcroForm tree.
            try:
                for page in doc:
                    widget = page.first_widget
                    while widget is not None:
                        rect = widget.rect
                        value = str(widget.field_value or "")
                        next_widget = widget.next
                        if value:
                            page.insert_textbox(
                                rect, value,
                                fontname="helv", fontsize=10, color=(0, 0, 0),
                                align=fitz.TEXT_ALIGN_LEFT,
                            )
                        widget = next_widget
                # Best-effort: remove the AcroForm catalog entry.
                if "/AcroForm" in doc.xref_get_keys(-1):
                    doc.xref_set_key(-1, "AcroForm", "null")
            except Exception as exc:
                warnings.append(f"Flattening failed: {exc}")

        doc.save(out_path, garbage=4, deflate=True, clean=True)

    return out_path, num_filled, warnings
