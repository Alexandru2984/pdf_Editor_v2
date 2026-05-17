"""Single-shot document summarization via Groq.

Distinct from chat: no embedding/index round trip, no per-question state.
Read the first N pages, hand the text to the LLM with a "summarize this"
prompt, return the answer. Output isn't persisted — clients re-call if
they want a fresh take.
"""

from __future__ import annotations

import logging

from .extract import extract_text_from_pdf

logger = logging.getLogger(__name__)

# Llama 3.3 70B has 128k context. We leave a generous chunk for the
# response + system prompt and cap input at ~80k chars (~20k tokens).
# Truncation is preferable to a 413-style refusal — partial summary
# beats no summary.
MAX_TEXT_CHARS = 80_000
MAX_PAGES = 50


def _build_prompt(text: str, language: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                f"You are a precise document summarizer. Reply in {language}. "
                "Produce 200-400 words covering: (1) main topic, (2) key "
                "points or arguments, (3) conclusions or recommendations. "
                "Stick to what the document actually says — no speculation, "
                "no filler, no boilerplate apologies. Output plain prose, "
                "no bullets or markdown unless the document itself is "
                "structured that way."
            ),
        },
        {
            "role": "user",
            "content": f"Summarize this document:\n\n{text}",
        },
    ]


def summarize_pdf(pdf_path: str, language: str = "English") -> dict:
    """Return ``{'summary': str, 'truncated': bool, 'chars_used': int}``.

    Raises ``ValueError`` if the PDF has no extractable text (scanned
    document — caller should run OCR first) or the upstream call fails.
    """
    from ..views.chat import _call_groq  # local import — chat owns the Groq client

    text = extract_text_from_pdf(pdf_path)
    if text.startswith("No text found"):
        raise ValueError("Document has no extractable text. Run OCR first.")

    truncated = len(text) > MAX_TEXT_CHARS
    if truncated:
        text = text[:MAX_TEXT_CHARS]

    answer, error = _call_groq(_build_prompt(text, language))
    if error:
        raise ValueError(f"Summary generation failed: {error}")

    return {
        "summary": answer,
        "truncated": truncated,
        "chars_used": len(text),
    }
