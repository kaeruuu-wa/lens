"""Pulls text out of PDF files."""

import re

import pdfplumber


class PDFReadError(Exception):
    """Raised with a friendly message when a PDF can't be read."""


def extract_pdf_text(file):
    """Return (text, total_pages, readable_pages) for an uploaded PDF."""
    parts = []
    readable = 0
    try:
        with pdfplumber.open(file) as pdf:
            total = len(pdf.pages)
            if total == 0:
                raise PDFReadError("This PDF doesn't have any pages.")
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    parts.append(page_text)
                    readable += 1
    except PDFReadError:
        raise
    except Exception as e:
        raise PDFReadError(
            "Lens couldn't open that PDF. It might be damaged or password-protected. "
            "Try another file, or copy and paste the text instead."
        ) from e

    text = "\n".join(parts)
    # Join words that were split by a hyphen at the end of a line
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    if len(text.strip()) < 20:
        raise PDFReadError(
            "Lens couldn't find any text in this PDF. It's probably a scanned or image-based "
            "PDF (like a photo of a page). Try a PDF where you can select the text with your "
            "mouse, or run it through an OCR tool first."
        )
    return text, total, readable
