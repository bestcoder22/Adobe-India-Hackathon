import fitz

class PDFParseError(Exception):
    """Custom exception for PDF parsing failures."""
    pass

def extract_text_blocks(pdf_path, min_text_length=1):
    """
    Extracts structured text blocks from a PDF.

    Args:
      pdf_path (str): Path to the PDF file.
      min_text_length (int): Ignore blocks shorter than this.

    Returns:
      List[List[dict]]: Outer list per page; inner list of blocks.
                        Each block is PyMuPDF dict with keys:
                        'bbox', 'lines' (each line has 'spans').
    Raises:
      PDFParseError: if the PDF cannot be opened or has no extractable text.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise PDFParseError(f"Cannot open PDF: {e}")

    all_pages = []
    for page_num, page in enumerate(doc, start=1):
        try:
            data = page.get_text("dict")
            blocks = data.get("blocks", [])
        except Exception as e:
            raise PDFParseError(f"Failed to extract page {page_num}: {e}")

        # Filter out non‑text blocks or extremely small ones
        text_blocks = []
        for b in blocks:
            # Some blocks represent images or drawings: skip if no 'lines'
            if "lines" not in b or not b["lines"]:
                continue

            # Compute total text length in this block
            text = "".join(span["text"] for line in b["lines"] for span in line["spans"])
            if len(text.strip()) < min_text_length:
                continue

            # Attach page number for downstream reference
            b["_page_num"] = page_num
            rect = page.rect
            b["_page_width"]  = rect.width
            b["_page_height"] = rect.height
            text_blocks.append(b)

        if not text_blocks:
            # Optional: warn or treat empty pages as non‑fatal
            # e.g., print(f"Warning: no text on page {page_num}")
            pass

        all_pages.append(text_blocks)

    if not any(all_pages):
        raise PDFParseError("No text blocks found in entire document.")

    return all_pages
