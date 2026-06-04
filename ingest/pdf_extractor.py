from pypdf import PdfReader
import os
import re


def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Extract text from a PDF with metadata.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        dict with keys:
            - filename (str): Base filename of the PDF.
            - full_text (str): Concatenated text of all pages with page markers.
            - pages (list[dict]): Per-page text with page numbers.
            - total_pages (int): Number of pages in the PDF.
    """
    reader = PdfReader(pdf_path)
    filename = os.path.basename(pdf_path)
    full_text = ""
    page_texts = []

    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        raw = re.sub(r'-\n', '', raw)
        raw = re.sub(r'\n+', '\n', raw)
        raw = re.sub(r' +', ' ', raw)
        page_texts.append({"page": i + 1, "text": raw})
        full_text += f"\n[Page {i+1}]\n{raw}"

    return {
        "filename": filename,
        "full_text": full_text,
        "pages": page_texts,
        "total_pages": len(reader.pages)
    }


def extract_all_pdfs(docs_folder: str) -> list[dict]:
    """
    Extract text from all PDFs in a folder.

    Args:
        docs_folder: Path to the directory containing PDF files.

    Returns:
        list[dict]: A list of extraction results, one per PDF.
    """
    results = []
    pdf_files = [f for f in os.listdir(docs_folder) if f.endswith('.pdf')]

    if not pdf_files:
        print(f"No PDFs found in {docs_folder}")
        return results

    for pdf_file in pdf_files:
        path = os.path.join(docs_folder, pdf_file)
        print(f"Extracting: {pdf_file}")
        try:
            result = extract_text_from_pdf(path)
            results.append(result)
            print(f"  → {result['total_pages']} pages extracted")
        except Exception as e:
            print(f"  → ERROR: {e}")

    return results
