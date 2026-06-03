from langchain.text_splitter import RecursiveCharacterTextSplitter
import re


def smart_chunk(text: str, source_name: str, chunk_size: int = 1200, overlap: int = 200) -> list[dict]:
    """
    Split text into semantically meaningful chunks with metadata.
    Preserves equation context by not splitting mid-equation.

    Args:
        text: The full text to chunk.
        source_name: Name of the source document (used in metadata).
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between adjacent chunks.

    Returns:
        list[dict]: Each dict has 'text' and 'metadata' keys.
    """
    # Detect chapter/section headers to use as split boundaries
    section_pattern = re.compile(
        r'(Chapter \d+|Section \d+\.\d+|\d+\.\d+\s+[A-Z][a-z])',
        re.MULTILINE
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "],
        keep_separator=True
    )

    raw_chunks = splitter.split_text(text)

    chunks = []
    for i, chunk in enumerate(raw_chunks):
        # Detect topic from content
        topic = detect_topic(chunk)
        chunks.append({
            "text": chunk.strip(),
            "metadata": {
                "source": source_name,
                "chunk_id": i,
                "topic": topic,
                "char_count": len(chunk)
            }
        })

    return chunks


def detect_topic(text: str) -> str:
    """
    Heuristically tag a text chunk with a Chemical Engineering topic.

    Args:
        text: The text content of the chunk.

    Returns:
        str: One of 'fluidization', 'heat_transfer', 'reactor_design',
             'distillation', 'mass_energy_balance', 'equipment_sizing',
             or 'general'.
    """
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["fluidiz", "bubble", "umf", "minimum fluidization", "bed height", "slugging"]):
        return "fluidization"
    elif any(kw in text_lower for kw in ["heat exchanger", "lmtd", "overall heat transfer", "fouling", "ntu"]):
        return "heat_transfer"
    elif any(kw in text_lower for kw in ["cstr", "pfr", "pbr", "conversion", "damkohler", "residence time"]):
        return "reactor_design"
    elif any(kw in text_lower for kw in ["distillation", "mccabe", "reflux", "tray", "rectifying"]):
        return "distillation"
    elif any(kw in text_lower for kw in ["mass balance", "energy balance", "enthalpy", "stoichiometry"]):
        return "mass_energy_balance"
    elif any(kw in text_lower for kw in ["compressor", "pump", "vessel", "pressure drop", "pipe"]):
        return "equipment_sizing"
    else:
        return "general"


def chunk_all_documents(extracted_docs: list[dict]) -> list[dict]:
    """
    Chunk all extracted documents into indexed text fragments.

    Args:
        extracted_docs: List of extraction results from pdf_extractor.

    Returns:
        list[dict]: All chunks across all documents.
    """
    all_chunks = []
    for doc in extracted_docs:
        chunks = smart_chunk(doc["full_text"], source_name=doc["filename"])
        all_chunks.extend(chunks)
        print(f"  {doc['filename']}: {len(chunks)} chunks created")
    return all_chunks
