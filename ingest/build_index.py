"""
Run this script ONCE to build the FAISS knowledge base from your PDFs.
Usage: python ingest/build_index.py
Place your PDF textbooks in the sample_docs/ folder first.

Options:
  --all           Process ALL PDFs in sample_docs/ (default: core books only)
  --pdf <name>    Process a single PDF by filename (partial match OK)
  --rebuild       Delete existing index and rebuild from scratch
"""

import os
import sys
import time
import shutil
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Prevent transformers from importing TensorFlow (causes protobuf crash) ──
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

from ingest.pdf_extractor import extract_all_pdfs, extract_text_from_pdf
from ingest.chunker import chunk_all_documents, smart_chunk
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DOCS_FOLDER = "sample_docs"
INDEX_FOLDER = "knowledge_base"

# ✅ all-MiniLM-L6-v2: best speed/quality tradeoff for CPU
#    384-dim, ~90MB download, 5× faster than all-mpnet-base-v2
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Number of chunks to embed per batch. Lower = less RAM, more progress updates.
BATCH_SIZE = 512

# ── Core textbooks (balanced coverage, manageable size) ──────────────────────
# These 5 books cover the main ChemE domains without overwhelming the system.
# You can add more later with: python ingest/build_index.py --pdf "Perry"
CORE_PDFS = [
    "Coulson & Richardson — Chemical Engineering series.pdf",
    "Fogler — Elements of Chemical Reaction Engineering.pdf",
    "Kunii & Levenspiel — Fluidization Engineering.pdf",
    "Heat Transfer by Cengel 2nd Ed - PDF Room.pdf",
    "B.K Dutta.pdf",
]


def get_pdf_list(use_all: bool = False, single_pdf: str = None) -> list[str]:
    """
    Determine which PDFs to process.

    Args:
        use_all: If True, process all PDFs in the docs folder.
        single_pdf: If set, find a PDF matching this substring.

    Returns:
        list[str]: Filenames of PDFs to process.
    """
    all_pdfs = [f for f in os.listdir(DOCS_FOLDER) if f.lower().endswith('.pdf')]

    if single_pdf:
        matches = [f for f in all_pdfs if single_pdf.lower() in f.lower()]
        if not matches:
            print(f"ERROR: No PDF matching '{single_pdf}' found in {DOCS_FOLDER}/")
            print(f"Available PDFs:")
            for f in all_pdfs:
                print(f"  - {f}")
            sys.exit(1)
        return matches

    if use_all:
        return all_pdfs

    # Default: use core PDFs only (that actually exist in the folder)
    available_core = [f for f in CORE_PDFS if f in all_pdfs]
    if not available_core:
        print("WARNING: None of the core PDFs found. Falling back to all PDFs.")
        return all_pdfs

    skipped = [f for f in CORE_PDFS if f not in all_pdfs]
    if skipped:
        print(f"NOTE: {len(skipped)} core PDFs not found in {DOCS_FOLDER}/:")
        for f in skipped:
            print(f"  - {f}")

    return available_core


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds / 3600:.1f}hr"


def build_index():
    """
    End-to-end pipeline to build a FAISS vector index from PDF documents.

    Supports:
        - Core-only mode (default): processes 5 key textbooks (~15K chunks)
        - All mode (--all): processes every PDF in sample_docs/
        - Single PDF mode (--pdf): processes one specific PDF
        - Incremental mode: merges new PDFs into existing index
        - Rebuild mode (--rebuild): deletes existing index first

    Returns:
        None. Prints progress and saves the index to disk.
    """
    # ── Parse arguments ──────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="ChemCopilot Knowledge Base Builder")
    parser.add_argument("--all", action="store_true",
                        help="Process ALL PDFs in sample_docs/ (default: core books only)")
    parser.add_argument("--pdf", type=str, default=None,
                        help="Process a single PDF by filename (partial match OK)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Delete existing index and rebuild from scratch")
    args = parser.parse_args()

    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback: simple counter if tqdm not installed
        class tqdm:  # type: ignore
            def __init__(self, iterable=None, **kwargs):
                self._it = iter(iterable) if iterable else None
                self.total = kwargs.get("total", 0)
                self.desc = kwargs.get("desc", "")
                self.n = 0
            def __iter__(self):
                for item in self._it:
                    self.n += 1
                    print(f"  {self.desc}: {self.n}/{self.total}", end="\r", flush=True)
                    yield item
                print()
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def update(self, n=1):
                self.n += n
                print(f"  {self.desc}: {self.n}/{self.total}", end="\r", flush=True)

    print("=" * 50)
    print("ChemCopilot Knowledge Base Builder")
    print("=" * 50)

    # ── Handle rebuild ───────────────────────────────────────────────────
    if args.rebuild:
        index_path = os.path.join(INDEX_FOLDER, "index.faiss")
        if os.path.exists(index_path):
            print("\n🗑️  Deleting existing index for rebuild...")
            os.remove(os.path.join(INDEX_FOLDER, "index.faiss"))
            os.remove(os.path.join(INDEX_FOLDER, "index.pkl"))
            print("   Deleted ✓")

    # ── Determine which PDFs to process ──────────────────────────────────
    pdf_list = get_pdf_list(use_all=args.all, single_pdf=args.pdf)
    mode = "ALL" if args.all else ("SINGLE" if args.pdf else "CORE (5 books)")
    print(f"\n📚 Mode: {mode}")
    print(f"   PDFs to process: {len(pdf_list)}")
    for f in pdf_list:
        size_mb = os.path.getsize(os.path.join(DOCS_FOLDER, f)) / (1024 * 1024)
        print(f"   • {f} ({size_mb:.0f} MB)")

    # ── Step 1: Extract PDFs ─────────────────────────────────────────────
    print("\n[1/4] Extracting text from PDFs...")
    t_start = time.time()

    docs = []
    for pdf_file in pdf_list:
        path = os.path.join(DOCS_FOLDER, pdf_file)
        print(f"  Extracting: {pdf_file}")
        try:
            result = extract_text_from_pdf(path)
            docs.append(result)
            print(f"    → {result['total_pages']} pages extracted")
        except Exception as e:
            print(f"    → ERROR: {e}")

    if not docs:
        print("ERROR: No documents could be extracted. Check your PDFs.")
        return

    t_extract = time.time() - t_start
    print(f"  ⏱ Extraction took {format_time(t_extract)}")

    # ── Step 2: Chunk documents ──────────────────────────────────────────
    print(f"\n[2/4] Chunking {len(docs)} documents (chunk_size=1200, overlap=200)...")
    t_start = time.time()
    chunks = chunk_all_documents(docs)
    total_chunks = len(chunks)
    t_chunk = time.time() - t_start
    print(f"  Total chunks created: {total_chunks}")
    print(f"  ⏱ Chunking took {format_time(t_chunk)}")

    # Estimate embedding time (~50 chunks/sec on CPU with MiniLM)
    est_seconds = total_chunks / 50
    print(f"\n  ⏱ Estimated embedding time: ~{format_time(est_seconds)}")

    # ── Step 3: Load embedding model ─────────────────────────────────────
    print(f"\n[3/4] Loading embedding model: {EMBEDDING_MODEL}")
    print("  (First run downloads ~90MB — subsequent runs are instant)")
    t_start = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": BATCH_SIZE}
    )
    t_model = time.time() - t_start
    print(f"  Model loaded ✓ ({format_time(t_model)})")

    # ── Step 4: Build FAISS index in batches ─────────────────────────────
    # Check if we can merge into existing index (incremental mode)
    existing_index = os.path.join(INDEX_FOLDER, "index.faiss")
    vectorstore = None

    if os.path.exists(existing_index) and not args.rebuild:
        print(f"\n  📂 Loading existing index to merge into...")
        try:
            vectorstore = FAISS.load_local(
                INDEX_FOLDER, embeddings,
                allow_dangerous_deserialization=True
            )
            existing_count = vectorstore.index.ntotal
            print(f"     Existing vectors: {existing_count}")
        except Exception as e:
            print(f"     Could not load existing index: {e}")
            print(f"     Building fresh index instead.")
            vectorstore = None

    print(f"\n[4/4] Embedding {total_chunks} chunks in batches of {BATCH_SIZE}...")

    texts     = [c["text"]     for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    num_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE
    t_start = time.time()

    with tqdm(total=total_chunks, desc="Embedding", unit="chunk") as pbar:
        for batch_idx in range(num_batches):
            start = batch_idx * BATCH_SIZE
            end   = min(start + BATCH_SIZE, total_chunks)

            batch_texts     = texts[start:end]
            batch_metadatas = metadatas[start:end]

            batch_vs = FAISS.from_texts(
                texts=batch_texts,
                embedding=embeddings,
                metadatas=batch_metadatas
            )

            if vectorstore is None:
                vectorstore = batch_vs
            else:
                vectorstore.merge_from(batch_vs)

            pbar.update(end - start)

            # Show elapsed / estimated time
            elapsed = time.time() - t_start
            chunks_done = end
            rate = chunks_done / elapsed if elapsed > 0 else 0
            remaining = (total_chunks - chunks_done) / rate if rate > 0 else 0

            # Save checkpoint after every 10 batches (~5120 chunks)
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                os.makedirs(INDEX_FOLDER, exist_ok=True)
                vectorstore.save_local(INDEX_FOLDER)
                print(f"\n   💾 Checkpoint saved ({chunks_done}/{total_chunks} chunks)"
                      f" — {format_time(elapsed)} elapsed, ~{format_time(remaining)} remaining")

    total_vectors = vectorstore.index.ntotal
    print(f"\n✅ Knowledge base built successfully!")
    print(f"   Total vectors indexed: {total_vectors}")
    print(f"   Index saved to: {INDEX_FOLDER}/")
    print(f"   Total time: {format_time(time.time() - t_start)}")
    print(f"\nYou can now restart: streamlit run app.py")
    print(f"\n💡 To add more PDFs later:")
    print(f"   python ingest/build_index.py --pdf \"Perry\"")
    print(f"   python ingest/build_index.py --all")


if __name__ == "__main__":
    build_index()
