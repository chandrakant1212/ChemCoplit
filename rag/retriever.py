import os

# ── Prevent transformers from importing TensorFlow (causes protobuf crash) ──
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st

INDEX_FOLDER = "knowledge_base"
# Must match the model used in build_index.py
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_vectorstore():
    """
    Load the FAISS index from disk. Cached so it only loads once per session.

    Returns:
        FAISS vectorstore instance, or None if the index does not exist.
    """
    if not os.path.exists(os.path.join(INDEX_FOLDER, "index.faiss")):
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.load_local(
        INDEX_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def retrieve_context(query: str, vectorstore, k: int = 6, topic_filter: str = None) -> list:
    """
    Retrieve top-k relevant chunks using MMR search.
    Optionally filter by topic tag.

    Args:
        query: The user's search query.
        vectorstore: A loaded FAISS vectorstore instance (or None).
        k: Number of top results to return.
        topic_filter: Optional UI-friendly topic name to filter by.

    Returns:
        list: List of LangChain Document objects with page_content and metadata.
    """
    if vectorstore is None:
        return []

    search_kwargs = {"k": k, "fetch_k": 20}

    if topic_filter and topic_filter != "All Topics":
        topic_map = {
            "Fluidization & Fluidized Beds": "fluidization",
            "Heat Exchangers": "heat_transfer",
            "Reactor Design (CSTR/PFR/PBR)": "reactor_design",
            "Distillation": "distillation",
            "Mass & Energy Balances": "mass_energy_balance",
            "Equipment Sizing": "equipment_sizing"
        }
        tag = topic_map.get(topic_filter)
        if tag:
            search_kwargs["filter"] = {"topic": tag}

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )

    docs = retriever.invoke(query)
    return docs


def format_context(docs: list) -> str:
    """
    Format retrieved docs into a clean context string for the LLM.

    Args:
        docs: List of LangChain Document objects.

    Returns:
        str: Formatted context string with source annotations.
    """
    if not docs:
        return "No relevant textbook excerpts found. Answer from general chemical engineering knowledge."

    sections = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Textbook")
        topic = doc.metadata.get("topic", "general")
        sections.append(
            f"[Excerpt {i} | Source: {source} | Topic: {topic}]\n{doc.page_content}"
        )

    return "\n\n" + ("─" * 40) + "\n\n".join(sections)
