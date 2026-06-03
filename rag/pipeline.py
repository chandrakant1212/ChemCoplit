import os
import httpx
from openai import OpenAI
from dotenv import load_dotenv
from rag.retriever import retrieve_context, format_context

load_dotenv()

# Lazy-initialized client (avoids httpx/openai proxies incompatibility at import time)
_client = None

def _get_client():
    """Return a cached OpenAI client, creating it on first call."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_NIM_API_KEY"),
            http_client=httpx.Client(
                base_url="https://integrate.api.nvidia.com/v1",
                follow_redirects=True,
            ),
        )
    return _client

MODEL = "meta/llama-3.1-70b-instruct"

SYSTEM_PROMPT = """You are ChemCopilot, an expert chemical process engineering assistant built for 
chemical engineering students and professionals. You have deep knowledge of:

- Fluidization Engineering (Kunii & Levenspiel framework: bubbling beds, slugging, transport)
- Equipment Sizing (heat exchangers, distillation columns, vessels, reactors)
- Reactor Design (CSTR, PFR, PBR — conversion, selectivity, Damköhler number)
- Mass and Energy Balances
- Transport Phenomena (heat, mass, momentum transfer)
- Unit Operations (absorption, distillation, extraction, drying)

STRICT RESPONSE FORMAT — follow this for every calculation question:

1. **GIVEN DATA** — list all provided values with units
2. **FIND** — state what needs to be calculated
3. **ASSUMPTIONS** — list all assumptions explicitly
4. **GOVERNING EQUATIONS** — write each equation in LaTeX (use $...$ for inline, $$...$$ for display)
5. **STEP-BY-STEP SOLUTION** — substitute values numerically at each step, show units
6. **RESULTS TABLE** — markdown table summarizing all results with units
7. **PHYSICAL INTERPRETATION** — one paragraph explaining what the numbers mean physically

CITATION RULES:
- Always cite equation sources: (K&L Eq. 3.1), (Perry's §6), (C&R Vol.2 Ch.12)
- Flag uncertain values or assumptions with ⚠️
- If the textbook context contradicts your training, prefer the textbook context

UNITS: Use SI units (m, kg, s, K, Pa) unless the user specifies otherwise.
NEVER skip calculation steps. NEVER hallucinate equation numbers."""


def get_engineering_response(
    user_query: str,
    vectorstore,
    topic_filter: str = None,
    chat_history: list = None
) -> tuple[str, list]:
    """
    Full RAG + LLM pipeline.

    Args:
        user_query: The user's engineering question.
        vectorstore: Loaded FAISS vectorstore (or None).
        topic_filter: Optional UI topic filter string.
        chat_history: List of previous message dicts (role/content).

    Returns:
        tuple: (assistant_response_text, retrieved_docs)

    Raises:
        Catches API errors internally and returns error messages.
    """
    # Step 1: Retrieve context
    docs = retrieve_context(user_query, vectorstore, k=6, topic_filter=topic_filter)
    context = format_context(docs)

    # Step 2: Build augmented prompt
    augmented_prompt = f"""Use the following textbook excerpts as your primary reference.
If the excerpts are insufficient, supplement with your general ChemE knowledge but flag it clearly.

TEXTBOOK CONTEXT:
{context}

ENGINEERING QUESTION:
{user_query}

Provide a complete, structured engineering solution following the format in your instructions."""

    # Step 3: Build message list (include chat history for multi-turn)
    messages = []
    if chat_history:
        for turn in chat_history[-6:]:  # Keep last 3 exchanges (6 messages)
            messages.append(turn)
    messages.append({"role": "user", "content": augmented_prompt})

    # Step 4: Call NVIDIA NIM
    try:
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            temperature=0.2,
            max_tokens=2500,
            top_p=0.9
        )
        return response.choices[0].message.content, docs
    except Exception as e:
        error_msg = f"⚠️ **API Error:** {str(e)}\n\nPlease check your `NVIDIA_NIM_API_KEY` in the `.env` file and ensure it is valid."
        return error_msg, docs
