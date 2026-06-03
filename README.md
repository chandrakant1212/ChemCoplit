---
title: ChemCopilot
emoji: ⚗️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# ⚗️ ChemCopilot — AI Chemical Process Engineering Assistant

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NVIDIA NIM](https://img.shields.io/badge/NVIDIA_NIM-LLaMA_3.1_70B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF?style=for-the-badge&logo=meta&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?style=for-the-badge)

> **ChemCopilot** is a RAG-powered AI assistant for chemical process engineering. It retrieves 
> context from your textbooks (PDFs), augments queries with domain knowledge, and delivers 
> structured engineering solutions with LaTeX equations, unit tracking, and citation references.

![ChemCopilot Screenshot](https://via.placeholder.com/900x450?text=ChemCopilot+Screenshot)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📚 **RAG Pipeline** | Ingest ChemE textbooks as PDFs → chunk → embed → FAISS vector search |
| 🧮 **Built-in Calculators** | Fluidization (Umf, Ut), Heat Exchangers (LMTD, area), Reactor Design (CSTR/PFR) |
| 💬 **Multi-turn Chat** | Persistent conversation history with context-aware follow-ups |
| 📐 **LaTeX Rendering** | Automatic rendering of equations in responses via `st.markdown()` |
| 🔍 **Topic Filtering** | Filter RAG retrieval by fluidization, heat transfer, reactor design, etc. |
| 📊 **Plotly Visualizations** | Interactive velocity comparison charts for fluidization calculations |
| 🏷️ **Source Citations** | Every response shows which textbook excerpts were used |
| 🌙 **Dark Engineering UI** | Clean, professional dark theme with custom CSS |

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10+
- An [NVIDIA NIM API key](https://build.nvidia.com/) (free tier available)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/chemcopilot.git
cd chemcopilot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your NVIDIA NIM API key
```

### 4. Add your textbooks

Place your Chemical Engineering PDF textbooks in the `sample_docs/` folder. Recommended:
- Kunii & Levenspiel — *Fluidization Engineering*
- Fogler — *Elements of Chemical Reaction Engineering*
- Perry's Chemical Engineers' Handbook
- Coulson & Richardson — *Chemical Engineering* series

### 5. Build the knowledge base (one-time)

```bash
python -m ingest.build_index
```

This will:
1. Extract text from all PDFs in `sample_docs/`
2. Chunk text into ~800-character segments with topic tagging
3. Embed chunks using `all-mpnet-base-v2` (~420 MB download on first run)
4. Save the FAISS index to `knowledge_base/`

### 6. Run the application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 📝 Usage Examples

### Example 1: Fluidization Calculation
```
Calculate the minimum fluidization velocity for 300µm sand particles
(ρ_p = 2650 kg/m³) in air at 400°C (ρ_g = 0.524 kg/m³, µ = 3.3e-5 Pa·s)
```

### Example 2: Heat Exchanger Sizing
```
Size a counter-current shell-and-tube heat exchanger:
- Hot oil: 180°C → 120°C
- Cooling water: 30°C → 70°C  
- Q = 500 kW, U = 350 W/m²·K
```

### Example 3: Reactor Design
```
Calculate the CSTR volume required for 90% conversion of a second-order reaction.
k = 0.05 m³/(mol·s), F_A0 = 2 mol/s, C_A0 = 2 mol/m³
```

---

## 🏗️ Project Structure

```
chemcopilot/
├── app.py                          # Main Streamlit application
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── ingest/
│   ├── __init__.py
│   ├── pdf_extractor.py            # PDF text extraction
│   ├── chunker.py                  # Smart text chunking with topic detection
│   └── build_index.py              # FAISS index builder (run once)
├── rag/
│   ├── __init__.py
│   ├── retriever.py                # Vector search logic (MMR + topic filter)
│   └── pipeline.py                 # Full RAG + LLM pipeline (NVIDIA NIM)
├── calculations/
│   ├── __init__.py
│   ├── fluidization.py             # Umf, Ut, bubble velocity, bed regime
│   ├── heat_exchanger.py           # LMTD, NTU, area sizing
│   ├── reactor_design.py           # CSTR, PFR volume sizing
│   └── mass_balance.py             # General mass/energy balance helpers
├── knowledge_base/
│   └── .gitkeep                    # FAISS index saved here after ingest
└── sample_docs/
    └── .gitkeep                    # Place PDF textbooks here
```

---

## 🧠 Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Meta LLaMA 3.1 70B via NVIDIA NIM |
| **Embeddings** | `all-MiniLM-L6-v2` (sentence-transformers) |
| **Vector Store** | FAISS (CPU) |
| **Orchestration** | LangChain 0.2 |
| **Frontend** | Streamlit 1.35 |
| **Visualization** | Plotly 5.22 |
| **PDF Parsing** | pypdf 4.2 |

---

## 🚢 Deployment on Hugging Face Spaces

### Step 1: Create a Hugging Face Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Name your Space (e.g., `chemcopilot`)
3. Select **Docker** as the SDK
4. Choose **Blank** Docker template
5. Set visibility to **Public** (or Private)
6. Click **Create Space**

### Step 2: Set up Git LFS and push

```bash
# Install Git LFS (one-time)
git lfs install

# Initialize repo and add HF Spaces remote
cd chemcopilot
git init
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/chemcopilot

# Track large knowledge base files with LFS
git lfs track "knowledge_base/index.faiss"
git lfs track "knowledge_base/index.pkl"

# Add all files and push
git add .
git commit -m "Initial deployment to HF Spaces"
git push space main
```

> **Important:** Make sure your `knowledge_base/index.faiss` and `knowledge_base/index.pkl`
> files are built locally first (`python ingest/build_index.py`) before pushing.

### Step 3: Add your API key as a Secret

1. Go to your Space → **Settings** → **Variables and secrets**
2. Click **New secret**
3. Name: `NVIDIA_NIM_API_KEY`
4. Value: your NVIDIA NIM API key
5. Save — the Space will rebuild automatically

### Step 4: Verify

The Space will build the Docker image and deploy. This takes ~5-10 minutes.
Once live, your app will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/chemcopilot
```

---

## ⚠️ Important Notes

- The app works **without** a knowledge base — it will use general LLM knowledge and show a setup banner.
- Temperature is set to **0.2** for deterministic, precise engineering answers.
- Chat history is capped at the **last 6 messages** to stay within context limits.
- All calculation functions return **dicts** for easy display as `st.metric()` cards.
- The `.env` file is in `.gitignore` — **never commit your API keys**.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
