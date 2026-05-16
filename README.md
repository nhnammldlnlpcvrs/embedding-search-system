# Multimodal Search & RAG Assistant Engine

**Hybrid AI stack** — Julia computational backend × Python Streamlit frontend × Qdrant vector database × CLIP cross-modal embeddings × DeepSeek/Claude LLM generation.

<p align="center">
  <img src="https://img.shields.io/badge/Julia-1.10+-9558B2?logo=julia&logoColor=white" alt="Julia"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Qdrant-1.14+-DB2B39?logo=qdrant&logoColor=white" alt="Qdrant"/>
  <img src="https://img.shields.io/badge/CLIP-ViT--B/32-FF6F00" alt="CLIP"/>
  <img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License"/>
</p>

---

## Repository Architecture

```
embedding-search-system/
│
├── README.md                  # ← you are here
├── .gitignore
├── docker-compose.yml         # Qdrant local instance (REST :6333, gRPC :6334)
├── Project.toml               # Julia package manifest
├── requirements.txt           # Python dependencies
│
├── app.py                     # Streamlit web UI
├── main.jl                    # Julia CLI demo entry point
├── sample_kaggle.jl           # Flickr8k dataset ingestion pipeline
├── retrieval_system.jl        # Standalone semantic search (lightweight, no DB)
│
├── src/
│   ├── MiniVDB.jl             # Package entry — aggregates submodules + bridge API
│   ├── encoder.jl             # MultimodalEncoder (CLIP via PythonCall or mock)
│   ├── vector_store.jl        # DBClient (Qdrant REST API wrapper)
│   └── rag_engine.jl          # RAGAssistant (retrieve → prompt → LLM)
│
└── dataset/                   # git-ignored — place Flickr8k here
    └── Flickr8k/
        ├── Images/            # 8 091 × .jpg files
        │   ├── 1000268201_693b08cb0e.jpg
        │   ├── 1001773457_577c3a7d70.jpg
        │   └── ...
        └── captions.txt       # CSV: image,caption  (5 captions per image)
```

---

## Prerequisites

| Component | How to Install |
|---|---|
| **Docker** | [docker.com](https://docs.docker.com/get-docker/) — for Qdrant |
| **Julia 1.10+** | [julialang.org](https://julialang.org/downloads/) — core compute engine |
| **Python 3.10+** | [python.org](https://www.python.org/) — Streamlit frontend |
| **Flickr8k dataset** | [Kaggle — Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) — place under `dataset/Flickr8k/` |

### Install Julia dependencies

```bash
cd embedding-search-system
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

> This reads `Project.toml` and installs `CSV`, `DataFrames`, `HTTP`, `JSON3`, `PythonCall`, and stdlib packages.

### Install Python dependencies

```bash
pip install -r requirements.txt
```

> Installs `streamlit`, `juliacall`, and `Pillow`. Consider using a virtual environment.

### (Optional) Install CLIP for real multimodal embeddings

```bash
pip install torch clip Pillow
```

> Without this, the system falls back to deterministic mock embeddings — still fully functional for testing.

---

## How to Run

### Step 1 — Start Qdrant

```bash
docker compose up -d
```

> Qdrant is available at [http://localhost:6333](http://localhost:6333).  
> If Docker is unavailable, the system runs in **offline mode** with in-memory vector storage.

### Step 2 — Ingest the Flickr8k Dataset

```bash
# Ingest 20 images (plus their 5 captions each = 100 text vectors + 20 image vectors)
julia --project=. sample_kaggle.jl 20

# Or ingest the full dataset (8 091 images, ~48K vectors total)
julia --project=. sample_kaggle.jl 8091
```

> **CLI argument:** `sample_kaggle.jl [max_images]` — limits how many unique images to process.  
> Without an argument, defaults to **50 images**.  
> Each image produces **6 vectors**: 1 image embedding + 5 caption embeddings.

### Step 3 — Launch the Dashboard

```bash
streamlit run app.py
```

> Opens [http://localhost:8501](http://localhost:8501).  
> Use the sidebar to add more text/image documents, adjust Top-K, and execute multimodal RAG queries.

---

## Testing Scenarios

### Text → Image Retrieval

1. Ingest a few Flickr8k images via `sample_kaggle.jl 5`.
2. In the Streamlit UI, go to the **Search by Text** tab.
3. Enter: `"a child playing on stairs"`.
4. Click **Execute**. Observe that the top result is an image of a child climbing stairs — even though no text document contains that exact phrase.

### Image => Text Retrieval

1. Upload a query image via the **Search by Image** tab.

2. Click **Execute**. The system encodes the image via CLIP and retrieves the captions whose embeddings are closest in the shared vector space.

### RAG with Mock LLM

1. Leave the API key field empty in the sidebar.

2. Execute any query. The system returns a **structured mock answer** that references the retrieved contexts by index — proving the retrieval pipeline works end-to-end before you wire up a real LLM.

### RAG with Real LLM

1. Set `DEEPSEEK_API_KEY` in your environment or paste it into the sidebar.

2. Execute a query. The LLM receives the retrieved contexts and generates a **cited, grounded response**.

---

## Configuration Reference

| Variable | Purpose | Default |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API key for live RAG generation | (empty → mock mode) |
| `ANTHROPIC_API_KEY` | Claude API key (set provider to `:claude`) | (empty → mock mode) |
| `QDRANT_URL` | Qdrant REST endpoint | `http://localhost:6333` |
| `EMBEDDING_DIM` | Vector dimension (must match CLIP model) | `512` |
| `EMBED_BACKEND` | `:clip` or `:mock` | `:mock` |

---

## Module Overview

| Module | File | Responsibility |
|---|---|---|
| `MultimodalEncoder` | `src/encoder.jl` | `encode_text(text)`, `encode_image(path)` → 512-d `Vector{Float32}`. Real CLIP via PythonCall or deterministic mock hash. |
| `DBClient` | `src/vector_store.jl` | `QdrantClient`, `Point`, `SearchResult` structs + `create_collection!`, `upsert!`, `search` wrapping the Qdrant REST API. |
| `RAGAssistant` | `src/rag_engine.jl` | `RAGConfig`, `retrieve`, `build_prompt`, `generate`, `ask` — full retrieve-then-generate pipeline with DeepSeek & Claude support. |
| `MiniVDB` | `src/MiniVDB.jl` | Aggregates all submodules. Exposes bridge API (`insert_text`, `insert_image`, `query_rag`) for Streamlit/juliacall. |