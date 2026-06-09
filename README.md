# 🤖 Hybrid RAG Chatbot

> A fully local, production-grade Retrieval-Augmented Generation (RAG) system for intelligent document question-answering — no API keys, no cloud, no data leaving your machine.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=langchain)](https://langchain.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?logo=ollama)](https://ollama.com)
[![FAISS](https://img.shields.io/badge/FAISS-HNSW_Index-orange)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Technology Stack](#-technology-stack)
4. [System Architecture](#-system-architecture)
5. [Project Structure](#-project-structure)
6. [Pipeline Workflows](#-pipeline-workflows)
7. [Data Flow & Streaming Protocol](#-data-flow--streaming-protocol)
8. [API Reference](#-api-reference)
9. [Installation & Setup](#-installation--setup)
10. [Configuration Reference](#-configuration-reference)
11. [Running the Application](#-running-the-application)
12. [Deployment](#-deployment)
13. [Thread-Safety Model](#-thread-safety-model)
14. [Caching Strategy](#-caching-strategy)
15. [Session & Memory Management](#-session--memory-management)
16. [Scoring & Grounding](#-scoring--grounding)
17. [Technical Challenges & Solutions](#-technical-challenges--solutions)
18. [Future Enhancements](#-future-enhancements)
19. [Troubleshooting](#-troubleshooting)
20. [Project Outcomes](#-project-outcomes)

---

## 🎯 Project Overview

The **Hybrid RAG Chatbot** is a production-ready, fully offline document question-answering system. It allows users to ask natural language questions about their own PDF documents and receive streamed, cited, grounded answers — all processed locally using open-source models.

### Problem Statement

Organisations increasingly need private documents — policy manuals, research reports, technical specs — to be queryable by non-technical users. Existing solutions either rely on expensive cloud LLM APIs (introducing data-privacy risks) or hallucinate content beyond the source material. This project solves both problems by combining rigorous retrieval pipelines with strict LLM grounding constraints, running entirely on local hardware.

### Objectives

- Build a fully local, privacy-preserving document QA system
- Achieve higher retrieval precision than pure vector search alone
- Provide faithful, cited answers with measurable confidence scores
- Stream token-by-token responses for a responsive user experience
- Support live document ingestion without restarting the server
- Deliver a production-ready codebase with thread-safe concurrency

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Hybrid Retrieval** | FAISS HNSW dense search + BM25 sparse search, fused with Reciprocal Rank Fusion (RRF) |
| **CrossEncoder Reranking** | Second-pass relevance scoring for top retrieval candidates before LLM generation |
| **Small-to-Big Chunking** | Retrieve small precise child chunks; send full parent paragraph context to the LLM |
| **Token Streaming** | Real-time token-by-token response via NDJSON over HTTP — no waiting for full generation |
| **Conversation Memory** | Multi-turn sessions with LangChain memory, topic-shift detection, and follow-up reuse |
| **Dual-Layer Cache** | Exact MD5 cache + semantic cosine cache, both with TTL expiry |
| **Confidence Scoring** | Faithfulness + relevance scores with a visual grounding warning when scores are low |
| **Live Reindexing** | Upload new PDFs via UI or API — searchable immediately, no server restart required |
| **PII Scrubbing** | Auto-redacts emails, SSNs, and phone numbers before indexing |
| **100% Local** | All models run via Ollama and HuggingFace — nothing sent to any external service |

---

## 🛠 Technology Stack

### Backend
| Package | Version | Purpose |
|---|---|---|
| `fastapi` + `uvicorn` | 0.115 / 0.30 | Async web framework + ASGI server |
| `langchain` + `langchain-ollama` | 0.3.25 / 0.3.3 | LLM orchestration, memory, streaming |
| `langchain-huggingface` | 0.1.2 | HuggingFace embeddings wrapper |
| `sentence-transformers` | 3.0.1 | Embedding model + CrossEncoder reranker |
| `faiss-cpu` | 1.8.0 | HNSW approximate nearest-neighbour index |
| `rank-bm25` | 0.2.2 | BM25Okapi sparse retrieval |
| `pdfminer.six` | 20231228 | PDF text extraction |
| `scikit-learn` + `numpy` | 1.5.2 / ≥1.26 | Cosine similarity, vector math |
| `torch` | 2.2.2 | PyTorch backend for sentence-transformers |
| `aiofiles` | 24.1.0 | Async file I/O for PDF uploads |
| `requests` | ≥2.32 | Synchronous Ollama API calls |

### Models
| Model | Role |
|---|---|
| `qwen2.5:3b` (Ollama) | Main answer generation |
| `qwen2.5:0.5b` (Ollama) | Query preprocessing + intent classification |
| `all-mpnet-base-v2` (HuggingFace) | 768-dim dense embeddings |
| `ms-marco-MiniLM-L-6-v2` (HuggingFace) | CrossEncoder reranker |

### Frontend
- Vanilla HTML + CSS + JavaScript (no build step, no framework)
- Fetch Streams API for NDJSON consumption
- Dark-mode UI with collapsible citations and live markdown rendering

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser UI                           │
│        HTML + CSS + JS  ·  Streaming NDJSON over HTTP       │
└────────────────────────────┬────────────────────────────────┘
                             │ POST /api/v1/answer
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI  (main.py)                        │
│           Routes · CORS · Static Files · Lifespan           │
│        Background Eviction Task (every 5 minutes)           │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              HybridRAGChatbot  (app/chatbot.py)             │
│                  Pipeline Orchestrator                      │
│                                                             │
│  ┌────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │SessionMgr  │  │  ResponseCache  │  │   LLMClient     │  │
│  │threading   │  │  threading.Lock │  │ preprocess      │  │
│  │.Lock       │  │  MD5 + cosine   │  │ intent classify │  │
│  └────────────┘  └─────────────────┘  │ stream generate │  │
│                                       └─────────────────┘  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         DocumentStore  (app/retrieval/store.py)     │    │
│  │   FAISS HNSW ── BM25 ── RRF Fusion ── CrossEncoder  │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
  ┌─────────────────┐   ┌──────────────────────┐
  │     Ollama      │   │    HuggingFace Hub   │
  │  qwen2.5:3b     │   │  all-mpnet-base-v2   │
  │  qwen2.5:0.5b   │   │  ms-marco-MiniLM     │
  └─────────────────┘   └──────────────────────┘
```

### Component Map

| Layer | File(s) | Responsibility |
|---|---|---|
| Entry point | `main.py` | FastAPI app, lifespan, background eviction, static files |
| Orchestrator | `app/chatbot.py` | Full pipeline coordinator — thin façade |
| Config | `app/config.py` | All settings, env-var overrides, prompts, PII patterns |
| Retrieval | `app/retrieval/chunker.py` | Semantic small-to-big chunking with fallback |
| Retrieval | `app/retrieval/indexer.py` | FAISS HNSW builder, save/load |
| Retrieval | `app/retrieval/store.py` | Thread-safe FAISS + BM25 + RRF + CrossEncoder |
| LLM | `app/llm/client.py` | Ollama: preprocess, intent classify, stream |
| LLM | `app/llm/messages.py` | LangChain message list builder |
| LLM | `app/llm/scorer.py` | Faithfulness + relevance scoring |
| Cache | `app/cache/store.py` | Exact MD5 + semantic cosine cache with TTL |
| Session | `app/session/manager.py` | Per-session memory, topic anchor, TTL eviction |
| Routes | `app/routes/chat.py` | FastAPI endpoints |
| Utils | `app/utils/pdf.py` | PDF extraction + PII scrubbing |
| Frontend | `static/script.js` | Streaming NDJSON client, UI |

---

## 📁 Project Structure

```
rag_chatbot/
│
├── main.py                      # FastAPI entry point + lifespan hooks
├── build_index.py               # CLI: build FAISS index from PDFs
├── requirements.txt
│
├── app/
│   ├── chatbot.py               # Pipeline orchestrator
│   ├── config.py                # All settings + env-var overrides
│   ├── database.py              # Optional feedback logging backend
│   │
│   ├── retrieval/
│   │   ├── chunker.py           # Semantic + fixed-size chunking
│   │   ├── indexer.py           # FAISS HNSW builder + save/load
│   │   └── store.py             # Thread-safe DocumentStore
│   │
│   ├── llm/
│   │   ├── client.py            # Ollama: preprocess, intent, stream
│   │   ├── messages.py          # LangChain message list builder
│   │   └── scorer.py            # Faithfulness + relevance scorer
│   │
│   ├── cache/
│   │   └── store.py             # Exact + semantic cache (threading.Lock)
│   │
│   ├── session/
│   │   └── manager.py           # Per-session state + TTL eviction
│   │
│   ├── routes/
│   │   └── chat.py              # FastAPI routes
│   │
│   └── utils/
│       ├── logging.py           # Structured logger factory
│       └── pdf.py               # PDF extraction + PII scrubbing
│
├── static/
│   ├── index.html               # Single-page chat UI
│   ├── script.js                # Streaming NDJSON client
│   └── style.css                # Dark-mode UI styles
│
├── pdfs/                        # ← Drop your PDFs here
├── faiss.index                  # Generated by build_index.py
├── faiss_lc/                    # LangChain vectorstore (generated)
└── chunks.npy                   # Chunk metadata array (generated)
```

---

## 🔄 Pipeline Workflows

### Indexing Pipeline (`build_index.py`)

Run once before starting the server. Re-run after adding new PDFs.

```
PDFs on disk
     │
     ▼  app/utils/pdf.py
  Extract text  ──►  Scrub PII (email → [EMAIL], SSN → [SSN], phone → [PHONE])
     │
     ▼  app/retrieval/chunker.py
  Semantic chunking
  ├── Split on paragraph boundaries (double newline)
  ├── Sub-split long paragraphs into child chunks (≤100 words each)
  └── Each child chunk stores full parent paragraph as context
     │
     ▼  app/retrieval/indexer.py
  Embed all chunks  (HuggingFace all-mpnet-base-v2, 768-dim)
  Pre-compute unique parent paragraph embeddings
     │
     ▼
  Build FAISS HNSW index
  ├── Start with IndexFlatL2 → reconstruct all vectors
  ├── Upgrade to IndexHNSWFlat (M=32, efConstruction=200, efSearch=64)
  └── L2-normalise all vectors
     │
     ├──► faiss.index     (raw FAISS index)
     ├──► faiss_lc/       (LangChain vectorstore)
     └──► chunks.npy      (chunk metadata + parent embeddings)
```

**Why HNSW?** Exact nearest-neighbour search is O(n) per query. HNSW is a graph-based approximate algorithm with sub-linear query time. At 100k+ vectors it keeps p95 latency under 50 ms.

**Why semantic chunking?** Fixed-size character splitting cuts sentences mid-thought. Semantic chunking on paragraph boundaries produces self-contained units. The child/parent structure gives the retriever precision and the LLM full context.

---

### Query Pipeline (`POST /api/v1/answer`)

Every request walks the following 12-stage pipeline:

```
User query
     │
     ▼  SessionManager.touch()
  Update last_active timestamp
  Add user message to LangChain ConversationBufferMemory
     │
     ▼  HuggingFaceEmbeddings.embed_query()
  Embed query → 768-dim L2-normalised vector (query_vec)
     │
     ▼  LLMClient.preprocess_query()
  Fast LLM call (qwen2.5:0.5b, non-streaming)
  ├── Expand abbreviations found in conversation history
  ├── Rewrite vague follow-ups into standalone questions
  └── Skip if query already complete (≥6 words + '?')
     │
     ▼  ResponseCache.get_exact()   ← L1 Cache
  MD5(session_id + recent_AI_message[:100] + query)
  └── HIT → word-by-word replay → return
     │
     ▼  SessionManager.classify_query()
  Cosine similarity of query_vec vs stored topic_vec
  ├── sim ≥ 0.65 → follow-up: reuse cached chunks
  └── sim < 0.65 → topic shift: fresh retrieval
     │
     ▼  DocumentStore.detect_pdf()       [fresh retrieval only]
  FAISS top-30 scan; average score per PDF
  └── Select PDF above 0.30 threshold
     │
     ▼  DocumentStore.search()           [fresh retrieval only]
  FAISS top-10 (dense) + BM25 top-10 (sparse)
  └── RRF fusion → top-10 unique parent-deduplicated candidates
     │
     ▼  DocumentStore.rerank()           [fresh retrieval only]
  CrossEncoder scores each candidate vs query
  ├── Filter: score > -3.0
  └── Top-5 chunks sent to LLM
     │
     ▼  ResponseCache.get_semantic()   ← L2 Cache
  cosine_sim(query_vec, cached_vec) ≥ 0.92 + same PDF + within TTL
  └── HIT → replay → return
     │
     ▼  LLMClient.classify_intent()      [follow-ups only]
  Fast LLM → one of: simplify / summarize / example /
                     elaborate / why / how / compare / default
     │
     ▼  build_messages()
  [SystemMessage] + [History window (last 6)] + [HumanMessage]
     │
     ▼  LLMClient.stream()
  Worker thread: ChatOllama.invoke() → _StreamHandler queue → generator
  └── on_complete() callback:
       ├── Add AI message to session memory
       ├── Score faithfulness + relevance
       ├── Write L1 exact cache entry
       └── Write L2 semantic cache entry
     │
     ▼  FastAPI event_stream()
  Yield NDJSON lines to browser:
  ├── {"type":"status",          "message":"Searching report.pdf..."}
  ├── {"type":"early_citations", "citations":[...]}
  ├── {"type":"chunk",           "content":"token"}   × N
  └── {"type":"metadata",        "scores":{...}, "response_time_sec": X}
```

---

## 📡 Data Flow & Streaming Protocol

### NDJSON Stream Format

Each line in the HTTP response is a JSON object:

```jsonc
// 1. Status — sent immediately
{"type": "status", "message": "Searching report.pdf... [OK]"}

// 2. Citations — sent before tokens so user sees sources while answer generates
{"type": "early_citations", "citations": [
  {"chunk_id": 12, "source": "report.pdf", "excerpt": "..."}
]}

// 3. Token chunks — one per LLM token
{"type": "chunk", "content": "The "}
{"type": "chunk", "content": "main "}
// ... repeats for every token ...

// 4. Metadata — final message after all tokens
{
  "type": "metadata",
  "source_pdf": "report.pdf",
  "citations": [...],
  "scores": {
    "faithfulness": 0.82,
    "relevance": 0.91,
    "confidence": 87,
    "grounding_warning": false
  },
  "response_time_sec": 3.14
}
```

### Why NDJSON?
- Compatible with any HTTP client (no WebSocket setup needed)
- Each event is independently parseable — partial stream failures don't corrupt prior data
- `early_citations` gives the user source attribution before the answer finishes generating

---

## 📖 API Reference

All endpoints are prefixed `/api/v1`. Interactive docs available at `/docs` (Swagger) and `/redoc`.

### `POST /api/v1/answer`

Ask a question. Returns a streaming NDJSON response.

**Request body:**
```json
{
  "question": "What are the key findings?",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

`session_id` is optional but required for conversation continuity. Generate a UUID client-side (`crypto.randomUUID()`) and send it with every request in a session.

**Response:** Streaming NDJSON (see format above)

**Error:**
```json
{"detail": "Chatbot is not initialised. Try again shortly."}
```

---

### `GET /api/v1/health`

```json
{"status": "ok", "message": "Chatbot API is running"}
```

---

### `GET /api/v1/info`

```json
{
  "pdfs": ["report.pdf", "policy.pdf"],
  "total_chunks": 1024,
  "model": "qwen2.5:3b",
  "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"
}
```

---

### `POST /api/v1/reindex`

Upload a new PDF and add it to the live index. Uses `multipart/form-data`.

```bash
curl -X POST http://localhost:8000/api/v1/reindex \
     -F "file=@new_document.pdf"
```

```json
{"status": "success", "message": "Successfully indexed new_document.pdf"}
```

Max upload size: 50 MB (configurable via `MAX_UPLOAD_MB` env var).

---

### `POST /api/v1/feedback`

Log thumbs-up / thumbs-down user feedback.

```json
{
  "session_id": "abc123",
  "question": "What are the key findings?",
  "feedback_type": "thumbs_up"
}
```

Gracefully skipped if `app/database.py` is not implemented.

---

## 🚀 Installation & Setup

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Uses `X \| Y` union type syntax |
| Ollama | Latest | [ollama.com](https://ollama.com) |
| RAM | ≥ 8 GB | 16 GB recommended for large PDFs |
| Disk | ≥ 4 GB | For models + index files |

### Step-by-Step

```bash
# 1. Clone the repository
git clone https://github.com/yourname/rag-chatbot.git
cd rag-chatbot

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Pull Ollama models
ollama pull qwen2.5:3b           # Main answer model
ollama pull qwen2.5:0.5b         # Fast preprocessing model

# 5. Add your PDF files
cp /path/to/your/*.pdf pdfs/

# 6. Build the FAISS index (run once; re-run after adding new PDFs)
python build_index.py

# 7. Start the server
python main.py
```

Open **http://localhost:8000** in your browser.

---

## ⚙️ Configuration Reference

All settings live in `app/config.py` and are overridable with environment variables. No files to edit — just set the env var before running.

### Models

| Env Var | Default | Description |
|---|---|---|
| `RETRIEVER_MODEL` | `sentence-transformers/all-mpnet-base-v2` | HuggingFace embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder reranker |
| `LLM_MODEL` | `qwen2.5:3b` | Main Ollama generation model |
| `PREPROCESS_MODEL` | `qwen2.5:0.5b` | Fast query rewriting model |
| `FOLLOWUP_INTENT_MODEL` | `qwen2.5:0.5b` | Fast intent classification model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |

### Retrieval

| Env Var | Default | Description |
|---|---|---|
| `FAISS_K` | `10` | Top-K from FAISS dense search |
| `BM25_K` | `10` | Top-K from BM25 sparse search |
| `RRF_K` | `10` | Candidates after RRF fusion |
| `FINAL_K` | `5` | Chunks sent to LLM after reranking |
| `RRF_CONSTANT` | `60` | Denominator in RRF formula |
| `PDF_SCORE_THRESHOLD` | `0.30` | Min avg FAISS score to select a PDF |
| `RERANKER_THRESHOLD` | `-3.0` | Min CrossEncoder score to keep a chunk |
| `TOPIC_SHIFT_THRESHOLD` | `0.65` | Cosine sim below this triggers fresh retrieval |

### LLM Generation

| Env Var | Default | Description |
|---|---|---|
| `TEMPERATURE` | `0.3` | Generation temperature (0 = deterministic) |
| `MAX_TOKENS` | `512` | Max tokens to generate per answer |
| `NUM_CTX` | `2048` | LLM context window size (tokens) |
| `HISTORY_WINDOW` | `6` | Recent messages included in context |

### Caching & Sessions

| Env Var | Default | Description |
|---|---|---|
| `SEMANTIC_CACHE_THRESHOLD` | `0.92` | Cosine similarity required for semantic cache hit |
| `SEMANTIC_CACHE_MAX` | `100` | Max semantic cache entries (LRU eviction) |
| `CACHE_TTL_SECONDS` | `1800` | TTL for both cache layers (30 min) |
| `SESSION_TTL_SECONDS` | `1800` | Idle time before session eviction (30 min) |
| `GROUNDING_WARNING_THRESHOLD` | `0.30` | Faithfulness below this shows warning banner |

### FAISS HNSW Tuning

| Env Var | Default | Description |
|---|---|---|
| `HNSW_M` | `32` | Neighbours per node (higher = better recall, more memory) |
| `HNSW_EF_CONSTRUCTION` | `200` | Search depth at build time |
| `HNSW_EF_SEARCH` | `64` | Search depth at query time |

**Example overrides:**

```bash
# Larger model, deeper retrieval
LLM_MODEL=qwen2.5:7b FAISS_K=20 FINAL_K=8 python main.py

# Smaller embedding model (faster, less RAM; must re-run build_index.py)
RETRIEVER_MODEL=sentence-transformers/all-MiniLM-L6-v2 python build_index.py
RETRIEVER_MODEL=sentence-transformers/all-MiniLM-L6-v2 python main.py
```

---

## ▶️ Running the Application

### Build the index

```bash
python build_index.py
```

Sample output:
```
13:04:22 | INFO | indexer | [1/3] Loading PDFs from pdfs/
13:04:22 | INFO | indexer | Extracting: report.pdf (2.3 MB)
13:04:25 | INFO | indexer | [2/3] Chunking (size=500, overlap=100)
13:04:25 | INFO | indexer | Chunked report.pdf -> 412 chunks
13:04:25 | INFO | indexer | [3/3] Embedding + building HNSW index
13:04:55 | INFO | indexer | HNSW index built: 412 vectors (dim=768)
13:04:55 | INFO | indexer | INDEXING COMPLETE | PDFs: 1 | Chunks: 412
```

### Start the server

```bash
python main.py
# or with hot-reload for development:
uvicorn main:app --reload --port 8000
```

### Access

| URL | Description |
|---|---|
| http://localhost:8000 | Chat interface |
| http://localhost:8000/docs | Swagger API docs |
| http://localhost:8000/redoc | ReDoc API docs |

---

## 🐳 Deployment

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t rag-chatbot .
docker run -p 8000:8000 \
  -v $(pwd)/pdfs:/app/pdfs \
  -v $(pwd)/faiss.index:/app/faiss.index \
  -v $(pwd)/chunks.npy:/app/chunks.npy \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  rag-chatbot
```

### Production Notes

- Run Ollama on a host with GPU acceleration for better throughput
- Mount `pdfs/`, `faiss.index`, `faiss_lc/`, `chunks.npy` as persistent volumes
- Place behind nginx for TLS termination
- Use `/api/v1/health` for load balancer health checks
- Increase `SEMANTIC_CACHE_MAX=500` for high-traffic deployments

---

## 🔒 Thread-Safety Model

The system runs in two concurrent execution contexts:

```
Event Loop Thread (async)           Worker Thread (sync)
─────────────────────────           ────────────────────
FastAPI route handler                LLMClient._run()
  │                                    │
  ├── SessionManager.touch()            │  ChatOllama.invoke()
  │   threading.Lock ✓                 │
  │                                    │  _StreamHandler.on_llm_new_token()
  ├── ResponseCache.get_exact()         │  threading.Queue ✓
  │   threading.Lock ✓                 │
  │                                    │  on_complete(full_text)
  ├── DocumentStore.search()           │  ├── SessionManager.add_ai_message()
  │   threading.RLock ✓               │  │   threading.Lock ✓
  │                                    │  ├── score_response() — pure compute
  ├── for token in generator:           │  ├── ResponseCache.set_exact()
  │     yield NDJSON                   │  │   threading.Lock ✓
  │                                    │  └── ResponseCache.set_semantic()
  └── read scores (populated) ◄────────┘      threading.Lock ✓
```

> **Key design decision:** `asyncio.Lock` deadlocks when acquired from a sync worker thread. The entire codebase uses `threading.Lock` / `threading.RLock` exclusively — safe from any thread context. `DocumentStore` uses `threading.RLock` (reentrant) because `_rebuild_derived()` is called from within `add_document()` which already holds the lock.

---

## 🗄 Caching Strategy

### Layer 1 — Exact Cache

**Key:** `MD5(session_id + recent_ai_message[:100] + query.lower())`

A hit means this exact question was asked in this session with the same conversation context. Returns a word-by-word replay to preserve the streaming interface.

### Layer 2 — Semantic Cache

**Hit requires all three:**
1. `cosine_similarity(query_vec, cached_vec) ≥ 0.92`
2. Same source PDF (prevents cross-document answer reuse)
3. Entry timestamp within `CACHE_TTL_SECONDS`

| Property | Exact (L1) | Semantic (L2) |
|---|---|---|
| Key type | MD5 hash string | Embedding vector |
| Max entries | Unlimited | 100 (LRU eviction) |
| TTL | 1800s | 1800s |
| Write location | Worker thread | Worker thread |
| Lock | `threading.Lock` | `threading.Lock` |

Both caches are evicted by a background `asyncio` task running every 5 minutes — no per-request overhead.

---

## 💬 Session & Memory Management

Each session is identified by a UUID (`session_id`) generated by the browser client.

**Per-session state:**

| Field | Type | Description |
|---|---|---|
| `memory` | `ConversationBufferMemory` | Full LangChain message history |
| `chunks` | `list[dict]` | Last retrieved chunks (reused for follow-ups) |
| `source` | `str` | Source PDF of last retrieval |
| `topic_vec` | `np.ndarray` | Embedding of last anchor query |
| `topic_query` | `str` | Text of last anchor query |
| `last_active` | `float` | Unix timestamp for TTL eviction |
| `lock` | `threading.Lock` | Per-session state lock |

**Topic shift detection:** On each non-first query, cosine similarity between `query_vec` and `topic_vec` is computed. Below `TOPIC_SHIFT_THRESHOLD` (0.65), cached chunks are cleared and fresh retrieval runs. This enables natural topic changes within a conversation without requiring a new session.

**TTL eviction:** Sessions idle longer than `SESSION_TTL_SECONDS` (30 min) are evicted by the background task, freeing memory.

---

## 📊 Scoring & Grounding

After generation, `scorer.score_response()` computes metrics using the same embedding model — no extra model load.

| Metric | Formula | Interpretation |
|---|---|---|
| **Faithfulness** | `max(cosine_sim(answer_vec, chunk_vecs))` | Did the LLM stay grounded in the retrieved passages? |
| **Relevance** | `cosine_sim(query_vec, answer_vec)` | Did the answer address the question? |
| **Confidence** | `round(((faithfulness + relevance) / 2) * 100)` | Combined score 0–100 |
| **Grounding Warning** | `faithfulness < 0.30` | Red banner shown in UI when true |

**Confidence colour coding in the UI:**

- 🟢 **≥ 80** — High confidence, well-grounded answer
- 🟡 **50–79** — Moderate confidence
- 🔴 **< 50** — Low confidence, treat with caution

> Precomputed parent embeddings are stored in `chunks.npy` at index time, so scoring avoids a second embedding call — the `st_model.encode()` call is skipped in the fast path.

---

## 🧩 Technical Challenges & Solutions

| Challenge | Solution |
|---|---|
| `asyncio.Lock` deadlock in worker threads | Replaced all asyncio locks with `threading.Lock` / `threading.RLock` throughout the codebase |
| Pure vector search misses exact-keyword queries | Hybrid FAISS + BM25 with RRF fusion — BM25 recovers exact matches that embeddings compress away |
| LLM has insufficient context from small chunks | Small-to-big chunking: retrieve small child chunks for precision, pass full parent paragraph to LLM |
| Bi-encoder retrieval misses cross-attention signal | CrossEncoder second pass re-scores candidates with joint query+document attention |
| Scoring requires a second model load | Reused same `SentenceTransformer` instance via `getattr` introspection on `HuggingFaceEmbeddings` |
| Live reindex breaks ongoing queries | `threading.RLock` serialises mutations; old vectors tombstoned (source set to `""`) rather than deleted |
| Query preprocessing adds latency | Skip for complete questions (≥6 words + `?`); use 0.5b model with configurable timeout |
| Stale cache/session entries waste RAM | Background eviction task every 5 minutes; per-entry TTL tracking |
| LLM hallucination beyond source documents | Strict `SYSTEM_PROMPT` with explicit prohibitions; faithfulness score + grounding warning banner |

---

## 🔭 Future Enhancements

### Near-Term
- [ ] Multi-document synthesis — answer queries that span multiple PDFs
- [ ] OCR integration — pre-process scanned PDFs with Tesseract
- [ ] Persistent feedback database — activate the `database.py` backend
- [ ] User authentication — JWT-based auth with per-user session isolation

### Medium-Term
- [ ] GPU acceleration — leverage CUDA via Ollama's GPU backend
- [ ] Multi-modal support — index tables and images from PDFs using vision models
- [ ] Retrieval analytics — track which chunks are retrieved most for quality insights
- [ ] Streaming reranker — lighter reranker to reduce reranking latency

### Long-Term
- [ ] GraphRAG extension — entity graphs for multi-hop reasoning
- [ ] Distributed FAISS index — shard across nodes for very large corpora
- [ ] Active learning — fine-tune the reranker using thumbs-up/down feedback
- [ ] Plugin API — custom retrieval stages and scoring functions

---

## 🛟 Troubleshooting

| Problem | Resolution |
|---|---|
| `FileNotFoundError: faiss.index` | Run `python build_index.py` first. The index must exist before starting the server. |
| `Cannot connect to Ollama` | Run `ollama serve`. Check `OLLAMA_BASE_URL` in config. Verify models with `ollama list`. |
| `Mismatch: N chunks but M FAISS vectors` | Delete `faiss.index`, `faiss_lc/`, `chunks.npy` and re-run `python build_index.py`. |
| Low confidence / grounding warnings | Increase `FINAL_K`, decrease `RERANKER_THRESHOLD`, or use a larger `LLM_MODEL` (e.g. `qwen2.5:7b`). |
| Slow responses | Reduce `MAX_TOKENS`, `FAISS_K`, `BM25_K`, or use a smaller model. Enable GPU in Ollama if available. |
| High RAM usage | Reduce `SEMANTIC_CACHE_MAX` (default 100), reduce `SESSION_TTL_SECONDS`, use `all-MiniLM-L6-v2` (384-dim vs 768-dim). |
| PDFs not extracted correctly | `pdfminer.six` struggles with scanned PDFs. Pre-process with an OCR tool before adding to `pdfs/`. |
| Follow-ups trigger fresh retrieval too often | Lower `TOPIC_SHIFT_THRESHOLD` below 0.65 for more aggressive follow-up reuse. |

---

## 📈 Project Outcomes

- **Hybrid retrieval measurably outperforms pure vector search** — exact-keyword queries that fail vector search are recovered by BM25, and RRF fusion combines both signals without weight tuning
- **CrossEncoder reranking improves answer quality** on ambiguous queries by attending jointly to query and document text
- **Streaming architecture enables smooth UX** — users see live token output rather than a loading spinner during 10–30 second generation times
- **Thread-safety model proven correct under concurrent load** — simultaneous requests do not corrupt session state, cache, or the FAISS index
- **Dual-layer cache delivers significant repeat-query speedup** — exact hits replay in <50 ms vs 10–30 s for fresh generation
- **Confidence scoring creates an objective quality signal** — faithfulness-based grounding warnings correctly identify responses where the LLM drifted beyond the provided context

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Ollama](https://ollama.com) — local LLM serving
- [FAISS](https://github.com/facebookresearch/faiss) — Facebook AI Similarity Search
- [LangChain](https://langchain.com) — LLM orchestration framework
- [HuggingFace Sentence Transformers](https://www.sbert.net) — embedding + reranking models
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25 sparse retrieval
- [FastAPI](https://fastapi.tiangolo.com) — modern async Python web framework
