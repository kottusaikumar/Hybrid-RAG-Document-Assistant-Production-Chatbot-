# Hybrid RAG Chatbot

A fully local, production-grade Retrieval-Augmented Generation (RAG) chatbot that lets you ask questions about your own PDF documents. No API keys, no cloud, no data leaving your machine.

Built on **FAISS + BM25 hybrid search**, **CrossEncoder reranking**, **LangChain**, and **Ollama**. Streams answers token-by-token through a clean browser UI.

---

## Table of Contents

1. [What It Does](#1-what-it-does)
2. [Architecture Overview](#2-architecture-overview)
3. [How the Pipeline Works](#3-how-the-pipeline-works)
   - 3.1 [Indexing Pipeline](#31-indexing-pipeline)
   - 3.2 [Query Pipeline](#32-query-pipeline)
4. [Project Structure](#4-project-structure)
5. [Prerequisites](#5-prerequisites)
6. [Installation](#6-installation)
7. [Configuration Reference](#7-configuration-reference)
8. [Running the Application](#8-running-the-application)
9. [API Reference](#9-api-reference)
10. [Frontend](#10-frontend)
11. [Key Design Decisions](#11-key-design-decisions)
12. [Thread-Safety Model](#12-thread-safety-model)
13. [Caching Strategy](#13-caching-strategy)
14. [Session & Memory Management](#14-session--memory-management)
15. [Scoring & Grounding](#15-scoring--grounding)
16. [Adding New Documents (Live Reindex)](#16-adding-new-documents-live-reindex)
17. [Troubleshooting](#17-troubleshooting)
18. [Extending the Project](#18-extending-the-project)

---

## 1. What It Does

- **Ask questions** about one or more PDF documents in plain language
- **Hybrid retrieval**: combines dense vector search (FAISS) with sparse keyword search (BM25) and fuses results with Reciprocal Rank Fusion (RRF)
- **CrossEncoder reranking**: a second-pass model picks the most relevant passages before they reach the LLM
- **Streaming answers**: tokens are streamed to the browser as they are generated — no waiting for the full response
- **Conversation memory**: multi-turn conversations with topic-shift detection and follow-up intent classification
- **Dual-layer cache**: exact MD5 cache + semantic cosine cache, both with TTL expiry
- **Confidence scoring**: every answer is scored for faithfulness (did the LLM stick to the context?) and relevance (did it answer the question?)
- **Live reindexing**: upload a new PDF through the UI and it is searchable immediately
- **100% local**: all models run via Ollama and HuggingFace — nothing is sent to any external service

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Browser UI                              │
│          HTML + CSS + JS  ·  Streaming NDJSON over HTTP         │
└────────────────────────────┬────────────────────────────────────┘
                             │ POST /api/v1/answer
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI  (main.py)                           │
│              Routes · CORS · Static files · Lifespan            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 HybridRAGChatbot  (app/chatbot.py)              │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ SessionMgr  │  │ ResponseCache│  │    LLMClient           │ │
│  │ (threading) │  │ (threading)  │  │  preprocess · intent   │ │
│  └─────────────┘  └──────────────┘  │  classify · stream     │ │
│                                     └────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              DocumentStore  (app/retrieval/store.py)     │  │
│  │  FAISS HNSW ─── BM25 ─── RRF fusion ─── CrossEncoder    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
         ┌──────────────┐     ┌──────────────────┐
         │   Ollama     │     │  HuggingFace Hub  │
         │  (local LLM) │     │  (embeddings +    │
         │  qwen2.5:3b  │     │   cross-encoder)  │
         └──────────────┘     └──────────────────┘
```

### Component map

| Layer | File(s) | Responsibility |
|---|---|---|
| Entry point | `main.py` | FastAPI app, lifespan, middleware, static files |
| Orchestrator | `app/chatbot.py` | Full pipeline coordinator, thin façade |
| Config | `app/config.py` | All settings, env-var overrides, prompts |
| Retrieval | `app/retrieval/chunker.py` | Semantic chunking with fixed-size fallback |
| Retrieval | `app/retrieval/indexer.py` | FAISS HNSW index builder |
| Retrieval | `app/retrieval/store.py` | Thread-safe FAISS + BM25 + RRF + reranking |
| LLM | `app/llm/client.py` | Ollama calls: preprocess, intent, streaming |
| LLM | `app/llm/messages.py` | LangChain message list builder |
| LLM | `app/llm/scorer.py` | Faithfulness + relevance scoring |
| Cache | `app/cache/store.py` | Exact MD5 + semantic cosine cache with TTL |
| Session | `app/session/manager.py` | Per-session memory, topic anchor, eviction |
| Routes | `app/routes/chat.py` | FastAPI endpoints |
| Utils | `app/utils/pdf.py` | PDF extraction + PII scrubbing |
| Utils | `app/utils/logging.py` | Structured logging setup |
| CLI | `indexer.py` | One-shot index builder CLI |
| Frontend | `static/script.js` | Streaming NDJSON client, UI interactions |
| Frontend | `static/style.css` | Dark-mode UI, all component styles |

---

## 3. How the Pipeline Works

### 3.1 Indexing Pipeline

Run once with `python indexer.py` before starting the server.

```
PDFs on disk
     │
     ▼  app/utils/pdf.py
  Extract text  ──►  Scrub PII (email, SSN, phone)
     │
     ▼  app/retrieval/chunker.py
  Semantic chunking
  ├── Split on paragraph boundaries (double newline)
  ├── Sub-split long paragraphs into child chunks (≤100 words)
  └── Each child carries its full parent paragraph as context
     │
     ▼  app/retrieval/indexer.py
  Embed all chunks  (HuggingFace all-mpnet-base-v2, 768-dim)
     │
     ▼
  Build FAISS HNSW index
  ├── Flat index  →  reconstruct all vectors
  ├── Upgrade to IndexHNSWFlat (M=32, ef_construction=200)
  └── Normalize L2
     │
     ├──► faiss.index          (raw FAISS for chatbot)
     ├──► faiss_lc/            (LangChain vectorstore)
     └──► chunks.npy           (chunk metadata array)
```

**Why semantic chunking?** Fixed-size character splitting slices sentences mid-thought. Semantic chunking on paragraph boundaries produces self-contained units. The child/parent structure means small chunks are retrieved (precision) but the LLM sees the full paragraph context (recall).

**Why HNSW?** Exact nearest-neighbor search is O(n) per query. HNSW is a graph-based approximate search algorithm with sub-linear query time. At 100k+ vectors, HNSW keeps p95 latency under 50ms. At dev scale (thousands of vectors) the difference is negligible, but the index is already production-ready.

---

### 3.2 Query Pipeline

Every call to `GET /api/v1/answer` walks this pipeline:

```
User query
     │
     ▼  SessionManager
  Evict stale sessions + update last-active timestamp
  Add user message to LangChain ConversationBufferMemory
     │
     ▼  HuggingFaceEmbeddings
  Embed query  →  384-dim L2-normalized vector (query_vec)
     │
     ▼  LLMClient.preprocess_query()
  Fast LLM call (qwen2.5:0.5b, non-streaming)
  ├── Expand abbreviations found in conversation history
  ├── Rewrite vague follow-ups into standalone questions
  └── Skip if query already looks complete (≥6 words + '?')
     │
     ▼  ResponseCache.get_exact()
  Exact cache check  (MD5 of session_id + recent_ai + query)
  └── HIT → replay cached answer word-by-word → return
     │
     ▼  SessionManager.classify_query()
  Topic shift detection
  ├── Cosine similarity of query_vec vs stored topic anchor
  ├── sim ≥ 0.65  →  follow-up: reuse cached chunks
  └── sim < 0.65  →  topic shift: fresh retrieval
     │
     ▼  DocumentStore.detect_pdf()   [only on fresh retrieval]
  Score all PDFs: average FAISS score of top-30 results per PDF
  └── Best PDF above threshold (0.30) is selected
     │
     ▼  DocumentStore.search()       [only on fresh retrieval]
  Hybrid search restricted to selected PDF
  ├── FAISS: top-10 dense hits
  ├── BM25:  top-10 sparse hits (tokenized, punctuation-stripped)
  └── RRF fusion  →  top-10 unique parent-deduplicated candidates
     │
     ▼  DocumentStore.rerank()       [only on fresh retrieval]
  CrossEncoder scores each candidate vs query
  ├── Filter: score must be > -3.0
  └── Top-5 chunks sent to LLM
     │
     ▼  ResponseCache.get_semantic()
  Semantic cache check  (cosine similarity ≥ 0.92, source-aware)
  └── HIT → replay cached answer → return
     │
     ▼  LLMClient.classify_intent()  [only on follow-ups]
  Fast LLM call → one of: simplify / summarize / example /
                          elaborate / why / how / compare / default
     │
     ▼  build_messages()
  Assemble LangChain message list
  ├── [SystemMessage]   — strict grounding rules
  ├── [History window]  — last 6 messages
  └── [HumanMessage]    — context + query + intent instruction
     │
     ▼  LLMClient.stream()
  Streaming generation (qwen2.5:3b via LangChain ChatOllama)
  ├── Worker thread: LLM → _StreamHandler queue → token generator
  └── on_complete() callback (sync, in worker thread):
       ├── Add AI message to session memory
       ├── Score faithfulness + relevance
       ├── Write exact cache entry
       └── Write semantic cache entry
     │
     ▼  FastAPI event_stream()
  Yield NDJSON lines to browser:
  ├── {"type":"status",         "message":"..."}
  ├── {"type":"early_citations","citations":[...]}
  ├── {"type":"chunk",          "content":"token"}   × N
  └── {"type":"metadata",       "scores":{...}, ...}
```

---

## 4. Project Structure

```
rag_chatbot/
│
├── main.py                      # FastAPI entry point
├── indexer.py                   # CLI: build FAISS index from PDFs
│
├── app/
│   ├── __init__.py
│   ├── chatbot.py               # Orchestrator — thin pipeline façade
│   ├── config.py                # All settings + env-var overrides
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── chunker.py           # Semantic small-to-big chunking
│   │   ├── indexer.py           # FAISS HNSW builder + save/load
│   │   └── store.py             # Thread-safe DocumentStore
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py            # Ollama: preprocess, intent, stream
│   │   ├── messages.py          # LangChain message list builder
│   │   └── scorer.py            # Faithfulness + relevance scorer
│   │
│   ├── cache/
│   │   ├── __init__.py
│   │   └── store.py             # Exact + semantic cache (threading.Lock)
│   │
│   ├── session/
│   │   ├── __init__.py
│   │   └── manager.py           # Per-session state + TTL eviction
│   │
│   ├── routes/
│   │   ├── __init__.py
│   │   └── chat.py              # FastAPI routes (/answer, /health, ...)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py           # Structured logger factory
│       └── pdf.py               # PDF extraction + PII scrubbing
│
├── static/
│   ├── index.html               # Single-page chat UI
│   ├── script.js                # Streaming NDJSON client
│   └── style.css                # Dark-mode UI styles
│
├── pdfs/                        # Drop your PDFs here
├── faiss.index                  # Generated by indexer.py
├── faiss_lc/                    # Generated by indexer.py
├── chunks.npy                   # Generated by indexer.py
└── requirements.txt
```

---

## 5. Prerequisites

### System

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Uses `str \| None` union syntax |
| Ollama | Latest | [ollama.com](https://ollama.com) |
| RAM | ≥ 8 GB | 16 GB recommended for larger PDFs |
| Disk | ≥ 4 GB | For models + index files |

### Ollama models

Pull these before running:

```bash
ollama pull qwen2.5:3b        # Main answer model
ollama pull qwen2.5:0.5b      # Fast model for preprocessing + intent
```

You can swap any model supported by Ollama — see [Configuration Reference](#7-configuration-reference).

### Python packages

All declared in `requirements.txt`. Key dependencies:

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | Async web server |
| `langchain` + `langchain-ollama` | LLM orchestration + streaming |
| `langchain-huggingface` | HuggingFace embeddings wrapper |
| `faiss-cpu` | Vector similarity search |
| `sentence-transformers` | Embedding model + CrossEncoder |
| `rank-bm25` | BM25 sparse retrieval |
| `pdfminer.six` | PDF text extraction |
| `numpy` + `scikit-learn` | Vector math + cosine similarity |

---

## 6. Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourname/rag-chatbot.git
cd rag-chatbot

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull Ollama models
ollama pull qwen2.5:3b
ollama pull qwen2.5:0.5b

# 5. Add your PDFs
cp /path/to/your/*.pdf pdfs/

# 6. Build the index  (run once; re-run after adding new PDFs)
python indexer.py

# 7. Start the server
python main.py
```

Open **http://localhost:8000** in your browser.

---

## 7. Configuration Reference

Every setting lives in `app/config.py` and can be overridden with environment variables. No config files to edit — just set the env var before running.

### Models

| Env var | Default | Description |
|---|---|---|
| `RETRIEVER_MODEL` | `sentence-transformers/all-mpnet-base-v2` | HuggingFace embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder reranker |
| `LLM_MODEL` | `qwen2.5:3b` | Main Ollama generation model |
| `PREPROCESS_MODEL` | `qwen2.5:0.5b` | Fast model for query rewriting |
| `FOLLOWUP_INTENT_MODEL` | `qwen2.5:0.5b` | Fast model for intent classification |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server base URL |

### Chunking

| Env var | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `500` | Max words per fixed-size fallback chunk |
| `CHUNK_OVERLAP` | `100` | Overlap words in fixed-size fallback |
| `WORDS_PER_CHILD_CHUNK` | `100` | Max words per semantic child chunk |
| `MIN_PARA_LEN` | `50` | Min characters to keep a paragraph |
| `MIN_SENT_LEN` | `10` | Min characters to keep a sentence |

### Retrieval

| Env var | Default | Description |
|---|---|---|
| `FAISS_K` | `10` | Top-K from FAISS dense search |
| `BM25_K` | `10` | Top-K from BM25 sparse search |
| `RRF_K` | `10` | Candidates after RRF fusion |
| `FINAL_K` | `5` | Chunks sent to LLM after reranking |
| `RRF_CONSTANT` | `60` | Denominator in RRF formula |
| `PDF_DETECT_POOL` | `30` | Top-N FAISS hits used to select PDF |
| `PDF_SCORE_THRESHOLD` | `0.30` | Min avg FAISS score to select a PDF |
| `RERANKER_THRESHOLD` | `-3.0` | Min CrossEncoder score to keep a chunk |
| `TOPIC_SHIFT_THRESHOLD` | `0.65` | Cosine sim below this triggers fresh retrieval |

### FAISS HNSW

| Env var | Default | Description |
|---|---|---|
| `HNSW_M` | `32` | Number of neighbours per node |
| `HNSW_EF_CONSTRUCTION` | `200` | Search depth during index build |
| `HNSW_EF_SEARCH` | `64` | Search depth at query time |

Increasing `HNSW_M` and `HNSW_EF_CONSTRUCTION` improves recall at the cost of index build time and memory. `HNSW_EF_SEARCH` trades query speed for recall at runtime.

### LLM generation

| Env var | Default | Description |
|---|---|---|
| `TEMPERATURE` | `0.3` | Generation temperature (0 = deterministic) |
| `FOLLOWUP_TEMPERATURE_BOOST` | `0.2` | Added to temperature for follow-up replies |
| `FOLLOWUP_TEMPERATURE_MAX` | `0.6` | Cap on boosted follow-up temperature |
| `NUM_CTX` | `2048` | LLM context window size (tokens) |
| `MAX_TOKENS` | `512` | Max tokens to generate per answer |
| `HISTORY_WINDOW` | `6` | Recent messages included in context |

### Intent classifier

| Env var | Default | Description |
|---|---|---|
| `FOLLOWUP_INTENT_TIMEOUT` | `10` | Seconds before falling back to "default" intent |
| `INTENT_CLASSIFIER_CTX` | `256` | Context window for the classifier call |
| `INTENT_CLASSIFIER_TOKENS` | `5` | Max tokens the classifier may output |

### Query preprocessor

| Env var | Default | Description |
|---|---|---|
| `PREPROCESS_TIMEOUT` | `60` | Seconds before giving up on preprocessing |
| `PREPROCESS_CTX` | `512` | Context window for preprocessor call |
| `PREPROCESS_HISTORY_WINDOW` | `6` | Messages sent to preprocessor for context |
| `MIN_WORDS_COMPLETE_QUERY` | `6` | Queries with ≥ this many words + '?' are skipped |

### Caching

| Env var | Default | Description |
|---|---|---|
| `SEMANTIC_CACHE_THRESHOLD` | `0.92` | Cosine similarity required for a semantic cache hit |
| `SEMANTIC_CACHE_MAX` | `100` | Maximum semantic cache entries (LRU eviction) |
| `CACHE_TTL_SECONDS` | `1800` | Seconds before a cache entry is considered stale |

### Sessions

| Env var | Default | Description |
|---|---|---|
| `SESSION_TTL_SECONDS` | `1800` | Idle time before a session is evicted |

### Scoring

| Env var | Default | Description |
|---|---|---|
| `GROUNDING_WARNING_THRESHOLD` | `0.30` | Faithfulness below this triggers a grounding warning |

### Server

| Env var | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |

**Example** — using a different LLM and higher FAISS recall:

```bash
LLM_MODEL=llama3.2:3b FAISS_K=20 FINAL_K=8 python main.py
```

---

## 8. Running the Application

### Index your documents

```bash
python indexer.py
```

Output:
```
13:04:22 | INFO     | indexer | [1/3] Loading PDFs from pdfs/
13:04:22 | INFO     | indexer | Extracting: report.pdf (2.3 MB)
13:04:25 | INFO     | indexer | [2/3] Chunking (size=500, overlap=100)
13:04:25 | INFO     | indexer | Chunked report.pdf -> 412 chunks
13:04:25 | INFO     | indexer | [3/3] Embedding + building HNSW index
13:04:55 | INFO     | indexer | HNSW index built: 412 vectors (dim=768)
13:04:55 | INFO     | indexer | INDEXING COMPLETE | PDFs: 1 | Chunks: 412
```

Re-run `python indexer.py` whenever you add new PDFs to `pdfs/` and want to rebuild the full index. For adding a single document without rebuilding, use the live reindex endpoint instead (see [§16](#16-adding-new-documents-live-reindex)).

### Start the server

```bash
python main.py
# or
uvicorn main:app --reload --port 8000
```

### Access the UI

| URL | Description |
|---|---|
| http://localhost:8000 | Chat interface |
| http://localhost:8000/docs | Interactive Swagger API docs |
| http://localhost:8000/redoc | ReDoc API docs |

---

## 9. API Reference

All endpoints are prefixed `/api/v1`.

### `POST /answer`

Ask a question. Returns a streaming NDJSON response.

**Request body:**
```json
{
  "question": "What are the key findings?",
  "session_id": "abc123"
}
```

`session_id` is optional but required for conversation continuity. Generate a UUID client-side and send it with every request in a session.

**Streaming response** — each line is a JSON object:

```jsonc
{"type": "status",          "message": "Searching report.pdf... [OK]"}
{"type": "early_citations", "citations": [{"chunk_id": 12, "source": "report.pdf", "excerpt": "..."}]}
{"type": "chunk",           "content": "The "}
{"type": "chunk",           "content": "main "}
// ... one line per token ...
{"type": "metadata",
 "source_pdf": "report.pdf",
 "citations":  [...],
 "scores":     {"faithfulness": 0.82, "relevance": 0.91, "confidence": 87, "grounding_warning": false},
 "response_time_sec": 3.14}
```

**Error response** (4xx/5xx):
```json
{"detail": "Chatbot is not initialised. Try again shortly."}
```

### `GET /health`

```json
{"status": "ok", "message": "Chatbot API is running"}
```

### `GET /info`

```json
{
  "pdfs":         ["report.pdf", "policy.pdf"],
  "total_chunks": 1024,
  "model":        "qwen2.5:3b",
  "reranker":     "cross-encoder/ms-marco-MiniLM-L-6-v2"
}
```

### `POST /reindex`

Upload a new PDF and add it to the live index. Uses `multipart/form-data`.

```bash
curl -X POST http://localhost:8000/api/v1/reindex \
     -F "file=@new_document.pdf"
```

```json
{"status": "success", "message": "Successfully indexed new_document.pdf"}
```

### `POST /feedback`

Log thumbs-up / thumbs-down feedback. Requires `app/database.py` to be present (optional — gracefully skipped if absent).

```json
{
  "session_id":    "abc123",
  "question":      "What are the key findings?",
  "feedback_type": "thumbs_up"
}
```

---

## 10. Frontend

The UI is a single HTML page (`static/index.html`) with vanilla JavaScript (`static/script.js`) — no build step, no framework.

### Key behaviours

**Streaming rendering**: The JavaScript reads the NDJSON stream line-by-line using the Streams API (`response.body.getReader()`). Tokens accumulate in a `data-raw` attribute on the text element and are re-rendered through the markdown parser on each chunk, so the cursor blinks at the real end of the growing text.

**Markdown parsing order** (`parseMarkdown` in `script.js`): Backtick code spans are extracted before HTML escaping, so `<` and `>` inside code are safe. After extraction: HTML escape → bold → restore code spans → newlines. This ordering prevents `<code>` from being double-escaped and ensures `**bold**` inside a code span is left raw.

**Grounding warning**: When `scores.grounding_warning` is `true` in the metadata frame, a red banner is inserted below the answer reading *"Low faithfulness score — this answer may not be fully grounded in the document."* This was missing from the original implementation.

**Collapsible citations**: Rendered before streaming starts (`early_citations` frame) so the user can see sources while the answer is generating. Collapsed by default with a caret toggle.

**Session ID**: Generated once on page load using `crypto.randomUUID()` and sent with every request. Changing the page or refreshing starts a new session.

---

## 11. Key Design Decisions

### Why hybrid search instead of pure vector search?

Vector search excels at semantic similarity — paraphrased questions, conceptual queries. BM25 excels at exact matches — names, error codes, document IDs, technical terms that embeddings compress into shared space with similar-looking tokens. Real user queries span both types. Hybrid search with RRF fusion covers the full spectrum without tuning a weight between the two.

### Why CrossEncoder reranking?

The bi-encoder embedding model used for retrieval encodes the query and each document independently, then measures cosine distance. This is fast but loses the interaction signal between query and document. A cross-encoder sees the query and each candidate together (full attention across both), producing a far more accurate relevance score. The cost is O(candidates) inference calls — manageable when applied to the top 10 RRF results, not the full corpus.

### Why small-to-big chunking?

Retrieving small, precise child chunks maximises retrieval precision. But the LLM benefits from broader context. Each child chunk carries its full parent paragraph, so the retriever finds the right passage and the LLM sees enough context to give a complete answer.

### Why `threading.Lock` instead of `asyncio.Lock`?

The answer generator (`scored_generator` / `on_complete`) runs in a `threading.Thread` (spawned by `LLMClient.stream`). `asyncio.Lock` can only be acquired from the event loop thread — acquiring it from a worker thread raises or deadlocks silently. `threading.Lock` is safe to acquire from any thread and can be used with `with` statements from both sync and async code. All shared mutable state (`ResponseCache`, `SessionManager`, `DocumentStore`) uses `threading.Lock` internally.

### Why no `asyncio.Lock` at all?

The original codebase mixed `asyncio.Lock` (declared in `__init__`) with sync worker threads, creating race conditions on cache writes. The fix is uniform: use `threading.Lock` everywhere since the actual concurrent execution happens in threads, not coroutines. FastAPI's async route handlers call `get_response` and then iterate the returned generator synchronously in `event_stream` — the async boundary is only at the HTTP layer.

### Why separate `scored_generator` / `on_complete`?

Scores are computed from the complete generated text, which is only available after the last token. The `on_complete` callback fires synchronously in the worker thread immediately after the last token is placed in the queue, before the generator returns. This guarantees that by the time `event_stream` finishes iterating the generator and reads `scores`, the dict is fully populated. The caller never needs to poll or wait.

---

## 12. Thread-Safety Model

```
Event loop thread (async)          Worker thread (sync)
─────────────────────────          ────────────────────
FastAPI route handler               LLMClient._run()
  │                                   │
  ├── SessionManager.touch()           │  LangChain ChatOllama.invoke()
  │   threading.Lock ✓                │
  │                                   │  _StreamHandler.on_llm_new_token()
  ├── ResponseCache.get_exact()        │  threading.Queue ✓
  │   threading.Lock ✓                │
  │                                   │  on_complete(full_text)
  ├── SessionManager.classify_query()  │  ├── SessionManager.add_ai_message()
  │   threading.Lock ✓                │  │   threading.Lock ✓
  │                                   │  ├── scorer.score_response()
  ├── DocumentStore.search()          │  │   (pure computation, no lock needed)
  │   threading.RLock ✓              │  ├── ResponseCache.set_exact()
  │                                   │  │   threading.Lock ✓
  ├── for token in generator:          │  └── ResponseCache.set_semantic()
  │     yield NDJSON                  │      threading.Lock ✓
  │                                   │
  └── read scores (populated)     ◄───┘
```

**Rules:**
- All shared mutable state is protected by `threading.Lock` or `threading.RLock`
- Locks are never held across I/O operations (no deadlock risk)
- `DocumentStore` uses `threading.RLock` (reentrant) because `_rebuild_derived` is called from `add_document` which itself holds the lock
- Cache writes happen in the worker thread after generation, not in the event loop — this is intentional and safe because `threading.Lock` is used

---

## 13. Caching Strategy

### Layer 1: Exact cache

Key: `MD5(session_id + recent_ai_message[:100] + query.lower())`

A hit means this exact question was asked in this session with the same conversation context. Returns a word-by-word replay of the cached text to preserve the streaming interface.

### Layer 2: Semantic cache

Checked after PDF detection. Each entry stores:
- The query embedding vector
- The full response payload
- The source PDF name
- A timestamp

A hit requires:
1. `cosine_similarity(query_vec, cached_vec) ≥ 0.92`
2. Same source PDF (prevents cross-document answer reuse)
3. Entry timestamp within `CACHE_TTL_SECONDS` (1800s default)

Both caches are evicted at the start of each request cycle via `evict_stale()`. After a `add_document()` call, stale semantic cache entries for the updated PDF will naturally expire within `CACHE_TTL_SECONDS`.

---

## 14. Session & Memory Management

Each session is identified by a `session_id` string (UUID generated client-side). The `SessionManager` registry maps session IDs to `SessionState` objects.

**Per-session state:**
- `memory`: LangChain `ConversationBufferMemory` — the full message history
- `chunks`: last retrieved chunks (reused for follow-up queries)
- `source`: source PDF of the last retrieval
- `topic_vec`: embedding of the last "anchor" query
- `topic_query`: text of the last anchor query
- `last_active`: float timestamp for TTL eviction
- `lock`: `threading.Lock` for this session's state

**Topic shift detection:** On each non-first query, the cosine similarity between `query_vec` and `topic_vec` is computed. If similarity < `TOPIC_SHIFT_THRESHOLD` (0.65), the session's cached chunks are cleared and full retrieval runs again with the new query as the anchor. This allows natural topic changes within a conversation without the user having to start a new session.

**TTL eviction:** `SessionManager.evict_stale()` is called at the start of every request. Sessions idle for longer than `SESSION_TTL_SECONDS` (30 minutes) are removed from the registry, freeing memory.

---

## 15. Scoring & Grounding

After the LLM finishes generating, `scorer.score_response()` computes two metrics using the same embedding model used for retrieval (no extra model load):

**Faithfulness** — `max(cosine_similarity(answer_vec, chunk_vecs))`: how similar the generated answer is to the best-matching retrieved chunk. Low faithfulness (< 0.30) means the LLM likely generated content not present in the context.

**Relevance** — `cosine_similarity(query_vec, answer_vec)`: how similar the answer is to the original question. Low relevance means the answer drifted off-topic.

**Confidence** — `round(((faithfulness + relevance) / 2) * 100)`: a single 0–100 score displayed in the UI with colour coding (green ≥ 80, amber 50–79, red < 50).

**Grounding warning** — if faithfulness < `GROUNDING_WARNING_THRESHOLD` (0.30), a red warning banner is shown below the answer.

---

## 16. Adding New Documents (Live Reindex)

You can add a PDF to the running server without restarting:

**Via the UI:** Click the **Reindex PDF** button in the header and select a file.

**Via the API:**
```bash
curl -X POST http://localhost:8000/api/v1/reindex \
     -F "file=@new_document.pdf"
```

**What happens internally (`DocumentStore.add_document`):**
1. PDF is saved to `pdfs/`
2. Text is extracted and PII-scrubbed
3. Document is semantically chunked
4. All chunks are embedded in batch
5. Vectors are added to the live FAISS index
6. Chunk metadata is appended to `self._chunks`
7. BM25 index is built for the new PDF
8. `faiss.index` and `chunks.npy` are persisted to disk
9. LangChain vectorstore is synced to `faiss_lc/`

**Entire mutation is serialised** under `DocumentStore._lock` (a `threading.RLock`). Concurrent queries continue using the previous index state until the lock is released.

---

## 17. Troubleshooting

### `FileNotFoundError: faiss.index`

Run `python indexer.py` first. The server requires a built index.

### `Cannot connect to Ollama`

Ensure Ollama is running (`ollama serve`) and the models are pulled:
```bash
ollama list
ollama pull qwen2.5:3b
ollama pull qwen2.5:0.5b
```

Check `OLLAMA_BASE_URL` in config if Ollama is on a different host or port.

### `Mismatch: N chunks but M FAISS vectors`

The chunk store and FAISS index are out of sync. Delete `faiss.index`, `faiss_lc/`, and `chunks.npy`, then re-run `python indexer.py`.

### Answers are not grounded / low confidence

- Increase `FINAL_K` to give the LLM more context (default 5)
- Decrease `RERANKER_THRESHOLD` to be less aggressive at filtering (default -3.0)
- Try a larger `LLM_MODEL` (e.g. `qwen2.5:7b`)
- Check that the PDF was extracted correctly — `pdfminer.six` struggles with scanned PDFs. Use OCR pre-processing if needed.

### Slow responses

- Reduce `FAISS_K` and `BM25_K` (less retrieval, faster RRF)
- Reduce `MAX_TOKENS` for shorter answers
- Use a smaller `LLM_MODEL` (`qwen2.5:0.5b`)
- Enable GPU acceleration in Ollama if available

### High RAM usage

- Reduce `SEMANTIC_CACHE_MAX` (default 100)
- Reduce `SESSION_TTL_SECONDS` to evict sessions sooner
- Use a smaller embedding model (e.g. `all-MiniLM-L6-v2`, 384-dim vs 768-dim)

---

## 18. Extending the Project

### Swap the embedding model

Change `RETRIEVER_MODEL` to any HuggingFace sentence-transformers model. Re-run `python indexer.py` — the new model's dimension will be picked up automatically.

```bash
RETRIEVER_MODEL=sentence-transformers/all-MiniLM-L6-v2 python indexer.py
RETRIEVER_MODEL=sentence-transformers/all-MiniLM-L6-v2 python main.py
```

### Swap the LLM

Any model available in Ollama works. Pull it and set `LLM_MODEL`:
```bash
ollama pull mistral:7b
LLM_MODEL=mistral:7b python main.py
```

### Add a new follow-up intent

In `app/config.py`, add one entry to `INTENT_INSTRUCTIONS`:
```python
"define": (
    "Provide a precise definition of the key term in the question, "
    "using only the context. Quote the source if helpful."
),
```

Then add one line to `INTENT_CLASSIFIER_SYSTEM`:
```
  define     → user wants a precise definition of a term
```

No other code changes needed. The LLM classifier will now route to this intent.

### Add feedback storage

Create `app/database.py` with a `log_feedback(session_id, question, feedback_type)` function. The route handler already calls it; it currently skips gracefully if the module is absent.

```python
# app/database.py  (minimal SQLite example)
import sqlite3, pathlib

DB = pathlib.Path("feedback.db")

def _init():
    with sqlite3.connect(DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY, ts REAL,
                session_id TEXT, question TEXT, feedback TEXT
            )""")

_init()

def log_feedback(session_id, question, feedback_type):
    import time
    with sqlite3.connect(DB) as conn:
        conn.execute(
            "INSERT INTO feedback VALUES (NULL,?,?,?,?)",
            (time.time(), session_id, question, feedback_type)
        )
```

### Run with Docker

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

---

## License

MIT
