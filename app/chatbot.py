"""
app/chatbot.py
--------------
HybridRAGChatbot — the public façade used by FastAPI routes.

Orchestrates the full pipeline:
  Evict → Embed → Cache check → PDF detect → Hybrid search →
  Rerank → Build messages → Stream → Score → Cache write

All heavy logic (retrieval, caching, session, LLM) lives in the
sub-packages. This file is intentionally thin.

Thread-safety contract
----------------------
- DocumentStore uses threading.RLock internally for index mutations.
- ResponseCache uses threading.Lock — safe to call from scored_generator
  (worker thread) without asyncio.
- SessionManager uses threading.Lock per session — safe from both sync
  and async contexts.
- No asyncio.Lock is used anywhere, eliminating the original race where
  asyncio locks were acquired from sync worker threads.
"""

import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.utils.logging import get_logger
from app.retrieval import DocumentStore
from app.cache import ResponseCache
from app.session import SessionManager
from app.llm import LLMClient, score_response
from app.llm.messages import build_messages

log = get_logger(__name__)

# Response returned when retrieval comes up empty
_NOT_FOUND = "This question is not covered in the provided PDFs."


class HybridRAGChatbot:

    def __init__(self) -> None:
        log.info("Initialising HybridRAGChatbot...")

        # ── Embedding model ────────────────────────────────────────────────────
        log.info("Loading embeddings: %s", settings.RETRIEVER_MODEL)
        self._embeddings = HuggingFaceEmbeddings(
            model_name    = settings.RETRIEVER_MODEL,
            model_kwargs  = {"device": "cpu"},
            encode_kwargs = {"normalize_embeddings": True},
        )
        # Raw SentenceTransformer reference for scoring (avoids re-embedding)
        self._st_model = _get_st_model(self._embeddings)

        # ── Sub-systems ────────────────────────────────────────────────────────
        self._store   = DocumentStore(
            faiss_index_path = settings.FAISS_INDEX_PATH,
            chunk_store_path = settings.CHUNK_STORE_PATH,
            embedding_model  = self._embeddings,
        )
        self._cache   = ResponseCache()
        self._sessions = SessionManager()
        self._llm     = LLMClient()

        log.info("HybridRAGChatbot ready (%d PDFs, %d chunks)",
                 len(self._store.pdf_names), self._store.chunk_count)

    # ── Public properties ──────────────────────────────────────────────────────

    @property
    def pdf_names(self) -> list[str]:
        return self._store.pdf_names

    @property
    def chunk_count(self) -> int:
        return self._store.chunk_count

    # ── Document ingestion ─────────────────────────────────────────────────────

    def add_document(self, pdf_path: str, filename: str) -> bool:
        return self._store.add_document(pdf_path, filename)

    # ── Main pipeline ──────────────────────────────────────────────────────────

    def get_response(
        self, query: str, session_id: str | None = None
    ) -> tuple:
        """
        Full RAG pipeline. Returns:
            (token_generator, source_pdfs: list[str], citations: list[dict], scores: dict)

        scores is a mutable dict that is populated by scored_generator after
        the last token is yielded. Callers (routes/chat.py) must drain the
        generator fully before reading scores.
        """
        scores = {
            "faithfulness"    : None,
            "relevance"       : None,
            "confidence"      : None,
            "grounding_warning": False,
        }

        query = query.strip()
        if not query:
            return _NOT_FOUND, [], [], scores

        # NOTE: evict_stale() is now run by the background task in main.py
        # and is no longer called on every request.

        if session_id:
            self._sessions.touch(session_id)
            self._sessions.add_user_message(session_id, query)

        log.info("Query: %s", query)

        # ── Embed once; reuse for detection, search, and cache ────────────────
        query_vec = np.array(
            [self._embeddings.embed_query(query)], dtype=np.float32
        )
        faiss.normalize_L2(query_vec)

        # ── Preprocess (read history inside session lock snapshot) ────────────
        history = self._sessions.get_history(session_id, settings.PREPROCESS_HISTORY_WINDOW) if session_id else []
        search_query = self._llm.preprocess_query(query, history)

        # Re-embed if preprocessing changed the query
        if search_query != query:
            query_vec = np.array(
                [self._embeddings.embed_query(search_query)], dtype=np.float32
            )
            faiss.normalize_L2(query_vec)

        # ── Exact cache ───────────────────────────────────────────────────────
        recent_ai  = self._sessions.get_last_ai_message(session_id) if session_id else ""
        cache_key  = self._cache.make_key(session_id or "", recent_ai, query)
        cached     = self._cache.get_exact(cache_key)
        if cached:
            log.info("Exact cache hit")
            scores.update(cached.get("scores", {}))
            return _replay(cached["text"]), cached["sources"], cached["citations"], scores

        # ── Follow-up vs new topic ────────────────────────────────────────────
        if session_id:
            is_followup, cached_chunks, selected_pdf = self._sessions.classify_query(
                session_id, query_vec
            )
        else:
            is_followup, cached_chunks, selected_pdf = False, [], None

        # ── Retrieval (only when not a follow-up) ─────────────────────────────
        if is_followup:
            retrieved_chunks = cached_chunks
            log.info("Follow-up: reusing %d cached chunks from '%s'", len(retrieved_chunks), selected_pdf)
        else:
            selected_pdf = self._store.detect_pdf(query_vec)
            if not selected_pdf:
                return _NOT_FOUND, [], [], scores

            rrf_candidates = self._store.search(search_query, selected_pdf, query_vec)
            if not rrf_candidates:
                return _NOT_FOUND, [], [], scores

            retrieved_chunks = self._store.rerank(search_query, rrf_candidates)
            if not retrieved_chunks:
                return _NOT_FOUND, [], [], scores

            if session_id:
                self._sessions.update_topic(session_id, query, query_vec, retrieved_chunks, selected_pdf)

        # ── Semantic cache (source-aware, checked after PDF is known) ─────────
        sem_cached = self._cache.get_semantic(query_vec, selected_pdf)
        if sem_cached:
            if session_id:
                self._sessions.add_ai_message(session_id, sem_cached["text"])
            scores.update(sem_cached.get("scores", {}))
            return _replay(sem_cached["text"]), sem_cached["sources"], sem_cached["citations"], scores

        # ── Build context + citations ──────────────────────────────────────────
        context = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks
        )
        citations = [
            {
                "chunk_id": c.get("chunk_id", i),
                "source"  : c.get("source", ""),
                "excerpt" : c.get("text", "")[:200].strip(),
            }
            for i, c in enumerate(retrieved_chunks)
        ]
        source_pdfs = [selected_pdf]

        # ── Previous answer + intent instruction ──────────────────────────────
        previous_answer    = self._sessions.get_last_ai_message(session_id) if session_id else ""
        intent_instruction = (
            self._llm.classify_intent(query, previous_answer)
            if previous_answer else ""
        )

        # ── Build LangChain messages ───────────────────────────────────────────
        history_window = self._sessions.get_history(session_id, settings.HISTORY_WINDOW) if session_id else []
        messages = build_messages(
            query              = query,
            context            = context,
            history            = history_window,
            previous_answer    = previous_answer,
            intent_instruction = intent_instruction,
        )

        # ── Capture for closure (avoid self references in generator) ──────────
        _retrieved  = retrieved_chunks
        _source_pdf = selected_pdf
        _sources    = source_pdfs
        _citations  = citations
        _key        = cache_key
        _query_vec  = query_vec
        _st_model   = self._st_model
        _cache      = self._cache
        _sessions   = self._sessions
        _sid        = session_id
        _sq         = search_query

        def on_complete(full_text: str) -> None:
            """
            Called synchronously from the worker thread after streaming ends.
            Writes to memory, scores, and both cache layers — all using
            threading.Lock (safe from any thread).

            Guard: if full_text is empty (LLM error path), skip caching and
            session update to avoid poisoning the cache with blank entries.
            """
            if not full_text.strip():
                log.warning("on_complete called with empty text — skipping cache write")
                return

            if _sid:
                _sessions.add_ai_message(_sid, full_text)

            result = score_response(_sq, full_text, _retrieved, _st_model)
            scores.update(result)

            payload = {
                "text"      : full_text,
                "sources"   : _sources,
                "citations" : _citations,
                "scores"    : result,
            }
            _cache.set_exact(_key, payload, _source_pdf)
            _cache.set_semantic(_query_vec, payload, _source_pdf)

        token_gen = self._llm.stream(
            messages    = messages,
            is_followup = bool(previous_answer),
            on_complete = on_complete,
        )

        return token_gen, source_pdfs, citations, scores


# ── Private helpers ───────────────────────────────────────────────────────────

def _replay(text: str):
    """
    Yield a cached answer token-by-token, preserving all whitespace
    (including newlines) so the streamed output matches a fresh generation.
    Splits on whitespace boundaries while keeping the whitespace tokens.
    """
    import re
    tokens = re.split(r"(\s+)", text)
    for token in tokens:
        if token:          # skip empty strings produced by split at boundaries
            yield token


def _get_st_model(embeddings):
    """
    Robustly obtain the underlying SentenceTransformer from a
    HuggingFaceEmbeddings wrapper, regardless of langchain-huggingface version.
    Raises AttributeError with a clear message if neither attribute exists.
    """
    for attr in ("client", "_client", "model"):
        model = getattr(embeddings, attr, None)
        if model is not None:
            return model
    raise AttributeError(
        "Cannot find the underlying SentenceTransformer inside "
        f"{type(embeddings).__name__}. Known attributes: {list(vars(embeddings).keys())}"
    )
