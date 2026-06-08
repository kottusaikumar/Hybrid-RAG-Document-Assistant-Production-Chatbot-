"""
app/retrieval/store.py
----------------------
Thread-safe document store wrapping:
  - FAISS HNSW index (dense retrieval)
  - BM25 per PDF (sparse retrieval)
  - RRF fusion
  - CrossEncoder reranking
  - Live document addition

All index mutations are protected by a single threading.RLock so they
are safe to call from both sync threads and async event-loop code.
"""

import re
import threading
import numpy as np
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers.cross_encoder import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain.schema import Document

from app.config import settings
from app.utils.logging import get_logger
from app.retrieval.chunker import chunk_document
from app.retrieval.indexer import build_hnsw_index, save_index
from app.utils.pdf import pdf_to_text

log = get_logger(__name__)


class DocumentStore:
    """
    Central store for all retrieval state.
    Thread-safe for concurrent reads and serialised writes.
    """

    def __init__(
        self,
        faiss_index_path: str,
        chunk_store_path: str,
        embedding_model: HuggingFaceEmbeddings,
    ) -> None:
        self._lock   = threading.RLock()
        self._lc_lock = threading.Lock()   # serialises LangChain vectorstore I/O

        log.info("Loading FAISS index from %s", faiss_index_path)
        self._index  = faiss.read_index(faiss_index_path)
        log.info("FAISS index loaded (%d vectors)", self._index.ntotal)

        log.info("Loading chunk store from %s", chunk_store_path)
        self._chunks: list[dict] = np.load(chunk_store_path, allow_pickle=True).tolist()
        log.info("%d chunks loaded", len(self._chunks))

        if len(self._chunks) != self._index.ntotal:
            raise ValueError(
                f"Mismatch: {len(self._chunks)} chunks but {self._index.ntotal} FAISS vectors. "
                "Run: python indexer.py"
            )

        self._embeddings = embedding_model

        log.info("Loading CrossEncoder reranker: %s", settings.RERANKER_MODEL)
        self._reranker = CrossEncoder(settings.RERANKER_MODEL)

        # Derived indices (built from _chunks; rebuilt on add_document)
        self._pdf_names:     list[str]       = []
        self._chunk_pos:     dict[str, list[int]] = {}  # pdf -> global positions
        self._bm25:          dict[str, BM25Okapi]  = {}

        self._rebuild_derived()
        log.info("DocumentStore ready (%d PDFs)", len(self._pdf_names))

    # ── Public read properties ────────────────────────────────────────────────

    @property
    def pdf_names(self) -> list[str]:
        with self._lock:
            return list(self._pdf_names)

    @property
    def chunk_count(self) -> int:
        with self._lock:
            return len(self._chunks)

    # ── PDF detection ─────────────────────────────────────────────────────────

    def detect_pdf(self, query_vec: np.ndarray) -> str | None:
        """
        Return the PDF whose chunks are most similar to query_vec on average,
        or None if no PDF exceeds PDF_SCORE_THRESHOLD.
        """
        with self._lock:
            pdf_names = list(self._pdf_names)
            chunks    = self._chunks
            index     = self._index

        pool = settings.PDF_DETECT_POOL
        scores, positions = index.search(query_vec, pool)

        pdf_scores: dict[str, list[float]] = {p: [] for p in pdf_names}
        for score, pos in zip(scores[0], positions[0]):
            if 0 <= pos < len(chunks):
                src = chunks[pos].get("source", "")
                if src in pdf_scores:
                    pdf_scores[src].append(float(score))

        avg = {
            pdf: (sum(s) / len(s) if s else 0.0)
            for pdf, s in pdf_scores.items()
        }
        best = max(avg, key=avg.get, default=None)
        if best is None or avg[best] < settings.PDF_SCORE_THRESHOLD:
            log.warning("No PDF passed score threshold (%.2f)", settings.PDF_SCORE_THRESHOLD)
            return None

        log.info("Selected PDF: %s (avg_score=%.3f)", best, avg[best])
        return best

    # ── Hybrid search ─────────────────────────────────────────────────────────

    def search(self, query: str, pdf: str, query_vec: np.ndarray) -> list[dict]:
        """
        Hybrid FAISS + BM25 + RRF search restricted to `pdf`.
        Returns up to RRF_K unique-parent chunks before reranking.
        """
        with self._lock:
            pdf_positions    = list(self._chunk_pos.get(pdf, []))
            chunks           = self._chunks
            index            = self._index
            bm25             = self._bm25.get(pdf)

        if not pdf_positions:
            return []

        pdf_pos_set = set(pdf_positions)

        # FAISS
        search_k = min(len(chunks), max(200, len(pdf_positions) * 3))
        scores, positions = index.search(query_vec, search_k)
        faiss_hits: dict[int, float] = {}
        for score, pos in zip(scores[0], positions[0]):
            if pos in pdf_pos_set:
                faiss_hits[pos] = float(score)
                if len(faiss_hits) == settings.FAISS_K:
                    break

        # BM25
        bm25_hits: dict[int, float] = {}
        if bm25:
            tokens      = _tokenize(query)
            bm25_scores = bm25.get_scores(tokens)
            top_local   = np.argsort(bm25_scores)[::-1][: settings.BM25_K]
            for local_pos in top_local:
                global_pos          = pdf_positions[local_pos]
                bm25_hits[global_pos] = float(bm25_scores[local_pos])

        # RRF fusion
        rrf: dict[int, float] = {}
        k = settings.RRF_CONSTANT
        for rank, pos in enumerate(sorted(faiss_hits, key=faiss_hits.get, reverse=True)):
            rrf[pos] = rrf.get(pos, 0.0) + 1 / (k + rank)
        for rank, pos in enumerate(sorted(bm25_hits, key=bm25_hits.get, reverse=True)):
            rrf[pos] = rrf.get(pos, 0.0) + 1 / (k + rank)

        # Deduplicate by parent_id; collect up to RRF_K unique parents
        retrieved: list[dict] = []
        seen_parents: set     = set()
        for pos in sorted(rrf, key=rrf.get, reverse=True):
            chunk     = chunks[pos]
            parent_id = chunk.get("parent_id", chunk.get("chunk_id", pos))
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                retrieved.append({
                    "text"    : chunk.get("parent_text", chunk["text"]),
                    "source"  : chunk["source"],
                    "chunk_id": parent_id,
                    "parent_embedding": chunk.get("parent_embedding"),
                })
            if len(retrieved) == settings.RRF_K:
                break

        log.info("Hybrid search: %d unique parent candidates from '%s'", len(retrieved), pdf)
        return retrieved

    # ── CrossEncoder reranking ────────────────────────────────────────────────

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        """Rerank with CrossEncoder; filter by RERANKER_THRESHOLD; return top FINAL_K."""
        if not chunks:
            return []

        pairs  = [(query, c["text"]) for c in chunks]
        scores = self._reranker.predict(pairs)

        ranked   = sorted(zip(scores, chunks), key=lambda x: float(x[0]), reverse=True)
        filtered = [(s, c) for s, c in ranked if s > settings.RERANKER_THRESHOLD]

        if not filtered:
            log.warning("All %d chunks filtered by reranker threshold %.1f",
                        len(chunks), settings.RERANKER_THRESHOLD)
            return []

        top = [c for _, c in filtered[: settings.FINAL_K]]
        log.info("Reranked %d -> %d chunks", len(chunks), len(top))
        return top

    # ── Live document addition ────────────────────────────────────────────────

    def add_document(self, pdf_path: str, filename: str) -> bool:
        """
        Chunk, embed, and add a new PDF to the live index.
        Entire mutation is serialised under self._lock.
        Returns True on success.

        Re-upload behaviour: FAISS HNSW does not support vector deletion, so
        old vectors remain in the index but become unreachable — their chunk
        metadata entries are tombstoned (source set to "") so no search path
        can surface them again.  chunk_pos is reset to the fresh positions only.
        """
        import os
        log.info("Reindexing: %s", filename)

        text    = pdf_to_text(pdf_path)
        size_mb = round(os.path.getsize(pdf_path) / (1024 * 1024), 1)
        doc     = {
            "filename": filename,
            "size_mb" : size_mb,
            "text"    : text,
            "words"   : len(text.split()),
        }

        new_chunks = chunk_document(doc)
        if not new_chunks:
            log.error("No chunks produced for %s", filename)
            return False

        texts   = [c["text"] for c in new_chunks]
        vectors = np.array(self._embeddings.embed_documents(texts), dtype=np.float32)
        faiss.normalize_L2(vectors)

        # Precompute parent embeddings for new chunks to avoid re-embedding at query time
        unique_parents = list(set(c["parent_text"] for c in new_chunks))
        parent_vecs = self._embeddings.embed_documents(unique_parents)
        parent_to_vec = {text: vec for text, vec in zip(unique_parents, parent_vecs)}
        for c in new_chunks:
            c["parent_embedding"] = parent_to_vec[c["parent_text"]]

        with self._lock:
            # Tombstone old chunks for this PDF so their FAISS positions become
            # unreachable (source="" never matches any pdf_pos_set).
            if filename in self._pdf_names:
                old_positions = self._chunk_pos.get(filename, [])
                if old_positions:
                    tombstone = {
                        "source": "", "text": "", "parent_text": "",
                        "size_mb": 0, "chunk_id": -1, "parent_id": -1,
                        "parent_embedding": None,
                    }
                    for pos in old_positions:
                        self._chunks[pos] = dict(tombstone, chunk_id=pos)
                    log.info("Tombstoned %d old chunk(s) for '%s'", len(old_positions), filename)

            start_idx = len(self._chunks)

            for i, c in enumerate(new_chunks):
                c["chunk_id"] = start_idx + i

            self._index.add(vectors)
            self._chunks.extend(new_chunks)

            if filename not in self._pdf_names:
                self._pdf_names.append(filename)

            # Reset to new positions only — do NOT merge with old to avoid duplicates
            new_positions             = list(range(start_idx, start_idx + len(new_chunks)))
            self._chunk_pos[filename] = new_positions

            # Rebuild BM25 over new chunks only
            new_texts            = [_tokenize_str(self._chunks[p]["text"]) for p in new_positions]
            self._bm25[filename] = BM25Okapi(new_texts)

            # Persist
            faiss.write_index(self._index, settings.FAISS_INDEX_PATH)
            np.save(settings.CHUNK_STORE_PATH, self._chunks)

        # Sync LangChain vectorstore (outside the main lock — I/O is slow)
        self._sync_lc_store(new_chunks)

        log.info("Indexed %s (%d chunks)", filename, len(new_chunks))
        return True

    # ── Private helpers ────────────────────────────────────────────────────────

    def _rebuild_derived(self) -> None:
        """Rebuild pdf_names, chunk_pos, and bm25 from self._chunks. Call under lock."""
        with self._lock:
            self._pdf_names = sorted({c.get("source", "") for c in self._chunks})
            self._chunk_pos = {pdf: [] for pdf in self._pdf_names}
            for pos, c in enumerate(self._chunks):
                src = c.get("source", "")
                if src in self._chunk_pos:
                    self._chunk_pos[src].append(pos)

            self._bm25 = {}
            for pdf, positions in self._chunk_pos.items():
                if positions:
                    tokenized      = [_tokenize_str(self._chunks[p]["text"]) for p in positions]
                    self._bm25[pdf] = BM25Okapi(tokenized)
                    log.info("BM25 built for '%s' (%d chunks)", pdf, len(positions))
                else:
                    log.warning("'%s' has 0 chunks — BM25 skipped", pdf)

    def _sync_lc_store(self, new_chunks: list[dict]) -> None:
        """Update the LangChain vectorstore on disk. Non-critical — log but don't raise.

        Uses a dedicated lock so concurrent add_document calls don't corrupt
        the on-disk store (the main _lock is intentionally released before this
        slow I/O operation).
        """
        with self._lc_lock:
            try:
                lc_docs = [
                    Document(
                        page_content=c["text"],
                        metadata={k: c[k] for k in ("parent_text", "source", "size_mb", "chunk_id", "parent_id")},
                    )
                    for c in new_chunks
                ]
                lc_path = settings.FAISS_LC_PATH
                try:
                    store = LangchainFAISS.load_local(lc_path, self._embeddings, allow_dangerous_deserialization=True)
                    store.add_documents(lc_docs)
                except Exception:
                    store = LangchainFAISS.from_documents(lc_docs, self._embeddings)
                store.save_local(lc_path)
                log.info("LangChain vectorstore synced -> %s", lc_path)
            except Exception as exc:
                log.warning("LangChain vectorstore sync failed: %s", exc)


# ── Module-level tokenize helpers ─────────────────────────────────────────────

def _tokenize(query: str) -> list[str]:
    """Tokenize for BM25 queries: lowercase + strip punctuation."""
    return re.sub(r"[^\w\s]", "", query.lower()).split()


def _tokenize_str(text: str) -> list[str]:
    """Tokenize for BM25 index building — same logic for consistency."""
    return _tokenize(text)
