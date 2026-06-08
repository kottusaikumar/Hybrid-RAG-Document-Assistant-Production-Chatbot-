"""
app/retrieval/indexer.py
------------------------
Builds the FAISS HNSW index from a list of chunk dicts.
Called once from the CLI (indexer.py) and during live reindexing
(add_document in store.py).
All HNSW parameters come from config.
"""

import numpy as np
import faiss
from pathlib import Path

from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.utils.logging import get_logger

log = get_logger(__name__)


def build_hnsw_index(
    chunks: list[dict],
    embedding_model: HuggingFaceEmbeddings,
) -> tuple[LangchainFAISS, faiss.Index, np.ndarray]:
    """
    Embed all chunks, build a LangChain FAISS vectorstore backed by HNSW,
    and return (lc_vectorstore, raw_index, vectors).

    The raw index and vectors are returned so callers can save them
    separately for fast numpy reconstruction during add_document.
    """
    # Precompute unique parent embeddings to bypass slow CPU re-encoding during scoring
    log.info("Precomputing parent embeddings for %d chunks...", len(chunks))
    unique_parents = list(set(c["parent_text"] for c in chunks))
    log.info("Found %d unique parent paragraphs to embed", len(unique_parents))
    parent_vecs = embedding_model.embed_documents(unique_parents)
    parent_to_vec = {text: vec for text, vec in zip(unique_parents, parent_vecs)}
    for c in chunks:
        c["parent_embedding"] = parent_to_vec[c["parent_text"]]

    lc_docs = _chunks_to_lc_docs(chunks)
    lc_store = LangchainFAISS.from_documents(lc_docs, embedding_model)

    flat_index = lc_store.index
    if not hasattr(flat_index, "reconstruct_n"):
        raise ValueError(
            f"FAISS index type {type(flat_index).__name__} does not support "
            "reconstruct_n. Only IndexFlatL2 / IndexFlatIP is supported as the "
            "initial flat index. Re-run build_index.py with a compatible index type."
        )

    n, dim = flat_index.ntotal, flat_index.d
    vectors = np.zeros((n, dim), dtype=np.float32)
    flat_index.reconstruct_n(0, n, vectors)
    faiss.normalize_L2(vectors)

    hnsw = faiss.IndexHNSWFlat(dim, settings.HNSW_M)
    hnsw.hnsw.efConstruction = settings.HNSW_EF_CONSTRUCTION
    hnsw.hnsw.efSearch       = settings.HNSW_EF_SEARCH
    hnsw.add(vectors)

    lc_store.index = hnsw
    log.info("HNSW index built: %d vectors (dim=%d)", hnsw.ntotal, dim)
    return lc_store, hnsw, vectors


def save_index(
    lc_store: LangchainFAISS,
    chunks: list[dict],
    faiss_path: str  = settings.FAISS_INDEX_PATH,
    lc_path: str     = settings.FAISS_LC_PATH,
    chunk_path: str  = settings.CHUNK_STORE_PATH,
) -> None:
    """Persist raw FAISS index, LangChain vectorstore, and chunk store."""
    lc_store.save_local(lc_path)
    faiss.write_index(lc_store.index, faiss_path)
    np.save(chunk_path, chunks)
    log.info("Saved: %s | %s | %s", faiss_path, lc_path, chunk_path)


def _chunks_to_lc_docs(chunks: list[dict]) -> list[Document]:
    return [
        Document(
            page_content=c["text"],
            metadata={
                "parent_text": c["parent_text"],
                "source"     : c["source"],
                "size_mb"    : c["size_mb"],
                "chunk_id"   : c["chunk_id"],
                "parent_id"  : c["parent_id"],
            },
        )
        for c in chunks
    ]
