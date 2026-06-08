"""
build_index.py
--------------
CLI tool to process all PDFs in the pdfs/ folder and build the FAISS index.

Run once before starting the server:
    python build_index.py
"""

import os
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.retrieval.chunker import chunk_document
from app.retrieval.indexer import build_hnsw_index, save_index
from app.utils.pdf import pdf_to_text
from app.utils.logging import get_logger

log = get_logger("indexer")


def load_pdfs(folder: str) -> list[dict]:
    folder_path = Path(folder)
    pdf_files   = sorted(folder_path.glob("*.pdf"))

    if not pdf_files:
        log.error("No PDFs found in %s", folder)
        return []

    documents = []
    for pdf_path in pdf_files:
        size_mb = round(pdf_path.stat().st_size / (1024 * 1024), 1)
        log.info("Extracting: %s (%.1f MB)", pdf_path.name, size_mb)
        try:
            text = pdf_to_text(pdf_path)
            documents.append({
                "filename": pdf_path.name,
                "size_mb" : size_mb,
                "text"    : text,
                "words"   : len(text.split()),
            })
            log.info("  -> %d words extracted", documents[-1]["words"])
        except Exception as exc:
            log.error("  -> Failed: %s", exc)

    return documents


def chunk_all(documents: list[dict]) -> list[dict]:
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        log.info("Chunked %s -> %d chunks", doc["filename"], len(chunks))
    log.info("Total: %d chunks across %d PDFs", len(all_chunks), len(documents))
    return all_chunks


def build_index():
    log.info("=" * 55)
    log.info("         INDEXING PIPELINE STARTED")
    log.info("=" * 55)

    os.makedirs(settings.PDF_FOLDER, exist_ok=True)

    log.info("[1/3] Loading PDFs from %s", settings.PDF_FOLDER)
    documents = load_pdfs(settings.PDF_FOLDER)
    if not documents:
        return

    log.info("[2/3] Chunking (size=%d, overlap=%d)", settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    all_chunks = chunk_all(documents)

    log.info("[3/3] Embedding + building HNSW index (%s)", settings.RETRIEVER_MODEL)
    embedding_model = HuggingFaceEmbeddings(
        model_name    = settings.RETRIEVER_MODEL,
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
    )
    lc_store, _, _ = build_hnsw_index(all_chunks, embedding_model)

    save_index(lc_store, all_chunks)

    log.info("=" * 55)
    log.info("         INDEXING COMPLETE")
    log.info("  PDFs   : %d", len(documents))
    log.info("  Chunks : %d", len(all_chunks))
    log.info("  Run    : python main.py")
    log.info("=" * 55)


if __name__ == "__main__":
    build_index()
