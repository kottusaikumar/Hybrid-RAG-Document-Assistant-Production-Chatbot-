"""
app/retrieval/chunker.py
------------------------
Semantic small-to-big chunking with fixed-size fallback.
Accepts plain dicts; returns plain dicts (no LangChain dependency here).
All tunables come from config.
"""

import re
from app.config import settings
from app.utils.logging import get_logger

log = get_logger(__name__)


def chunk_document(doc: dict) -> list[dict]:
    """
    Split a document dict into child chunks with parent context.

    doc keys expected:
        filename : str
        size_mb  : float
        text     : str
        words    : int  (informational only)

    Returns a list of chunk dicts, each with:
        text, parent_text, source, size_mb, chunk_id, parent_id
    """
    text       = doc["text"]
    filename   = doc["filename"]
    size_mb    = doc["size_mb"]

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if len(p.strip()) >= settings.MIN_PARA_LEN
    ]

    chunks: list[dict] = []
    chunk_id = 0

    for p_idx, paragraph in enumerate(paragraphs):
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", paragraph)
            if len(s.strip()) >= settings.MIN_SENT_LEN
        ]

        current_child: list[str] = []
        current_words = 0

        for sentence in sentences:
            word_count = len(sentence.split())
            if current_words + word_count > settings.WORDS_PER_CHILD_CHUNK and current_child:
                chunks.append(_make_chunk(
                    " ".join(current_child), paragraph,
                    filename, size_mb, chunk_id, p_idx
                ))
                chunk_id    += 1
                current_child = [sentence]
                current_words = word_count
            else:
                current_child.append(sentence)
                current_words += word_count

        if current_child:
            chunks.append(_make_chunk(
                " ".join(current_child), paragraph,
                filename, size_mb, chunk_id, p_idx
            ))
            chunk_id += 1

    # Fallback: fixed-size word windows when semantic split yields nothing
    if not chunks:
        log.warning("%s: semantic split produced 0 chunks — using fixed-size fallback", filename)
        words = text.split()
        for start in range(0, len(words), settings.CHUNK_SIZE - settings.CHUNK_OVERLAP):
            end        = min(start + settings.CHUNK_SIZE, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(_make_chunk(
                chunk_text, chunk_text,
                filename, size_mb, len(chunks), len(chunks)
            ))

    return chunks


def _make_chunk(
    text: str, parent_text: str,
    source: str, size_mb: float,
    chunk_id: int, parent_id: int
) -> dict:
    return {
        "text"       : text,
        "parent_text": parent_text,
        "source"     : source,
        "size_mb"    : size_mb,
        "chunk_id"   : chunk_id,
        "parent_id"  : parent_id,
    }
