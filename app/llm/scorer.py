"""
app/llm/scorer.py
-----------------
Computes faithfulness and relevance scores after generation.
Uses the underlying SentenceTransformer model directly (no re-embedding).
"""

import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings
from app.utils.logging import get_logger

log = get_logger(__name__)


def score_response(
    query: str,
    answer: str,
    retrieved_chunks: list[dict],
    st_model,                       # raw SentenceTransformer instance
) -> dict:
    """
    Returns:
        faithfulness (float | None) : cosine sim of answer vs best chunk
        relevance    (float | None) : cosine sim of query vs answer
        confidence   (int  | None)  : mean of both, scaled 0-100
        grounding_warning (bool)    : True if faithfulness < threshold
    """
    if not answer or not retrieved_chunks:
        return _empty()

    try:
        answer_vec  = st_model.encode([answer])
        query_vec   = st_model.encode([query])

        # Reuse precomputed parent embeddings if they exist (fast!), otherwise encode dynamically (fallback)
        has_precomputed = all(c.get("parent_embedding") is not None for c in retrieved_chunks)
        if has_precomputed:
            chunk_vecs = np.array([c["parent_embedding"] for c in retrieved_chunks], dtype=np.float32)
            # Ensure the vectors are normalized (SentenceTransformers encode normalizes, but let's be safe)
            faiss.normalize_L2(chunk_vecs)
        else:
            chunk_vecs = st_model.encode([c["text"] for c in retrieved_chunks])

        faithfulness = float(np.clip(cosine_similarity(answer_vec, chunk_vecs)[0].max(), 0.0, 1.0))
        relevance    = float(np.clip(cosine_similarity(query_vec, answer_vec)[0][0], 0.0, 1.0))
        confidence   = round(((faithfulness + relevance) / 2) * 100)

        grounding_warning = faithfulness < settings.GROUNDING_WARNING_THRESHOLD

        log.info(
            "Score — faithfulness=%.2f relevance=%.2f confidence=%d%%%s",
            faithfulness, relevance, confidence,
            " [LOW FAITHFULNESS]" if grounding_warning else "",
        )
        return {
            "faithfulness"    : faithfulness,
            "relevance"       : relevance,
            "confidence"      : confidence,
            "grounding_warning": grounding_warning,
        }

    except Exception as exc:
        log.warning("Scoring failed: %s", exc)
        return _empty()


def _empty() -> dict:
    return {
        "faithfulness"    : None,
        "relevance"       : None,
        "confidence"      : None,
        "grounding_warning": False,
    }
