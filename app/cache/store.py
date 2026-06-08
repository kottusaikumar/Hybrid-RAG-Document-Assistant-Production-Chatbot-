"""
app/cache/store.py
------------------
Thread-safe dual-layer cache:
  1. Exact cache  — keyed by MD5(session + context + query)
  2. Semantic cache — cosine similarity over stored query vectors

Both layers share a single threading.Lock (not asyncio.Lock) so they
are safe to write from a worker thread (inside scored_generator) and
to read from the async event loop.

Each entry carries a timestamp so stale entries can be evicted
(prevents serving answers from before a document update).
"""

import hashlib
import threading
import time
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings
from app.utils.logging import get_logger

log = get_logger(__name__)


class ResponseCache:
    """
    Dual-layer response cache.

    Exact cache  : O(1) dict lookup keyed by MD5 of (session, recent_ai, query).
    Semantic cache: linear scan over stored query vectors — fast enough for
                    SEMANTIC_CACHE_MAX ≤ 500 entries.
    """

    def __init__(self) -> None:
        self._lock      = threading.Lock()
        self._exact:    dict[str, dict] = {}
        self._semantic: list[dict]      = []      # each entry: {query_vec, payload, source, ts}

    # ── Exact cache ────────────────────────────────────────────────────────────

    def make_key(self, session_id: str, recent_ai: str, query: str) -> str:
        raw = f"{session_id}:{recent_ai[:100]}:{query.lower()}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get_exact(self, key: str) -> dict | None:
        with self._lock:
            entry = self._exact.get(key)
            if entry and not self._is_stale(entry):
                return entry["payload"]
            return None

    def set_exact(self, key: str, payload: dict, source: str) -> None:
        with self._lock:
            self._exact[key] = {"payload": payload, "source": source, "ts": time.time()}

    # ── Semantic cache ─────────────────────────────────────────────────────────

    def get_semantic(self, query_vec: np.ndarray, source: str | None) -> dict | None:
        # Snapshot eligible entries under the lock, then compute similarity
        # outside it so CPU-intensive work does not block concurrent writers.
        with self._lock:
            candidates = [
                e for e in self._semantic
                if (not source or e.get("source") == source)
                and not self._is_stale(e)
            ]

        for entry in candidates:
            sim = float(cosine_similarity(query_vec, entry["query_vec"])[0][0])
            if sim >= settings.SEMANTIC_CACHE_THRESHOLD:
                log.info("Semantic cache hit (sim=%.3f)", sim)
                return entry["payload"]
        return None

    def set_semantic(
        self, query_vec: np.ndarray, payload: dict, source: str | None
    ) -> None:
        with self._lock:
            self._semantic.append({
                "query_vec": query_vec,
                "payload"  : payload,
                "source"   : source,
                "ts"       : time.time(),
            })
            if len(self._semantic) > settings.SEMANTIC_CACHE_MAX:
                self._semantic.pop(0)

    # ── Bulk eviction (called alongside session eviction) ─────────────────────

    def evict_stale(self) -> None:
        with self._lock:
            self._semantic = [e for e in self._semantic if not self._is_stale(e)]
            stale_keys = [k for k, v in self._exact.items() if self._is_stale(v)]
            for k in stale_keys:
                del self._exact[k]


    # ── Helper ────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_stale(entry: dict) -> bool:
        return (time.time() - entry.get("ts", 0)) > settings.CACHE_TTL_SECONDS
