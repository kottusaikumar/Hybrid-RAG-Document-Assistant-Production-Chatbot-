"""
app/session/manager.py
----------------------
Per-session state management.

Uses threading.Lock (not asyncio.Lock) for every session so state is safe
to read and write from both async route handlers and sync worker threads
(e.g. inside scored_generator).

Stores per session:
  - LangChain ConversationBufferMemory
  - Last retrieved chunks + source PDF (for follow-up reuse)
  - Topic anchor vector (for topic-shift detection)
  - Last-active timestamp (for TTL eviction)
"""

import threading
import time

import numpy as np
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings
from app.utils.logging import get_logger

log = get_logger(__name__)


class SessionState:
    """All mutable state for a single chat session."""

    __slots__ = (
        "memory", "chunks", "source",
        "topic_query", "topic_vec", "last_active", "lock",
    )

    def __init__(self) -> None:
        self.memory: ConversationBufferMemory = ConversationBufferMemory(
            return_messages=True, memory_key="history", input_key="input"
        )
        self.chunks:      list[dict]        = []
        self.source:      str | None        = None
        self.topic_query: str               = ""
        self.topic_vec:   np.ndarray | None = None
        self.last_active: float             = time.time()
        self.lock:        threading.Lock    = threading.Lock()


class SessionManager:
    """
    Central registry for all SessionState objects.

    The registry dict itself is protected by a module-level RLock
    to avoid TOCTOU races when creating or evicting sessions.
    Sessions are created lazily on first access.
    """

    def __init__(self) -> None:
        self._registry: dict[str, SessionState] = {}
        self._rlock    = threading.RLock()

    # ── Session access ────────────────────────────────────────────────────────

    def get(self, session_id: str) -> SessionState:
        """Return (lazily creating) the SessionState for session_id."""
        with self._rlock:
            if session_id not in self._registry:
                self._registry[session_id] = SessionState()
            return self._registry[session_id]

    def touch(self, session_id: str) -> None:
        """Update last_active timestamp."""
        state = self.get(session_id)
        with state.lock:
            state.last_active = time.time()

    # ── Memory helpers ────────────────────────────────────────────────────────

    def add_user_message(self, session_id: str, text: str) -> None:
        state = self.get(session_id)
        with state.lock:
            state.memory.chat_memory.add_user_message(text)

    def add_ai_message(self, session_id: str, text: str) -> None:
        state = self.get(session_id)
        with state.lock:
            state.memory.chat_memory.add_ai_message(text)

    def get_history(self, session_id: str, window: int) -> list:
        """Return up to `window` recent messages (thread-safe snapshot)."""
        state = self.get(session_id)
        with state.lock:
            return list(state.memory.chat_memory.messages[-window:])

    def get_last_ai_message(self, session_id: str) -> str:
        """Return the most recent AI response text, or empty string."""
        state = self.get(session_id)
        with state.lock:
            for msg in reversed(state.memory.chat_memory.messages):
                if isinstance(msg, AIMessage):
                    return msg.content
        return ""

    def has_ai_message(self, session_id: str) -> bool:
        state = self.get(session_id)
        with state.lock:
            return any(isinstance(m, AIMessage) for m in state.memory.chat_memory.messages)

    # ── Follow-up / topic detection ────────────────────────────────────────────

    def classify_query(
        self, session_id: str, query_vec: np.ndarray
    ) -> tuple[bool, list[dict], str | None]:
        """
        Decide whether this query is a follow-up or a topic shift.

        Returns (is_followup, cached_chunks, cached_source).
        If is_followup is False, the caller should do fresh retrieval and
        then call update_topic().
        """
        state = self.get(session_id)
        with state.lock:
            has_chunks = bool(state.chunks)
            has_ai     = any(isinstance(m, AIMessage) for m in state.memory.chat_memory.messages)

            if not has_chunks or not has_ai:
                return False, [], None

            if state.topic_vec is None:
                return False, [], None

            sim = float(cosine_similarity(query_vec, state.topic_vec)[0][0])
            log.info("Topic similarity: %.3f (threshold=%.2f)", sim, settings.TOPIC_SHIFT_THRESHOLD)

            if sim < settings.TOPIC_SHIFT_THRESHOLD:
                log.info("Topic shift detected — clearing cached chunks")
                state.chunks = []
                state.source = None
                return False, [], None

            return True, list(state.chunks), state.source

    def update_topic(
        self,
        session_id: str,
        query: str,
        query_vec: np.ndarray,
        chunks: list[dict],
        source: str,
    ) -> None:
        """Store fresh retrieval results as the new topic anchor."""
        state = self.get(session_id)
        with state.lock:
            state.topic_query = query
            state.topic_vec   = query_vec
            state.chunks      = chunks
            state.source      = source

    # ── TTL eviction ──────────────────────────────────────────────────────────

    def evict_stale(self) -> None:
        """Remove sessions idle longer than SESSION_TTL_SECONDS."""
        now = time.time()
        # Snapshot (id, last_active) pairs — each last_active is read under the
        # session's own lock to avoid a TOCTOU with concurrent touch() calls.
        with self._rlock:
            snapshot = list(self._registry.items())

        stale = []
        for sid, s in snapshot:
            with s.lock:
                idle = now - s.last_active
            if idle > settings.SESSION_TTL_SECONDS:
                stale.append(sid)

        if stale:
            with self._rlock:
                for sid in stale:
                    self._registry.pop(sid, None)
            log.info("Evicted %d stale session(s)", len(stale))
