"""
app/llm/client.py
-----------------
Thin wrapper around Ollama for all LLM interactions:
  - Query preprocessing (fast model, non-streaming)
  - Follow-up intent classification (fast model, non-streaming)
  - Main answer generation (main model, streaming via LangChain)

The streaming generator runs in a daemon thread; tokens flow through
a thread-safe queue so the FastAPI route can yield NDJSON to the browser.
"""

import queue
import re
import threading

import requests
from langchain_ollama import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from app.config import settings
from app.utils.logging import get_logger

log = get_logger(__name__)


# ── Streaming callback ─────────────────────────────────────────────────────────

class _StreamHandler(BaseCallbackHandler):
    """Puts each streamed token into a thread-safe queue."""

    _DONE = object()  # sentinel

    def __init__(self) -> None:
        super().__init__()
        self._q        = queue.Queue()
        self.full_text = ""

    def on_llm_new_token(self, token: str, **_):
        self.full_text += token
        self._q.put(token)

    def on_llm_end(self, *_, **__):
        self._q.put(self._DONE)

    def on_llm_error(self, *_, **__):
        self._q.put(self._DONE)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is self._DONE:
                break
            yield item


# ── Main LLM client ────────────────────────────────────────────────────────────

class LLMClient:
    """
    Encapsulates all LLM calls.
    A new ChatOllama instance is created per streaming call so that
    callbacks are correctly bound to the handler for that request.
    """

    def __init__(self) -> None:
        log.info("Loading ChatOllama: %s", settings.LLM_MODEL)
        # Verify connectivity with a lightweight instance
        self._llm = ChatOllama(
            model       = settings.LLM_MODEL,
            base_url    = settings.OLLAMA_BASE_URL,
            temperature = settings.TEMPERATURE,
            num_ctx     = settings.NUM_CTX,
            num_predict = settings.MAX_TOKENS,
            streaming   = True,
        )
        log.info("ChatOllama ready")

    # ── Query preprocessing ────────────────────────────────────────────────────

    def preprocess_query(self, query: str, history: list) -> str:
        """
        Expand abbreviations and rewrite vague follow-ups into a standalone
        question using recent conversation history.
        Skips if the query already looks complete.
        Returns the (possibly rewritten) query string.
        """
        if not history:
            return query

        # Skip complete questions (≥ MIN_WORDS_COMPLETE_QUERY words ending with '?')
        words = query.strip().split()
        if query.strip().endswith("?") and len(words) >= settings.MIN_WORDS_COMPLETE_QUERY:
            log.debug("Preprocess skipped — complete question detected")
            return query

        history_text = "\n".join(
            f"{'USER' if isinstance(m, HumanMessage) else 'ASSISTANT'}: {m.content}"
            for m in history
        )

        result = self._ollama_call(
            model=settings.PREPROCESS_MODEL,
            system=(
                "You are a query preprocessor. Given a conversation history and a new user query, "
                "do the following in ONE step:\n\n"
                "1. EXPAND SHORTCUTS: If the query contains abbreviations, expand them ONLY if "
                "their full form appears in the history. NEVER guess — leave unknown shortcuts as-is.\n\n"
                "2. REWRITE FOLLOW-UPS: If the query is vague or incomplete, rewrite it as a "
                "complete standalone question using history context.\n\n"
                "3. LEAVE AS-IS: If the query is already complete, return it unchanged.\n\n"
                "IMPORTANT: Return ONLY the final query. No explanation, no quotes, no reasoning."
            ),
            user=f"Conversation history:\n{history_text}\n\nUser query: {query}",
            num_ctx=settings.PREPROCESS_CTX,
            timeout=settings.PREPROCESS_TIMEOUT,
        )

        if result and result.lower() != query.lower():
            log.info("Query preprocessed: '%s' -> '%s'", query, result)
            return result
        return query

    # ── Intent classification ──────────────────────────────────────────────────

    def classify_intent(self, query: str, previous_answer: str) -> str:
        """
        Returns the instruction string for the detected follow-up intent.
        Falls back to 'default' instruction on any failure.
        """
        label = self._ollama_call(
            model=settings.FOLLOWUP_INTENT_MODEL,
            system=settings.INTENT_CLASSIFIER_SYSTEM,
            user=(
                f"Previous answer (first 300 chars):\n{previous_answer[:300]}\n\n"
                f"User follow-up: {query}"
            ),
            num_ctx=settings.INTENT_CLASSIFIER_CTX,
            num_predict=settings.INTENT_CLASSIFIER_TOKENS,
            timeout=settings.FOLLOWUP_INTENT_TIMEOUT,
        )

        if label:
            label = re.sub(r"[^\w]", "", label.strip().lower())
            if label in settings.INTENT_INSTRUCTIONS:
                log.info("Follow-up intent: '%s'", label)
                return settings.INTENT_INSTRUCTIONS[label]
            log.warning("Unknown intent label '%s' — using default", label)

        return settings.INTENT_INSTRUCTIONS["default"]

    # ── Streaming generation ───────────────────────────────────────────────────

    def stream(
        self,
        messages: list,
        is_followup: bool,
        on_complete,
    ):
        """
        Returns a generator that yields string tokens.

        on_complete(full_text: str) is called synchronously from the worker
        thread after the last token has been yielded, so callers can write
        to caches and memory from within the same thread context.
        """
        temperature = (
            min(settings.TEMPERATURE + settings.FOLLOWUP_TEMPERATURE_BOOST,
                settings.FOLLOWUP_TEMPERATURE_MAX)
            if is_followup else settings.TEMPERATURE
        )

        handler = _StreamHandler()

        # Create a fresh ChatOllama instance per call with callbacks bound
        llm = ChatOllama(
            model       = settings.LLM_MODEL,
            base_url    = settings.OLLAMA_BASE_URL,
            temperature = temperature,
            num_ctx     = settings.NUM_CTX,
            num_predict = settings.MAX_TOKENS,
            streaming   = True,
            callbacks   = [handler],
        )

        def _run():
            try:
                llm.invoke(messages)
            except Exception as exc:
                log.error("LLM stream error: %s", exc)
                handler._q.put(handler._DONE)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        def _generate():
            for token in handler:
                yield token
            on_complete(handler.full_text)

        return _generate()

    # ── Shared Ollama helper ───────────────────────────────────────────────────

    def _ollama_call(
        self,
        model: str,
        system: str,
        user: str,
        num_ctx: int = 512,
        num_predict: int = 64,
        timeout: int = 60,
    ) -> str | None:
        payload = {
            "model"   : model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "options": {
                "temperature": 0.0,
                "num_ctx"    : num_ctx,
                "num_predict": num_predict,
            },
            "stream": False,
        }
        try:
            resp = requests.post(settings.OLLAMA_CHAT_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except requests.exceptions.Timeout:
            log.warning("Ollama call timed out (model=%s)", model)
        except Exception as exc:
            log.warning("Ollama call failed (model=%s): %s", model, exc)
        return None