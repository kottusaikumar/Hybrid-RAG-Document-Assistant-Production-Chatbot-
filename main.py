"""
main.py
-------
FastAPI application entry point.

Run:
    python main.py
    uvicorn main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.chatbot import HybridRAGChatbot
from app.config import settings
from app.routes import router
from app.utils.logging import get_logger

log     = get_logger("main")
BASE_DIR = Path(__file__).parent


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=" * 55)
    log.info("       STARTING CHATBOT API SERVER")
    log.info("=" * 55)

    _evict_task = None
    try:
        app.state.chatbot = HybridRAGChatbot()
        log.info("Chatbot loaded successfully")

        # Background eviction loop — runs every 5 minutes so per-request
        # eviction is no longer needed (avoids O(n) work on every call).
        async def _evict_loop():
            while True:
                await asyncio.sleep(300)   # 5 minutes
                try:
                    app.state.chatbot._sessions.evict_stale()
                    app.state.chatbot._cache.evict_stale()
                except Exception as exc:
                    log.warning("Eviction loop error: %s", exc)

        _evict_task = asyncio.create_task(_evict_loop())
        log.info("Background eviction task started (interval=300s)")

    except FileNotFoundError as exc:
        log.error("Index files not found: %s — run: python build_index.py", exc)
    except Exception as exc:
        log.error("Failed to load chatbot: %s", exc)

    log.info("API  : http://%s:%d", settings.HOST, settings.PORT)
    log.info("UI   : http://localhost:%d/", settings.PORT)
    log.info("Docs : http://localhost:%d/docs", settings.PORT)
    log.info("=" * 55)

    yield

    log.info("Shutting down...")
    if _evict_task is not None:
        _evict_task.cancel()
        try:
            await _evict_task
        except asyncio.CancelledError:
            pass
    app.state.chatbot = None


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Document Assistant API",
    description = (
        "Hybrid RAG chatbot: FAISS + BM25 + RRF retrieval, "
        "CrossEncoder reranking, streaming answers via local LLM."
    ),
    version     = "3.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

app.include_router(router, prefix="/api/v1", tags=["Chatbot"])


@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


# ── Dev runner ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)
