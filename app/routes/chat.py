"""
app/routes/chat.py
------------------
FastAPI routes for the chatbot API.

Streaming contract
------------------
scored_generator (inside chatbot.get_response) mutates the `scores` dict
only AFTER yielding the last token. The event_stream loop below reads
scores AFTER exhausting the generator — this ordering is guaranteed.
"""

import json
import os
import time
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings

router = APIRouter()

MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "50"))   # configurable upload cap


# ── Pydantic models ────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question  : str = Field(..., min_length=3, max_length=1000)
    session_id: str | None = Field(default=None)


class FeedbackRequest(BaseModel):
    session_id   : str | None = None
    question     : str
    feedback_type: str = Field(..., description="'thumbs_up' or 'thumbs_down'")


class HealthResponse(BaseModel):
    status : str
    message: str


# ── Answer (streaming) ────────────────────────────────────────────────────────

@router.post("/answer", summary="Ask a question")
async def get_answer(req: Request, request: QuestionRequest):
    chatbot = _get_chatbot(req)
    start   = time.time()

    answer_gen, sources, citations, scores = chatbot.get_response(
        request.question, request.session_id
    )
    source_pdf = sources[0] if sources else "Not found in any PDF"

    async def event_stream():
        # Error string (not a generator)
        if isinstance(answer_gen, str):
            elapsed = round(time.time() - start, 2)
            yield _ndjson({"type": "chunk",    "content": answer_gen})
            yield _ndjson({"type": "metadata", "source_pdf": source_pdf,
                           "citations": citations, "scores": scores,
                           "grounding_warning": scores.get("grounding_warning", False),
                           "response_time_sec": elapsed})
            return

        yield _ndjson({"type": "status", "message": f"Searching {source_pdf}... [OK]"})

        if citations:
            yield _ndjson({"type": "early_citations", "citations": citations})

        # Drain the generator — scores dict is populated on the last iteration
        for chunk in answer_gen:
            yield _ndjson({"type": "chunk", "content": chunk})

        # scores is now fully populated by on_complete()
        elapsed = round(time.time() - start, 2)
        yield _ndjson({
            "type"             : "metadata",
            "source_pdf"       : source_pdf,
            "citations"        : citations,
            "scores"           : scores,
            "grounding_warning": scores.get("grounding_warning", False),
            "response_time_sec": elapsed,
        })

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check():
    return HealthResponse(status="ok", message="Chatbot API is running")


# ── Info ───────────────────────────────────────────────────────────────────────

@router.get("/info", summary="Chatbot info")
async def info(req: Request):
    chatbot = _get_chatbot(req)
    return {
        "pdfs"        : chatbot.pdf_names,
        "total_chunks": chatbot.chunk_count,
        "model"       : settings.LLM_MODEL,
        "reranker"    : settings.RERANKER_MODEL,
    }


# ── Feedback ──────────────────────────────────────────────────────────────────

@router.post("/feedback", summary="Submit user feedback")
async def submit_feedback(request: FeedbackRequest):
    try:
        import app.database as db
        db.log_feedback(request.session_id, request.question, request.feedback_type)
        return {"status": "success"}
    except (ModuleNotFoundError, ImportError):
        # database.py is optional — silently ignore if not present or incomplete
        return {"status": "skipped", "detail": "Feedback logging not configured"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to log feedback: {exc}")


# ── Reindex (upload PDF) ──────────────────────────────────────────────────────

@router.post("/reindex", summary="Upload and index a new PDF")
async def reindex_pdf(req: Request, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Read once so we can size-check before writing
    content  = await file.read()
    size_mb  = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large ({size_mb:.1f} MB). "
                f"Maximum allowed size is {MAX_UPLOAD_MB:.0f} MB."
            ),
        )

    chatbot       = _get_chatbot(req)
    safe_filename = Path(file.filename).name   # strip any directory components
    pdf_path      = os.path.join(settings.PDF_FOLDER, safe_filename)
    os.makedirs(settings.PDF_FOLDER, exist_ok=True)

    async with aiofiles.open(pdf_path, "wb") as out:
        await out.write(content)

    success = chatbot.add_document(pdf_path, safe_filename)
    if success:
        return {"status": "success", "message": f"Successfully indexed {safe_filename}"}
    raise HTTPException(status_code=500, detail="Failed to chunk document.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_chatbot(req: Request):
    chatbot = getattr(req.app.state, "chatbot", None)
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot is not initialised. Try again shortly.")
    return chatbot


def _ndjson(obj: dict) -> str:
    return json.dumps(obj) + "\n"
