"""
app/database.py
---------------
Optional feedback-logging backend.

The /api/v1/feedback route calls log_feedback() from this module.
By default it is a no-op so the app works without any database setup.

To enable persistent feedback storage, replace the body of log_feedback()
with your preferred backend (SQLite, PostgreSQL, a flat file, etc.).

Example — append to a JSONL file:
    import json, time
    with open("feedback.jsonl", "a") as f:
        json.dump({
            "ts": time.time(),
            "session_id": session_id,
            "question": question,
            "feedback_type": feedback_type,
        }, f)
        f.write("\\n")
"""

from app.utils.logging import get_logger

log = get_logger(__name__)


def log_feedback(
    session_id: str | None,
    question: str,
    feedback_type: str,
) -> None:
    """
    Record a thumbs-up / thumbs-down signal from the user.

    Args:
        session_id    : browser session UUID (may be None)
        question      : the original question that was rated
        feedback_type : "thumbs_up" or "thumbs_down"
    """
    # ── Default: log to console only ─────────────────────────────────────────
    # Replace this with your database write when ready.
    log.info(
        "Feedback received | session=%s | type=%s | question=%.80s",
        session_id, feedback_type, question,
    )
