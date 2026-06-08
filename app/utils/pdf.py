"""
app/utils/pdf.py
----------------
PDF → plain text extraction with PII scrubbing.
Kept minimal: one public function, patterns driven entirely by config.
"""

import os
import re
from pathlib import Path

from app.config import settings


def pdf_to_text(pdf_path: str | Path) -> str:
    """
    Extract text from a PDF file and scrub PII.
    Raises FileNotFoundError or RuntimeError on failure.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        text = _extract(path)
        return _scrub_pii(text)
    except Exception as exc:
        raise RuntimeError(f"Failed to extract text from {path.name}: {exc}") from exc


# ── Private helpers ────────────────────────────────────────────────────────────

def _extract(path: Path) -> str:
    try:
        from pdfminer.high_level import extract_text
    except ImportError as exc:
        raise ImportError(
            "pdfminer.six is required. Run: pip install pdfminer.six"
        ) from exc

    raw = extract_text(str(path))
    return _clean(raw)


def _clean(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _scrub_pii(text: str) -> str:
    for pattern, replacement in settings.PII_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text
