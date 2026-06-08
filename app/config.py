"""
app/config.py
-------------
Single source of truth for all configuration.
All values are read from environment variables with sensible defaults.
No hardcoded values exist anywhere else in the codebase.
"""

import os
from pathlib import Path


class Settings:
    # ── Paths ─────────────────────────────────────────────────────────────────
    BASE_DIR         = Path(__file__).parent.parent
    PDF_FOLDER       = BASE_DIR / "pdfs"
    FAISS_INDEX_PATH = str(BASE_DIR / "faiss.index")
    FAISS_LC_PATH    = str(BASE_DIR / "faiss_lc")      # LangChain vectorstore
    CHUNK_STORE_PATH = str(BASE_DIR / "chunks.npy")

    # ── Models ────────────────────────────────────────────────────────────────
    RETRIEVER_MODEL         = os.getenv("RETRIEVER_MODEL",  "sentence-transformers/all-mpnet-base-v2")
    RERANKER_MODEL          = os.getenv("RERANKER_MODEL",   "cross-encoder/ms-marco-MiniLM-L-6-v2")
    LLM_MODEL               = os.getenv("LLM_MODEL",        "qwen2.5:3b")
    PREPROCESS_MODEL        = os.getenv("PREPROCESS_MODEL", "qwen2.5:0.5b")
    FOLLOWUP_INTENT_MODEL   = os.getenv("FOLLOWUP_INTENT_MODEL", "qwen2.5:0.5b")

    # ── Ollama endpoints ──────────────────────────────────────────────────────
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"

    # ── Chunking ──────────────────────────────────────────────────────────────
    CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    MIN_PARA_LEN  = int(os.getenv("MIN_PARA_LEN",  "50"))   # chars to keep a paragraph
    MIN_SENT_LEN  = int(os.getenv("MIN_SENT_LEN",  "10"))   # chars to keep a sentence
    WORDS_PER_CHILD_CHUNK = int(os.getenv("WORDS_PER_CHILD_CHUNK", "100"))

    # ── Retrieval ─────────────────────────────────────────────────────────────
    FAISS_K             = int(os.getenv("FAISS_K",   "10"))
    BM25_K              = int(os.getenv("BM25_K",    "10"))
    RRF_K               = int(os.getenv("RRF_K",     "10"))   # candidates after fusion
    FINAL_K             = int(os.getenv("FINAL_K",   "5"))    # chunks sent to LLM
    RRF_CONSTANT        = int(os.getenv("RRF_CONSTANT", "60"))
    PDF_DETECT_POOL     = int(os.getenv("PDF_DETECT_POOL", "30"))   # top-N for PDF selection
    PDF_SCORE_THRESHOLD = float(os.getenv("PDF_SCORE_THRESHOLD", "0.30"))
    RERANKER_THRESHOLD  = float(os.getenv("RERANKER_THRESHOLD",  "-3.0"))
    TOPIC_SHIFT_THRESHOLD = float(os.getenv("TOPIC_SHIFT_THRESHOLD", "0.65"))

    # ── FAISS HNSW tuning ────────────────────────────────────────────────────
    HNSW_M               = int(os.getenv("HNSW_M",   "32"))
    HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))
    HNSW_EF_SEARCH       = int(os.getenv("HNSW_EF_SEARCH", "64"))

    # ── LLM generation ────────────────────────────────────────────────────────
    TEMPERATURE  = float(os.getenv("TEMPERATURE",  "0.3"))
    FOLLOWUP_TEMPERATURE_BOOST = float(os.getenv("FOLLOWUP_TEMPERATURE_BOOST", "0.2"))
    FOLLOWUP_TEMPERATURE_MAX   = float(os.getenv("FOLLOWUP_TEMPERATURE_MAX",   "0.6"))
    NUM_CTX      = int(os.getenv("NUM_CTX",      "2048"))
    MAX_TOKENS   = int(os.getenv("MAX_TOKENS",   "512"))
    HISTORY_WINDOW = int(os.getenv("HISTORY_WINDOW", "6"))  # recent messages to include

    # ── Intent classifier ────────────────────────────────────────────────────
    FOLLOWUP_INTENT_TIMEOUT  = int(os.getenv("FOLLOWUP_INTENT_TIMEOUT",  "10"))
    INTENT_CLASSIFIER_CTX    = int(os.getenv("INTENT_CLASSIFIER_CTX",    "256"))
    INTENT_CLASSIFIER_TOKENS = int(os.getenv("INTENT_CLASSIFIER_TOKENS", "5"))

    # ── Query preprocessor ───────────────────────────────────────────────────
    PREPROCESS_TIMEOUT    = int(os.getenv("PREPROCESS_TIMEOUT", "60"))
    PREPROCESS_CTX        = int(os.getenv("PREPROCESS_CTX",     "512"))
    PREPROCESS_HISTORY_WINDOW = int(os.getenv("PREPROCESS_HISTORY_WINDOW", "6"))
    MIN_WORDS_COMPLETE_QUERY  = int(os.getenv("MIN_WORDS_COMPLETE_QUERY",  "6"))

    # ── Caching ───────────────────────────────────────────────────────────────
    SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
    SEMANTIC_CACHE_MAX       = int(os.getenv("SEMANTIC_CACHE_MAX",         "100"))
    CACHE_TTL_SECONDS        = int(os.getenv("CACHE_TTL_SECONDS",          "1800"))

    # ── Sessions ──────────────────────────────────────────────────────────────
    SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "1800"))

    # ── Scoring / grounding ───────────────────────────────────────────────────
    GROUNDING_WARNING_THRESHOLD = float(os.getenv("GROUNDING_WARNING_THRESHOLD", "0.30"))

    # ── Server ────────────────────────────────────────────────────────────────
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))

    # ── System prompt ─────────────────────────────────────────────────────────
    SYSTEM_PROMPT = """You are an intelligent document assistant. You help users understand documents by providing clear, accurate information based ONLY on the provided context.

STRICT RULES:
1. Answer using the information provided in the CONTEXT below. You may summarize or rephrase, but facts must come entirely from the context.
2. If the answer is NOT found in the context, reply with: "This question is not covered in the provided PDF."
3. NEVER use your own training knowledge or general knowledge.
4. NEVER make up facts, definitions, examples, or code.
5. When a "Previous Answer" is shown — follow the specific Instruction given exactly.
6. Keep answers well-structured and easy to read.
7. Adapt your tone to the document type:
   - Technical docs  -> clear and precise
   - Financial/legal -> factual and direct
   - Educational     -> helpful and explanatory
   - Business/HR     -> professional and structured"""

    # ── Intent labels and instructions ───────────────────────────────────────
    INTENT_INSTRUCTIONS: dict[str, str] = {
        "simplify": (
            "Rewrite the previous answer using SIMPLER words and a real-world analogy. "
            "Avoid jargon. Keep it short and easy to understand. "
            "Do NOT add new information not in the previous answer."
        ),
        "summarize": (
            "Give a BRIEF 1-3 sentence summary of the previous answer only. "
            "Keep only the most important point. No bullet points, no extra detail."
        ),
        "example": (
            "Give a CONCRETE EXAMPLE from the context only. "
            "If no example exists in the context, say: "
            "'No specific example is provided in the PDF.'"
        ),
        "elaborate": (
            "Add NEW points from the context that were NOT already covered "
            "in the previous answer. Do NOT repeat what was already said."
        ),
        "why": (
            "Explain WHY — the reason, purpose, or benefit — "
            "using only the context. Focus on 'why it matters'."
        ),
        "how": (
            "Explain HOW it works using steps or a process from the context. "
            "Use numbered steps if helpful."
        ),
        "compare": (
            "Compare and contrast the concepts using the context only. "
            "Use a simple side-by-side explanation if helpful."
        ),
        "default": (
            "Build on the previous answer with relevant NEW information from the context. "
            "Do NOT repeat the previous answer word for word."
        ),
    }

    INTENT_CLASSIFIER_SYSTEM = (
        "You are a follow-up intent classifier. "
        "Given a user's follow-up question and the previous answer, "
        "output EXACTLY ONE of these intent labels — nothing else:\n\n"
        "  simplify   → user wants a simpler or easier explanation\n"
        "  summarize  → user wants a brief or shorter version\n"
        "  example    → user wants a concrete example or illustration\n"
        "  elaborate  → user wants more detail or additional information\n"
        "  why        → user is asking for reasons, purpose, or benefits\n"
        "  how        → user is asking about steps, process, or mechanism\n"
        "  compare    → user wants a comparison or contrast\n"
        "  default    → none of the above / general follow-up\n\n"
        "Rules:\n"
        "- Output only the label word, lowercase, no punctuation.\n"
        "- If uncertain, output: default"
    )

    # ── PII scrubbing patterns ────────────────────────────────────────────────
    PII_PATTERNS: list[tuple[str, str]] = [
        (r"\b[\w.-]+@[\w.-]+\.\w+\b",  "[EMAIL]"),
        (r"\b\d{3}-\d{2}-\d{4}\b",     "[SSN]"),
        (r"\b[6-9]\d{9}\b",            "[PHONE]"),
    ]


settings = Settings()
