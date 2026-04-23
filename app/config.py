"""
app/config.py
─────────────────────────────────────────────────────────────
Central configuration module.
All settings are loaded from the .env file (or OS environment).
Import `settings` anywhere in the project — never hard-code values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Locate and load .env ───────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent   # project root
load_dotenv(BASE_DIR / ".env", override=False)       # won't override real env vars


class Settings:
    """
    Typed settings object.
    All values are read once at import time.
    """

    # ── Paths ──────────────────────────────────────────────────────────────
    BASE_DIR:   Path = BASE_DIR
    DATA_DIR:   Path = BASE_DIR / "data"
    CHROMA_DIR: Path = BASE_DIR / "chroma_db"

    # ── OpenAI ────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str  = os.getenv("OPENAI_API_KEY", "")

    # ── LLM ───────────────────────────────────────────────────────────────
    LLM_MODEL:       str   = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    LLM_MAX_TOKENS:  int   = int(os.getenv("LLM_MAX_TOKENS", "600"))

    # ── Embedding ─────────────────────────────────────────────────────────
    EMBEDDING_BACKEND:      str = os.getenv("EMBEDDING_BACKEND", "local")
    EMBEDDING_MODEL_LOCAL:  str = os.getenv("EMBEDDING_MODEL_LOCAL", "all-MiniLM-L6-v2")
    EMBEDDING_MODEL_OPENAI: str = os.getenv("EMBEDDING_MODEL_OPENAI", "text-embedding-ada-002")

    # ── Chunking ──────────────────────────────────────────────────────────
    CHUNK_SIZE:    int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # ── Retrieval ─────────────────────────────────────────────────────────
    RETRIEVAL_K:     int   = int(os.getenv("RETRIEVAL_K", "5"))
    RETRIEVAL_FETCH_K: int = int(os.getenv("RETRIEVAL_FETCH_K", "20"))

    CONFIDENCE_THRESHOLD:           float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.72"))
    CONFIDENCE_THRESHOLD_TECHNICAL: float = float(os.getenv("CONFIDENCE_THRESHOLD_TECHNICAL", "0.80"))

    # ── ChromaDB ──────────────────────────────────────────────────────────
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "support_kb")

    # ── API ───────────────────────────────────────────────────────────────
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # ── HITL ──────────────────────────────────────────────────────────────
    HITL_TIMEOUT_SECONDS: int = int(os.getenv("HITL_TIMEOUT_SECONDS", "300"))

    def __post_init__(self):
        # Ensure required directories exist
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def use_openai_llm(self) -> bool:
        return bool(self.OPENAI_API_KEY)

    @property
    def use_openai_embeddings(self) -> bool:
        return self.EMBEDDING_BACKEND == "openai" and bool(self.OPENAI_API_KEY)

    def summary(self) -> dict:
        """Return a safe (no secrets) config summary for health endpoints."""
        return {
            "llm_backend":        "openai" if self.use_openai_llm else "mock",
            "llm_model":          self.LLM_MODEL if self.use_openai_llm else "smart-mock",
            "embedding_backend":  "openai" if self.use_openai_embeddings else "local",
            "embedding_model":    self.EMBEDDING_MODEL_OPENAI if self.use_openai_embeddings else self.EMBEDDING_MODEL_LOCAL,
            "chunk_size":         self.CHUNK_SIZE,
            "chunk_overlap":      self.CHUNK_OVERLAP,
            "retrieval_k":        self.RETRIEVAL_K,
            "confidence_threshold": self.CONFIDENCE_THRESHOLD,
            "chroma_collection":  self.CHROMA_COLLECTION,
        }


# ── Singleton instance ─────────────────────────────────────────────────────
settings = Settings()

# ── Ensure dirs exist on import ────────────────────────────────────────────
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
