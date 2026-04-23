"""
app/utils/helpers.py
─────────────────────────────────────────────────────────────
Shared utility functions used across services.
"""

import hashlib
import logging
import re
import time
import uuid
from typing import Any


# ══════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════

def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ══════════════════════════════════════════════════════════════════════════
#  ID GENERATION
# ══════════════════════════════════════════════════════════════════════════

def make_chunk_id(source: str, page: int, index: int) -> str:
    """Deterministic chunk ID — same inputs always produce same ID."""
    raw = f"{source}::{page}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def make_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:8]}"


def make_ticket_id() -> str:
    return f"tkt_{uuid.uuid4().hex[:8]}"


# ══════════════════════════════════════════════════════════════════════════
#  TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Normalize raw PDF text:
    - Collapse multiple spaces/newlines
    - Remove control characters
    - Strip leading/trailing whitespace
    """
    if not text:
        return ""
    # Remove non-printable characters (except newline/tab)
    text = re.sub(r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]", " ", text)
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse horizontal whitespace runs
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def truncate(text: str, max_chars: int = 120) -> str:
    """Return first max_chars of text, appending '…' if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


# ══════════════════════════════════════════════════════════════════════════
#  TIMING
# ══════════════════════════════════════════════════════════════════════════

class Timer:
    """Simple wall-clock timer for measuring elapsed milliseconds."""

    def __init__(self):
        self._start = time.perf_counter()

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self._start) * 1000)

    def reset(self):
        self._start = time.perf_counter()


# ══════════════════════════════════════════════════════════════════════════
#  COSINE DISTANCE → SIMILARITY
# ══════════════════════════════════════════════════════════════════════════

def distance_to_similarity(distance: float) -> float:
    """
    Convert ChromaDB cosine distance (0=identical, 2=opposite)
    to a similarity score in [0.0, 1.0].
    """
    similarity = 1.0 - (distance / 2.0)
    return round(max(0.0, min(1.0, similarity)), 4)
