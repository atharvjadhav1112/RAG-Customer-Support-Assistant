"""
app/services/chunking.py
─────────────────────────────────────────────────────────────
Chunking Service

Responsibilities:
  1. Accept a list of PageDocument objects
  2. Split each page into overlapping word-based chunks
  3. Preserve page-level metadata in every chunk

Design notes:
  - Uses word-count as proxy for tokens (1 word ≈ 1.3 tokens for English)
  - Splits on sentence boundaries first, then falls back to word windows
  - Overlap ensures sentences crossing chunk boundaries are not lost
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from app.services.ingestion import PageDocument
from app.utils.helpers import get_logger, make_chunk_id, truncate

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class TextChunk:
    """One piece of text ready for embedding."""
    chunk_id:    str
    source:      str    # original PDF filename
    page:        int    # page number (1-based)
    text:        str    # chunk content
    chunk_index: int    # position in source document (0-based)
    word_count:  int    = field(init=False)
    preview:     str    = field(init=False)

    def __post_init__(self):
        self.word_count = len(self.text.split())
        self.preview    = truncate(self.text, 120)


# ══════════════════════════════════════════════════════════════════════════
#  CHUNKING SERVICE
# ══════════════════════════════════════════════════════════════════════════

class ChunkingService:
    """
    Splits PageDocuments into overlapping TextChunks.

    Strategy (in order of preference):
      1. Split page text into sentences using regex
      2. Greedily accumulate sentences into a window ≤ chunk_size words
      3. When window is full, emit chunk and slide back by overlap words

    Usage:
        svc = ChunkingService(chunk_size=500, chunk_overlap=50)
        chunks = svc.chunk(pages)
    """

    # Sentence boundary regex: end of sentence followed by whitespace
    _SENT_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Public API ─────────────────────────────────────────────────────────

    def chunk(self, pages: List[PageDocument]) -> List[TextChunk]:
        """
        Convert a list of pages into a flat list of TextChunks.

        Args:
            pages: Output from PDFIngestionService.extract()

        Returns:
            List[TextChunk] ordered by source → page → chunk_index
        """
        all_chunks: List[TextChunk] = []
        global_index = 0

        for page in pages:
            page_chunks = self._split_page(page, start_index=global_index)
            all_chunks.extend(page_chunks)
            global_index += len(page_chunks)

        logger.info(
            f"Chunking complete: {len(pages)} pages → {len(all_chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return all_chunks

    # ── Private helpers ────────────────────────────────────────────────────

    def _split_page(
        self, page: PageDocument, start_index: int
    ) -> List[TextChunk]:
        """Split a single page into chunks."""
        sentences = self._split_sentences(page.text)
        if not sentences:
            return []

        chunks: List[TextChunk] = []
        window_sents: List[str] = []    # sentences in current window
        window_words: int = 0           # word count of current window

        for sent in sentences:
            sent_words = len(sent.split())

            # Single sentence exceeds chunk_size → split by words directly
            if sent_words > self.chunk_size:
                # First flush what we have
                if window_sents:
                    chunks.append(
                        self._make_chunk(page, " ".join(window_sents),
                                         start_index + len(chunks))
                    )
                    window_sents, window_words = [], 0

                # Split the long sentence into word windows
                word_chunks = self._split_by_words(sent)
                for wc in word_chunks:
                    chunks.append(
                        self._make_chunk(page, wc, start_index + len(chunks))
                    )
                continue

            # Would adding this sentence overflow the window?
            if window_words + sent_words > self.chunk_size and window_sents:
                # Emit current window
                chunks.append(
                    self._make_chunk(page, " ".join(window_sents),
                                     start_index + len(chunks))
                )
                # Carry-over: keep last N overlap words
                window_sents, window_words = self._apply_overlap(window_sents)

            window_sents.append(sent)
            window_words += sent_words

        # Emit remaining window
        if window_sents:
            chunks.append(
                self._make_chunk(page, " ".join(window_sents),
                                 start_index + len(chunks))
            )

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences; filter empty strings."""
        parts = self._SENT_RE.split(text)
        return [p.strip() for p in parts if p.strip()]

    def _split_by_words(self, text: str) -> List[str]:
        """Hard word-window split for oversized sentences."""
        words  = text.split()
        result = []
        start  = 0
        while start < len(words):
            end = start + self.chunk_size
            result.append(" ".join(words[start:end]))
            if end >= len(words):
                break
            start = end - self.chunk_overlap
        return result

    def _apply_overlap(
        self, sentences: List[str]
    ) -> tuple[List[str], int]:
        """
        From the end of the current window, keep sentences that
        together contribute ≤ chunk_overlap words.
        """
        overlap_sents: List[str] = []
        word_budget = self.chunk_overlap

        for sent in reversed(sentences):
            w = len(sent.split())
            if w <= word_budget:
                overlap_sents.insert(0, sent)
                word_budget -= w
            else:
                break

        word_count = sum(len(s.split()) for s in overlap_sents)
        return overlap_sents, word_count

    def _make_chunk(
        self, page: PageDocument, text: str, index: int
    ) -> TextChunk:
        """Construct a TextChunk with a deterministic ID."""
        return TextChunk(
            chunk_id=make_chunk_id(page.source, page.page_num, index),
            source=page.source,
            page=page.page_num,
            text=text,
            chunk_index=index,
        )
