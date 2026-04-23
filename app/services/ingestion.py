"""
app/services/ingestion.py
─────────────────────────────────────────────────────────────
PDF Ingestion Service

Responsibilities:
  1. Accept a file path or bytes
  2. Extract text from every page using PyPDF (primary)
     + pdfplumber for table-heavy pages (fallback)
  3. Return a list of PageDocument objects for downstream chunking

Design notes:
  - Stateless: no DB access here; purely text extraction
  - Robust: falls back to pdfplumber if PyPDF yields empty pages
  - Clean: normalizes whitespace via helpers.clean_text
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pdfplumber
from pypdf import PdfReader

from app.utils.helpers import clean_text, get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class PageDocument:
    """Represents one page of extracted text from a PDF."""
    source:   str          # original filename
    page_num: int          # 1-based page number
    text:     str          # cleaned extracted text
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)

    def is_empty(self) -> bool:
        return self.char_count < 20   # fewer than 20 chars → skip


# ══════════════════════════════════════════════════════════════════════════
#  INGESTION SERVICE
# ══════════════════════════════════════════════════════════════════════════

class PDFIngestionService:
    """
    Extracts text from PDF files.

    Usage:
        service = PDFIngestionService()
        pages = service.extract(path="/data/policy.pdf")
    """

    def extract(
        self,
        path: str | Path | None = None,
        content: bytes | None = None,
        filename: str = "uploaded.pdf"
    ) -> List[PageDocument]:
        """
        Extract text from a PDF.

        Args:
            path:     File path on disk (mutually exclusive with content)
            content:  Raw PDF bytes (used when accepting uploads)
            filename: Original filename (for metadata)

        Returns:
            List of PageDocument, one per non-empty page.

        Raises:
            ValueError: if neither path nor content is provided
            FileNotFoundError: if path does not exist
        """
        if path is not None:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"PDF not found: {path}")
            content = path.read_bytes()
            filename = path.name

        if content is None:
            raise ValueError("Either 'path' or 'content' must be provided.")

        logger.info(f"Extracting text from: {filename} ({len(content):,} bytes)")

        pages = self._extract_with_pypdf(content, filename)

        # If PyPDF yields mostly empty pages, try pdfplumber
        non_empty = [p for p in pages if not p.is_empty()]
        if len(non_empty) < len(pages) * 0.5:
            logger.info("PyPDF yield low; retrying with pdfplumber…")
            pages = self._extract_with_pdfplumber(content, filename)

        pages = [p for p in pages if not p.is_empty()]
        logger.info(f"Extraction complete: {len(pages)} usable pages from {filename}")
        return pages

    # ── Private helpers ───────────────────────────────────────────────────

    def _extract_with_pypdf(
        self, content: bytes, filename: str
    ) -> List[PageDocument]:
        """Primary extractor using PyPDF."""
        pages: List[PageDocument] = []
        try:
            reader = PdfReader(io.BytesIO(content))
            for i, page in enumerate(reader.pages):
                raw = page.extract_text() or ""
                pages.append(
                    PageDocument(
                        source=filename,
                        page_num=i + 1,
                        text=clean_text(raw)
                    )
                )
        except Exception as exc:
            logger.warning(f"PyPDF extraction failed: {exc}")
        return pages

    def _extract_with_pdfplumber(
        self, content: bytes, filename: str
    ) -> List[PageDocument]:
        """Fallback extractor using pdfplumber (better for tables/columns)."""
        pages: List[PageDocument] = []
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Extract prose text
                    prose = page.extract_text() or ""

                    # Extract any tables and convert to readable text
                    table_texts: List[str] = []
                    for table in page.extract_tables():
                        rows = [
                            " | ".join(str(cell).strip() for cell in row if cell)
                            for row in table
                            if any(cell for cell in row)
                        ]
                        if rows:
                            table_texts.append("\n".join(rows))

                    full_text = prose
                    if table_texts:
                        full_text += "\n\n[TABLES]\n" + "\n\n".join(table_texts)

                    pages.append(
                        PageDocument(
                            source=filename,
                            page_num=i + 1,
                            text=clean_text(full_text)
                        )
                    )
        except Exception as exc:
            logger.warning(f"pdfplumber extraction failed: {exc}")
        return pages
