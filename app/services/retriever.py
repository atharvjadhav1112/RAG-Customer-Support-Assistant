"""
app/services/retriever.py
─────────────────────────────────────────────────────────────
Vector Store & Retrieval Service

Responsibilities:
  1. Initialise ChromaDB persistent client (singleton)
  2. Upsert embedded chunks into the collection
  3. Retrieve top-K semantically similar chunks for a query
  4. Apply Maximum Marginal Relevance (MMR) for diversity
  5. Provide collection statistics

Design notes:
  - ChromaDB uses HNSW index with cosine distance
  - MMR balances relevance and diversity to avoid redundant chunks
  - All DB operations are isolated here — no other module touches ChromaDB
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.services.chunking import TextChunk
from app.services.embedding import get_embedding_service
from app.utils.helpers import distance_to_similarity, get_logger, truncate

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievedChunk:
    """A chunk returned by similarity search, augmented with a score."""
    chunk_id: str
    source:   str
    page:     int
    text:     str
    score:    float   # similarity 0.0–1.0 (higher = more relevant)
    preview:  str     = ""

    def __post_init__(self):
        if not self.preview:
            self.preview = truncate(self.text, 120)


# ══════════════════════════════════════════════════════════════════════════
#  VECTOR STORE SERVICE
# ══════════════════════════════════════════════════════════════════════════

class VectorStoreService:
    """
    Wraps ChromaDB with upsert and MMR-based retrieval.

    Usage:
        store = VectorStoreService()
        store.upsert(chunks)                       # during ingestion
        results = store.retrieve("your query")     # during inference
    """

    def __init__(self):
        self._client     = self._init_client()
        self._collection = self._init_collection()
        self._embedder   = get_embedding_service()
        logger.info(
            f"VectorStore ready — collection='{settings.CHROMA_COLLECTION}', "
            f"chunks={self._collection.count()}"
        )

    # ── Initialisation ─────────────────────────────────────────────────────

    def _init_client(self) -> chromadb.PersistentClient:
        return chromadb.PersistentClient(
            path=str(settings.CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False)
        )

    def _init_collection(self):
        return self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )

    # ── Write ──────────────────────────────────────────────────────────────

    def upsert(self, chunks: List[TextChunk]) -> int:
        """
        Embed and upsert chunks into ChromaDB.

        Args:
            chunks: Output from ChunkingService.chunk()

        Returns:
            Number of chunks successfully stored.
        """
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks…")
        embeddings = self._embedder.embed_texts(texts)

        ids       = [c.chunk_id    for c in chunks]
        metadatas = [
            {
                "source":      c.source,
                "page":        c.page,
                "chunk_index": c.chunk_index,
                "word_count":  c.word_count
            }
            for c in chunks
        ]

        # Upsert in batches of 500 to stay within ChromaDB limits
        BATCH = 500
        for i in range(0, len(ids), BATCH):
            self._collection.upsert(
                ids        = ids[i:i+BATCH],
                documents  = texts[i:i+BATCH],
                embeddings = embeddings[i:i+BATCH],
                metadatas  = metadatas[i:i+BATCH]
            )

        logger.info(f"Upserted {len(ids)} chunks into '{settings.CHROMA_COLLECTION}'")
        return len(ids)

    # ── Read ───────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        fetch_k: int | None = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve top-k chunks most relevant to `query` using MMR.

        Args:
            query:   Natural language question
            k:       Number of chunks to return (defaults to settings.RETRIEVAL_K)
            fetch_k: Candidate pool size for MMR (default settings.RETRIEVAL_FETCH_K)

        Returns:
            List[RetrievedChunk] sorted by descending score.
        """
        k       = k       or settings.RETRIEVAL_K
        fetch_k = fetch_k or settings.RETRIEVAL_FETCH_K

        total = self._collection.count()
        if total == 0:
            logger.warning("ChromaDB collection is empty — no chunks to retrieve")
            return []

        # Cap fetch_k to actual collection size
        n_results = min(fetch_k, total)
        query_vec = self._embedder.embed_query(query)

        raw = self._collection.query(
            query_embeddings = [query_vec],
            n_results        = n_results,
            include          = ["documents", "metadatas", "distances"]
        )

        candidates = self._parse_results(raw)
        if not candidates:
            return []

        # Apply MMR diversification
        selected = self._mmr(candidates, k=min(k, len(candidates)))
        logger.info(
            f"Retrieved {len(selected)} chunks for query='{query[:60]}…' "
            f"(top score={selected[0].score:.3f})"
        )
        return selected

    # ── Statistics ─────────────────────────────────────────────────────────

    def count(self) -> int:
        return self._collection.count()

    def list_sources(self) -> List[str]:
        """Return sorted list of unique source filenames in the collection."""
        try:
            result = self._collection.get(include=["metadatas"])
            sources = {m["source"] for m in result["metadatas"] if m}
            return sorted(sources)
        except Exception:
            return []

    def stats(self) -> dict:
        sources = self.list_sources()
        return {
            "total_chunks":   self.count(),
            "document_count": len(sources),
            "documents":      sources
        }

    # ── Private helpers ────────────────────────────────────────────────────

    def _parse_results(self, raw: dict) -> List[RetrievedChunk]:
        """Convert raw ChromaDB query output into RetrievedChunk objects."""
        chunks: List[RetrievedChunk] = []
        ids       = raw["ids"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        for cid, doc, meta, dist in zip(ids, documents, metadatas, distances):
            chunks.append(RetrievedChunk(
                chunk_id = cid,
                source   = meta.get("source", "unknown"),
                page     = int(meta.get("page", 0)),
                text     = doc,
                score    = distance_to_similarity(dist)
            ))

        return chunks

    def _mmr(
        self,
        candidates: List[RetrievedChunk],
        k: int,
        lambda_: float = 0.5
    ) -> List[RetrievedChunk]:
        """
        Maximum Marginal Relevance selection.

        Iteratively picks the candidate that maximises:
            MMR(c) = λ * relevance(c) − (1−λ) * max_sim_to_selected(c)

        Args:
            candidates: Full candidate list, sorted by relevance.
            k:          Number of items to select.
            lambda_:    Trade-off: 1.0 = pure relevance, 0.0 = pure diversity.

        Returns:
            k diverse-yet-relevant RetrievedChunks.
        """
        selected:   List[RetrievedChunk] = []
        remaining:  List[RetrievedChunk] = candidates.copy()

        while len(selected) < k and remaining:
            if not selected:
                # First pick: highest relevance
                best = max(remaining, key=lambda c: c.score)
            else:
                def mmr_score(c: RetrievedChunk) -> float:
                    # Proxy for similarity to already-selected: word-overlap ratio
                    sel_words = set(" ".join(s.text for s in selected).split())
                    cand_words = set(c.text.split())
                    if not cand_words:
                        return 0.0
                    sim = len(cand_words & sel_words) / len(cand_words)
                    return lambda_ * c.score - (1 - lambda_) * sim

                best = max(remaining, key=mmr_score)

            selected.append(best)
            remaining.remove(best)

        return selected


# ── Module-level singleton ────────────────────────────────────────────────
_vector_store: VectorStoreService | None = None

def get_vector_store() -> VectorStoreService:
    """Return shared VectorStoreService (created once per process)."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store
