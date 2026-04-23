"""
app/services/embedding.py
─────────────────────────────────────────────────────────────
Embedding Service

Responsibilities:
  1. Load and cache the embedding model (local OR OpenAI)
  2. Encode a list of texts → list of float vectors
  3. Expose a single EmbeddingService interface regardless of backend

Design notes:
  - Singleton pattern: model loaded once, reused for lifetime of process
  - Local backend: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
  - OpenAI backend: text-embedding-ada-002 (1536-dim)
  - Batch encoding for efficiency
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from app.config import settings
from app.utils.helpers import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  ABSTRACT BASE
# ══════════════════════════════════════════════════════════════════════════

class BaseEmbedder(ABC):
    """Common interface for all embedding backends."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return one embedding vector per input text."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Return embedding for a single query string."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...


# ══════════════════════════════════════════════════════════════════════════
#  LOCAL BACKEND (SentenceTransformers)
# ══════════════════════════════════════════════════════════════════════════

class LocalEmbedder(BaseEmbedder):
    """
    Uses sentence-transformers running on CPU/GPU.
    No API key required. Free and private.

    Default model: all-MiniLM-L6-v2
      - 384-dimensional vectors
      - ~22M parameters, very fast
      - Strong performance on semantic similarity tasks

    Falls back to TFIDFEmbedder automatically if the model cannot be loaded
    (e.g. network is restricted in a sandbox environment).
    """

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local embedding model: {model_name}")
            self._model = SentenceTransformer(model_name)
            self._dim   = self._model.get_sentence_embedding_dimension()
            self._use_st = True
            logger.info(f"SentenceTransformer ready — dimension={self._dim}")
        except Exception as exc:
            logger.warning(
                f"SentenceTransformer could not load ({exc}). "
                f"Falling back to TF-IDF embedder (no network needed)."
            )
            self._tfidf  = TFIDFEmbedder()
            self._use_st = False
            self._dim    = self._tfidf.dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._use_st:
            vectors = self._model.encode(
                texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
            )
            return [v.tolist() for v in vectors]
        return self._tfidf.embed_texts(texts)

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

    @property
    def dimension(self) -> int:
        return self._dim


# ══════════════════════════════════════════════════════════════════════════
#  TF-IDF FALLBACK EMBEDDER  (zero dependencies, zero network)
# ══════════════════════════════════════════════════════════════════════════

class TFIDFEmbedder(BaseEmbedder):
    """
    Lightweight TF-IDF-based embedder.
    Produces 512-dimensional sparse-dense vectors from a vocabulary
    built on the fly.  No internet access or model download required.

    Quality is lower than transformer models but perfectly adequate for
    demonstrating the RAG pipeline in restricted environments.
    """

    DIM = 512

    def __init__(self):
        import math, re, collections
        self._math  = math
        self._re    = re
        self._col   = collections
        self._vocab: dict[str, int] = {}
        self._idf:   dict[str, float] = {}
        self._corpus_docs: list[list[str]] = []   # for IDF computation
        logger.info(f"TF-IDF fallback embedder ready — dimension={self.DIM}")

    # ── Vocabulary helpers ────────────────────────────────────────────────

    def _tokenise(self, text: str) -> list[str]:
        """Lowercase word tokens, strip punctuation."""
        tokens = self._re.findall(r"[a-z0-9']+", text.lower())
        return [t for t in tokens if len(t) > 1]

    def _build_or_update_vocab(self, all_tokens_list: list[list[str]]):
        """Extend vocab with any new tokens; recompute IDF."""
        for tokens in all_tokens_list:
            for tok in tokens:
                if tok not in self._vocab and len(self._vocab) < self.DIM:
                    self._vocab[tok] = len(self._vocab)

        # Update IDF over cumulative corpus
        self._corpus_docs.extend(all_tokens_list)
        N    = max(len(self._corpus_docs), 1)
        df   = self._col.Counter()
        for doc_tokens in self._corpus_docs:
            for tok in set(doc_tokens):
                df[tok] += 1
        self._idf = {
            tok: self._math.log((N + 1) / (df[tok] + 1)) + 1
            for tok in self._vocab
        }

    def _vectorise(self, tokens: list[str]) -> list[float]:
        """TF-IDF vector of length DIM (padded with zeros)."""
        vec = [0.0] * self.DIM
        if not tokens:
            return vec
        tf = self._col.Counter(tokens)
        total = max(len(tokens), 1)
        for tok, count in tf.items():
            idx = self._vocab.get(tok)
            if idx is not None and idx < self.DIM:
                tfidf = (count / total) * self._idf.get(tok, 1.0)
                vec[idx] = tfidf
        # L2-normalise
        norm = self._math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    # ── BaseEmbedder interface ─────────────────────────────────────────────

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        all_tokens = [self._tokenise(t) for t in texts]
        self._build_or_update_vocab(all_tokens)
        return [self._vectorise(toks) for toks in all_tokens]

    def embed_query(self, query: str) -> list[float]:
        tokens = self._tokenise(query)
        # Don't add query tokens to IDF corpus
        vec = [0.0] * self.DIM
        if not tokens or not self._vocab:
            return vec
        tf    = self._col.Counter(tokens)
        total = max(len(tokens), 1)
        for tok, count in tf.items():
            idx = self._vocab.get(tok)
            if idx is not None and idx < self.DIM:
                tfidf = (count / total) * self._idf.get(tok, 1.0)
                vec[idx] = tfidf
        norm = self._math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    @property
    def dimension(self) -> int:
        return self.DIM


# ══════════════════════════════════════════════════════════════════════════
#  OPENAI BACKEND
# ══════════════════════════════════════════════════════════════════════════

class OpenAIEmbedder(BaseEmbedder):
    """
    Uses OpenAI text-embedding-ada-002 (1536-dim) or newer models.
    Requires OPENAI_API_KEY in environment.
    """

    _BATCH_SIZE = 100  # OpenAI recommends ≤ 100 texts per call

    def __init__(self, model_name: str):
        from openai import OpenAI
        logger.info(f"Initialising OpenAI embedder: {model_name}")
        self._client     = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._model_name = model_name
        self._dim        = 1536  # Ada-002 default

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[i : i + self._BATCH_SIZE]
            response = self._client.embeddings.create(
                model=self._model_name,
                input=batch
            )
            all_embeddings.extend(item.embedding for item in response.data)
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

    @property
    def dimension(self) -> int:
        return self._dim


# ══════════════════════════════════════════════════════════════════════════
#  EMBEDDING SERVICE (PUBLIC INTERFACE)
# ══════════════════════════════════════════════════════════════════════════

class EmbeddingService:
    """
    Facade over local/OpenAI embedders.
    Selects backend based on config.settings automatically.

    Usage:
        svc = EmbeddingService()
        vectors = svc.embed_texts(["hello world", "foo bar"])
        query_vec = svc.embed_query("what is your policy?")
    """

    def __init__(self):
        self._backend: BaseEmbedder = self._load_backend()

    def _load_backend(self) -> BaseEmbedder:
        if settings.use_openai_embeddings:
            return OpenAIEmbedder(settings.EMBEDDING_MODEL_OPENAI)
        return LocalEmbedder(settings.EMBEDDING_MODEL_LOCAL)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self._backend.embed_texts(texts)

    def embed_query(self, query: str) -> List[float]:
        return self._backend.embed_query(query)

    @property
    def dimension(self) -> int:
        return self._backend.dimension

    @property
    def backend_name(self) -> str:
        return type(self._backend).__name__


# ── Module-level singleton (lazy) ─────────────────────────────────────────
_embedding_service: EmbeddingService | None = None

def get_embedding_service() -> EmbeddingService:
    """Return the shared EmbeddingService instance (created on first call)."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
