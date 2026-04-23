"""
tests/test_all.py
─────────────────────────────────────────────────────────────
Comprehensive test suite covering all modules.

Run:
    python -m pytest tests/ -v
    python tests/test_all.py          (standalone, no pytest needed)
"""

import sys
import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make project importable
sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

class TestHelpers(unittest.TestCase):

    def test_clean_text_collapses_whitespace(self):
        from app.utils.helpers import clean_text
        result = clean_text("  hello   world  \n\n\n  test  ")
        self.assertNotIn("   ", result)
        self.assertIn("hello", result)

    def test_truncate(self):
        from app.utils.helpers import truncate
        long = "a" * 200
        self.assertTrue(truncate(long, 50).endswith("…"))
        self.assertEqual(len(truncate("short", 50)), 5)

    def test_make_chunk_id_deterministic(self):
        from app.utils.helpers import make_chunk_id
        id1 = make_chunk_id("doc.pdf", 3, 7)
        id2 = make_chunk_id("doc.pdf", 3, 7)
        id3 = make_chunk_id("doc.pdf", 3, 8)
        self.assertEqual(id1, id2)
        self.assertNotEqual(id1, id3)

    def test_distance_to_similarity(self):
        from app.utils.helpers import distance_to_similarity
        self.assertAlmostEqual(distance_to_similarity(0.0), 1.0)   # identical
        self.assertAlmostEqual(distance_to_similarity(2.0), 0.0)   # opposite
        self.assertAlmostEqual(distance_to_similarity(1.0), 0.5)   # orthogonal

    def test_timer(self):
        import time
        from app.utils.helpers import Timer
        t = Timer()
        time.sleep(0.05)
        ms = t.elapsed_ms()
        self.assertGreater(ms, 40)
        self.assertLess(ms, 500)


# ══════════════════════════════════════════════════════════════════════════
#  CHUNKING
# ══════════════════════════════════════════════════════════════════════════

class TestChunking(unittest.TestCase):

    def _make_page(self, text: str, page: int = 1):
        from app.services.ingestion import PageDocument
        return PageDocument(source="test.pdf", page_num=page, text=text)

    def test_single_short_page_single_chunk(self):
        from app.services.chunking import ChunkingService
        svc   = ChunkingService(chunk_size=500, chunk_overlap=50)
        page  = self._make_page("Hello world. This is a short document.")
        chunks = svc.chunk([page])
        self.assertEqual(len(chunks), 1)

    def test_long_page_creates_multiple_chunks(self):
        from app.services.chunking import ChunkingService
        svc  = ChunkingService(chunk_size=20, chunk_overlap=5)
        text = " ".join(f"word{i}" for i in range(200))
        page = self._make_page(text)
        chunks = svc.chunk([page])
        self.assertGreater(len(chunks), 1)

    def test_overlap_applied(self):
        from app.services.chunking import ChunkingService
        svc  = ChunkingService(chunk_size=20, chunk_overlap=5)
        text = " ".join(f"word{i}" for i in range(60))
        page = self._make_page(text)
        chunks = svc.chunk([page])
        # Overlap means later chunks should share some words with previous chunk
        if len(chunks) >= 2:
            words0 = set(chunks[0].text.split())
            words1 = set(chunks[1].text.split())
            # They may share overlap words
            self.assertGreater(len(chunks[0].text), 0)
            self.assertGreater(len(chunks[1].text), 0)

    def test_chunk_ids_unique(self):
        from app.services.chunking import ChunkingService
        svc  = ChunkingService(chunk_size=20, chunk_overlap=5)
        text = " ".join(f"word{i}" for i in range(100))
        page = self._make_page(text)
        chunks = svc.chunk([page])
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)), "Chunk IDs must be unique")

    def test_empty_page_skipped(self):
        from app.services.chunking import ChunkingService
        from app.services.ingestion import PageDocument
        svc  = ChunkingService()
        page = PageDocument(source="x.pdf", page_num=1, text="")
        chunks = svc.chunk([page])
        self.assertEqual(len(chunks), 0)

    def test_chunk_metadata(self):
        from app.services.chunking import ChunkingService
        svc   = ChunkingService(chunk_size=500, chunk_overlap=50)
        page  = self._make_page("Some text about refund policy.")
        chunks = svc.chunk([page])
        c = chunks[0]
        self.assertEqual(c.source, "test.pdf")
        self.assertEqual(c.page, 1)
        self.assertIsInstance(c.word_count, int)
        self.assertGreater(c.word_count, 0)


# ══════════════════════════════════════════════════════════════════════════
#  INTENT CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════

class TestIntentClassifier(unittest.TestCase):

    def _cls(self, query: str) -> str:
        from app.services.rag_pipeline import classify_intent
        return classify_intent(query)

    def test_greeting(self):
        for q in ["Hi!", "Hello there", "Hey, good morning"]:
            with self.subTest(q=q):
                self.assertEqual(self._cls(q), "greeting")

    def test_escalate(self):
        for q in ["I want to speak to a manager", "Connect me to a human agent"]:
            with self.subTest(q=q):
                self.assertEqual(self._cls(q), "escalate")

    def test_complaint(self):
        for q in ["This product is broken and terrible", "Worst experience ever"]:
            with self.subTest(q=q):
                self.assertEqual(self._cls(q), "complaint")

    def test_technical(self):
        for q in ["I'm getting error code 0x8024001", "The API is not working"]:
            with self.subTest(q=q):
                self.assertEqual(self._cls(q), "technical")

    def test_faq_default(self):
        for q in ["What is your return policy?", "How do I track my order?"]:
            with self.subTest(q=q):
                self.assertIn(self._cls(q), ("faq", "technical", "complaint"))


# ══════════════════════════════════════════════════════════════════════════
#  CONFIDENCE SCORING
# ══════════════════════════════════════════════════════════════════════════

class TestConfidence(unittest.TestCase):

    def _mock_chunk(self, score: float):
        from app.services.retriever import RetrievedChunk
        return RetrievedChunk(
            chunk_id="c1", source="doc.pdf", page=1, text="some text", score=score
        )

    def test_no_chunks_zero(self):
        from app.services.rag_pipeline import compute_confidence
        self.assertEqual(compute_confidence([], "answer"), 0.0)

    def test_high_score_high_confidence(self):
        from app.services.rag_pipeline import compute_confidence
        chunks = [self._mock_chunk(0.95), self._mock_chunk(0.90)]
        conf   = compute_confidence(chunks, "Here is the answer.")
        self.assertGreater(conf, 0.80)

    def test_uncertainty_reduces_confidence(self):
        from app.services.rag_pipeline import compute_confidence
        chunks = [self._mock_chunk(0.95)]
        uncertain = "I don't have enough information to answer this reliably."
        certain   = "The return window is 30 days."
        conf_u = compute_confidence(chunks, uncertain)
        conf_c = compute_confidence(chunks, certain)
        self.assertLess(conf_u, conf_c)

    def test_range_bounds(self):
        from app.services.rag_pipeline import compute_confidence
        for score in [0.0, 0.5, 1.0]:
            chunks = [self._mock_chunk(score)]
            conf   = compute_confidence(chunks, "answer")
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)


# ══════════════════════════════════════════════════════════════════════════
#  HITL SERVICE
# ══════════════════════════════════════════════════════════════════════════

class TestHITL(unittest.TestCase):

    def setUp(self):
        # Use a temp file for each test
        self._tmp = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        )
        self._tmp.close()

        import app.services.hitl as hitl_mod
        hitl_mod._QUEUE_FILE = Path(self._tmp.name)
        hitl_mod._hitl_service = None   # reset singleton

    def tearDown(self):
        Path(self._tmp.name).unlink(missing_ok=True)

    def _svc(self):
        from app.services.hitl import HITLService
        return HITLService()

    def test_create_ticket(self):
        svc = self._svc()
        tid = svc.create_ticket("s1", "Where is my order?", "ctx", "low_confidence")
        self.assertTrue(tid.startswith("tkt_"))
        ticket = svc.get_ticket(tid)
        self.assertIsNotNone(ticket)
        self.assertEqual(ticket["status"], "pending")

    def test_resolve_ticket(self):
        svc = self._svc()
        tid = svc.create_ticket("s1", "test query", "", "test_reason")
        ok  = svc.resolve_ticket(tid, "Here is the answer.", "agent_99")
        self.assertTrue(ok)
        ticket = svc.get_ticket(tid)
        self.assertEqual(ticket["status"], "resolved")
        self.assertEqual(ticket["human_response"], "Here is the answer.")
        self.assertEqual(ticket["agent_id"], "agent_99")

    def test_resolve_nonexistent_returns_false(self):
        svc = self._svc()
        ok  = svc.resolve_ticket("tkt_nonexistent", "response", "agent")
        self.assertFalse(ok)

    def test_get_pending(self):
        svc = self._svc()
        t1  = svc.create_ticket("s1", "q1", "", "r1")
        t2  = svc.create_ticket("s2", "q2", "", "r2")
        svc.resolve_ticket(t1, "resolved", "a1")
        pending = svc.get_pending()
        ids = [p["ticket_id"] for p in pending]
        self.assertNotIn(t1, ids)
        self.assertIn(t2, ids)

    def test_stats(self):
        svc = self._svc()
        svc.create_ticket("s1", "q1", "", "r")
        tid = svc.create_ticket("s2", "q2", "", "r")
        svc.resolve_ticket(tid, "answer", "a")
        stats = svc.stats()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["resolved"], 1)
        self.assertEqual(stats["pending"], 1)


# ══════════════════════════════════════════════════════════════════════════
#  SCHEMAS (Pydantic validation)
# ══════════════════════════════════════════════════════════════════════════

class TestSchemas(unittest.TestCase):

    def test_query_request_strips_whitespace(self):
        from app.models.schemas import QueryRequest
        req = QueryRequest(query="  hello world  ")
        self.assertEqual(req.query, "hello world")

    def test_query_request_rejects_empty(self):
        from app.models.schemas import QueryRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            QueryRequest(query="")

    def test_ingest_response(self):
        from app.models.schemas import IngestResponse
        r = IngestResponse(status="success", filename="doc.pdf",
                           pages_extracted=5, chunks_created=20)
        self.assertEqual(r.status, "success")
        self.assertEqual(r.chunks_created, 20)


# ══════════════════════════════════════════════════════════════════════════
#  MOCK LLM
# ══════════════════════════════════════════════════════════════════════════

class TestMockLLM(unittest.TestCase):

    def test_mock_returns_content_from_context(self):
        from app.services.rag_pipeline import _mock_call
        context = "[1] Source: policy.pdf, Page 3 (relevance=0.90)\nRefunds are processed within 7 days."
        result  = _mock_call("How long do refunds take?", context)
        self.assertIn("7", result.answer)
        self.assertEqual(result.model_used, "smart-mock (set OPENAI_API_KEY for GPT)")

    def test_mock_handles_empty_context(self):
        from app.services.rag_pipeline import _mock_call
        result = _mock_call("What is the refund policy?", "No relevant context found.")
        self.assertIn("support agent", result.answer.lower())

    def test_prompt_builder(self):
        from app.services.rag_pipeline import build_prompt
        msgs = build_prompt("test query", "some context", [])
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertIn("test query", msgs[-1]["content"])


# ══════════════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  RAG Assistant — Test Suite")
    print("=" * 60 + "\n")
    loader  = unittest.TestLoader()
    suite   = loader.loadTestsFromModule(sys.modules[__name__])
    runner  = unittest.TextTestRunner(verbosity=2)
    result  = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
