"""
Full end-to-end integration test — no pytest needed.
Run: python3 integration_test.py
"""
import sys, uuid
sys.path.insert(0, '.')

SEP = "=" * 60

# ── Step 1: Ingest ────────────────────────────────────────────────────────
print(SEP)
print("STEP 1: Ingest sample PDF into ChromaDB")
print(SEP)

from app.services.ingestion import PDFIngestionService
from app.services.chunking import ChunkingService
from app.services.retriever import VectorStoreService
from app.config import settings

ingestor = PDFIngestionService()
chunker  = ChunkingService(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)

pages  = ingestor.extract(path="data/sample.pdf")
print(f"  Pages extracted : {len(pages)}")

chunks = chunker.chunk(pages)
print(f"  Chunks created  : {len(chunks)}")

store  = VectorStoreService()
stored = store.upsert(chunks)
print(f"  Chunks in ChromaDB: {store.count()}")

# ── Step 2: Retrieval ─────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 2: Semantic retrieval")
print(SEP)

results = store.retrieve("What is the refund policy?", k=3)
for r in results:
    print(f"  [{r.score:.3f}] {r.source} p{r.page}: {r.preview[:70]}...")

# ── Step 3: LangGraph pipeline ────────────────────────────────────────────
print()
print(SEP)
print("STEP 3: LangGraph full pipeline")
print(SEP)

from app.services.graph import run_graph

test_cases = [
    "What is your refund policy?",
    "Hello!",
    "I need to speak to a manager right now",
    "What are your store opening hours?",
    "My device is broken, terrible product",
    "I am getting error code 0x8024001 on install",
]

for query in test_cases:
    state  = run_graph(query, f"sess_{uuid.uuid4().hex[:6]}")
    intent = state.get("intent", "?")
    conf   = state.get("confidence") or 0.0
    hitl   = "HITL" if state.get("needs_hitl") else "Auto"
    ans    = (state.get("final_answer") or "")[:90].replace("\n", " ")
    ticket = state.get("hitl_ticket_id") or ""
    print(f"  Q : {query[:50]}")
    print(f"      intent={intent}  conf={conf:.2f}  route={hitl}  {'ticket='+ticket if ticket else ''}")
    print(f"      A : {ans}...")
    print()

# ── Step 4: HITL ──────────────────────────────────────────────────────────
print(SEP)
print("STEP 4: HITL create + resolve cycle")
print(SEP)

from app.services.hitl import HITLService
svc = HITLService()

tid = svc.create_ticket("sess_int", "Very complex edge case", "context", "low_confidence")
print(f"  Ticket created : {tid}")
print(f"  Pending count  : {len(svc.get_pending())}")

ok  = svc.resolve_ticket(tid, "Here is the definitive answer from the agent.", "agent_007")
t   = svc.get_ticket(tid)
print(f"  Resolved       : {ok}")
print(f"  Status         : {t['status']}")
print(f"  Agent response : {t['human_response'][:60]}...")

print()
print(SEP)
print("ALL INTEGRATION TESTS PASSED")
print(SEP)
