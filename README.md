# RAG Customer Support Assistant

**Production-level Retrieval-Augmented Generation (RAG) system** with Human-in-the-Loop (HITL) escalation, built on LangGraph, ChromaDB, and FastAPI.

---

## Architecture

```
PDF Files
    │
    ▼
┌─────────────────────────────────┐
│   INGESTION PIPELINE            │
│  PyPDF → Chunker → Embedder     │
│              │                  │
│              ▼                  │
│          ChromaDB               │
└─────────────────────────────────┘
                │
┌─────────────────────────────────────────────────────┐
│              LANGGRAPH WORKFLOW                      │
│                                                      │
│  Query → input_node → intent_router_node             │
│                │                                     │
│       ┌────────┼────────┐                            │
│       ▼        ▼        ▼                            │
│  greeting  retrieval  hitl_node ← (escalate intent)  │
│   _node     _node         │                          │
│       │       │           │                          │
│       │    llm_node       │                          │
│       │       │           │                          │
│       │  confidence_eval  │                          │
│       │    /        \     │                          │
│       │  high       low   │                          │
│       │   conf      conf  │                          │
│       │    │          └───┘                          │
│       │    ▼                                         │
│       └→ output_node → END                           │
└─────────────────────────────────────────────────────┘
```

---

## Project Structure

```
rag_project/
│
├── app/
│   ├── main.py              ← FastAPI app, lifespan, middleware
│   ├── config.py            ← All settings (loaded from .env)
│   ├── routes.py            ← All API endpoints
│   ├── services/
│   │   ├── ingestion.py     ← PDF → PageDocument (PyPDF + pdfplumber)
│   │   ├── chunking.py      ← PageDocument → TextChunk (sentence-aware)
│   │   ├── embedding.py     ← SentenceTransformers / OpenAI / TF-IDF fallback
│   │   ├── retriever.py     ← ChromaDB upsert + MMR retrieval
│   │   ├── rag_pipeline.py  ← Intent classifier, prompt builder, LLM call, confidence
│   │   ├── hitl.py          ← HITL queue (JSON persistence, create/resolve/poll)
│   │   └── graph.py         ← LangGraph StateGraph (all nodes + routers)
│   ├── models/
│   │   └── schemas.py       ← Pydantic v2 request/response schemas
│   └── utils/
│       └── helpers.py       ← Logging, IDs, text cleaning, Timer, distance→similarity
│
├── data/
│   ├── generate_sample_pdf.py   ← Generates a test PDF (run once)
│   ├── sample.pdf               ← Generated knowledge base
│   └── hitl_queue.json          ← HITL ticket store (auto-created)
│
├── chroma_db/               ← ChromaDB persistent storage (auto-created)
├── tests/
│   └── test_all.py          ← 31 unit + integration tests
│
├── frontend/
│   └── index.html           ← Web UI (served at /ui)
│
├── run.py                   ← One-command launcher
├── integration_test.py      ← Full pipeline E2E test
├── requirements.txt
├── .env                     ← Configuration (edit before running)
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure (optional)

Edit `.env`:

```env
# For GPT-4 responses (optional — smart mock works without it)
OPENAI_API_KEY=sk-...

# Confidence threshold (lower = more auto-answers, higher = more HITL)
CONFIDENCE_THRESHOLD=0.50
```

### 3. Run

```bash
python run.py
```

This will:
- Generate `data/sample.pdf` if no PDFs exist
- Ingest the PDF into ChromaDB automatically
- Start the API server at **http://localhost:8000**

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload a PDF to the knowledge base |
| `POST` | `/query` | Ask a customer support question |
| `GET`  | `/kb/stats` | Knowledge base statistics |
| `GET`  | `/kb/documents` | List indexed documents |
| `GET`  | `/hitl/queue` | Pending HITL escalations |
| `GET`  | `/hitl/all` | All escalation tickets |
| `GET`  | `/hitl/{id}` | Get one ticket |
| `POST` | `/hitl/{id}/resolve` | Agent resolves an escalation |
| `GET`  | `/health` | System health check |
| `GET`  | `/docs` | Swagger UI (interactive) |
| `GET`  | `/ui` | Web interface |

---

## API Usage Examples

### Upload a PDF

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/your/policy.pdf"
```

Response:
```json
{
  "status": "success",
  "filename": "policy.pdf",
  "pages_extracted": 12,
  "chunks_created": 47
}
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is your refund policy?"}'
```

Response:
```json
{
  "answer": "Refund requests must be submitted within 30 days...",
  "confidence": 0.89,
  "intent": "faq",
  "source": "policy.pdf",
  "sources": [
    {"chunk_id": "abc123", "source": "policy.pdf", "page": 3, "score": 0.91}
  ],
  "needs_hitl": false,
  "session_id": "sess_a1b2c3",
  "response_ms": 1240,
  "model_used": "gpt-4o-mini"
}
```

### Resolve a HITL Ticket

```bash
curl -X POST http://localhost:8000/hitl/tkt_abc123/resolve \
  -H "Content-Type: application/json" \
  -d '{"human_response": "Our policy allows returns within 30 days.", "agent_id": "agent_1"}'
```

---

## LangGraph Flow Details

### Nodes

| Node | Responsibility |
|------|----------------|
| `input_node` | Sanitise and validate query |
| `intent_router_node` | Classify: faq / complaint / technical / escalate / greeting |
| `retrieval_node` | MMR top-K retrieval from ChromaDB |
| `llm_node` | Generate grounded answer (OpenAI or mock) |
| `confidence_eval_node` | Score answer quality (0.0–1.0) |
| `hitl_node` | Create ticket, return escalation placeholder |
| `greeting_node` | Short-circuit for greetings (no retrieval) |
| `output_node` | Final pass-through |

### Routing Rules

```
intent == "greeting"  → greeting_node
intent == "escalate"  → hitl_node  (user explicitly asked)
everything else       → retrieval_node
                            ↓
                     confidence_eval_node
                       /            \
             score ≥ threshold    score < threshold
                   ↓                     ↓
              output_node           hitl_node
```

**Confidence thresholds** (configurable in `.env`):
- Default: `0.50`
- Technical queries: `0.55`
- With OpenAI embeddings, raise to `0.72` / `0.80` for stricter routing

---

## Embedding Backends

The system auto-selects an embedding backend:

| Backend | Condition | Dimension | Quality |
|---------|-----------|-----------|---------|
| OpenAI Ada-002 | `EMBEDDING_BACKEND=openai` + `OPENAI_API_KEY` set | 1536 | ⭐⭐⭐⭐⭐ |
| SentenceTransformers | HuggingFace reachable | 384 | ⭐⭐⭐⭐ |
| TF-IDF (fallback) | No network / no API key | 512 | ⭐⭐ |

The TF-IDF fallback requires no internet or API key — ideal for offline environments.

---

## Running Tests

```bash
# All 31 tests
python tests/test_all.py

# With pytest
pytest tests/ -v
```

### Test Coverage

- `TestHelpers` — text cleaning, ID generation, timer, distance conversion
- `TestChunking` — single/multi chunk, overlap, unique IDs, metadata
- `TestIntentClassifier` — all 5 intent classes
- `TestConfidence` — zero chunks, high/low scores, uncertainty penalty
- `TestHITL` — create, resolve, pending filter, stats, nonexistent ID
- `TestSchemas` — Pydantic validation, empty query rejection
- `TestMockLLM` — context extraction, empty context, prompt structure

---

## Full E2E Integration Test

```bash
python integration_test.py
```

Runs: PDF ingest → retrieval → LangGraph pipeline × 6 queries → HITL create/resolve cycle.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | _(blank)_ | Optional. Enables GPT-4 responses |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `EMBEDDING_BACKEND` | `local` | `local` or `openai` |
| `CHUNK_SIZE` | `500` | Words per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap words between chunks |
| `RETRIEVAL_K` | `5` | Chunks to retrieve per query |
| `CONFIDENCE_THRESHOLD` | `0.50` | Below this → HITL escalation |
| `CONFIDENCE_THRESHOLD_TECHNICAL` | `0.55` | Stricter threshold for technical queries |
| `HITL_TIMEOUT_SECONDS` | `300` | Seconds before ticket auto-times-out |
| `API_HOST` | `0.0.0.0` | Server bind address |
| `API_PORT` | `8000` | Server port |

---

## Adding Your Own PDFs

```bash
# Via API
curl -X POST http://localhost:8000/upload -F "file=@manual.pdf"

# Or drop files in data/ and restart
cp your_document.pdf data/
python run.py --reset    # wipe and re-ingest everything
```

---

## Production Notes

- **Replace TF-IDF** with SentenceTransformers or OpenAI for production-quality retrieval
- **Set OPENAI_API_KEY** for GPT-4 responses instead of smart mock
- **Raise confidence thresholds** to `0.72`/`0.80` once using transformer embeddings
- **HITL queue** uses JSON storage — swap `hitl.py` for Redis/PostgreSQL at scale
- **ChromaDB** handles millions of vectors; for 10M+ use Pinecone or Weaviate
