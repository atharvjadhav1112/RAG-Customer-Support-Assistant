#!/usr/bin/env python3
"""
run.py — One-command launcher for the RAG Customer Support Assistant.

Steps performed automatically:
  1. Generates data/sample.pdf if no PDF exists in data/
  2. Ingests all PDFs found in data/ into ChromaDB
  3. Starts the FastAPI server on http://localhost:8000

Usage:
    python run.py                   # start on default port 8000
    python run.py --port 9000       # start on custom port
    python run.py --skip-ingest     # skip PDF ingestion (use existing ChromaDB)
    python run.py --reset           # wipe ChromaDB and re-ingest
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# ── Parse args ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="RAG Customer Support Assistant")
parser.add_argument("--port",         type=int,  default=8000)
parser.add_argument("--host",         type=str,  default="0.0.0.0")
parser.add_argument("--skip-ingest",  action="store_true")
parser.add_argument("--reset",        action="store_true",
                    help="Wipe ChromaDB before starting")
args = parser.parse_args()

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

BANNER = """
╔══════════════════════════════════════════════════════════╗
║       RAG Customer Support Assistant  v1.0               ║
║       LangGraph  ·  ChromaDB  ·  HITL  ·  FastAPI        ║
╚══════════════════════════════════════════════════════════╝"""

print(BANNER)


# ── Optional reset ─────────────────────────────────────────────────────────
if args.reset:
    chroma = ROOT / "chroma_db"
    if chroma.exists():
        shutil.rmtree(chroma)
        print("  ♻  ChromaDB wiped — will re-ingest from scratch.")


# ── Generate sample PDF if data/ has no PDFs ──────────────────────────────
data_dir = ROOT / "data"
data_dir.mkdir(exist_ok=True)
pdfs = list(data_dir.glob("*.pdf"))

if not pdfs:
    print("\n  📄 No PDFs found in data/  — generating sample knowledge base...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "gen", str(data_dir / "generate_sample_pdf.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.generate()
        pdfs = list(data_dir.glob("*.pdf"))
    except Exception as exc:
        print(f"  ⚠  Could not generate sample PDF: {exc}")


# ── Ingest PDFs ────────────────────────────────────────────────────────────
if not args.skip_ingest and pdfs:
    print(f"\n  📥 Ingesting {len(pdfs)} PDF(s) into ChromaDB...")
    try:
        from app.services.ingestion import PDFIngestionService
        from app.services.chunking  import ChunkingService
        from app.services.retriever import VectorStoreService
        from app.config import settings

        ingestor = PDFIngestionService()
        chunker  = ChunkingService(
            chunk_size    = settings.CHUNK_SIZE,
            chunk_overlap = settings.CHUNK_OVERLAP
        )
        store = VectorStoreService()

        for pdf in pdfs:
            pages  = ingestor.extract(path=str(pdf))
            chunks = chunker.chunk(pages)
            stored = store.upsert(chunks)
            print(f"     ✅ {pdf.name}: {len(pages)} pages → {stored} chunks")

        print(f"\n  📊 ChromaDB total: {store.count()} chunks indexed")
    except Exception as exc:
        print(f"  ⚠  Ingestion warning: {exc}")
        print("     (Server will start; upload PDFs via POST /upload)")


# ── Print config ───────────────────────────────────────────────────────────
from app.config import settings
print(f"""
  🔧 Configuration:
     LLM backend   : {'OpenAI ' + settings.LLM_MODEL if settings.use_openai_llm else 'Smart Mock  ← set OPENAI_API_KEY for GPT'}
     Embeddings    : {'OpenAI Ada' if settings.use_openai_embeddings else 'Local SentenceTransformers / TF-IDF fallback'}
     Chunk size    : {settings.CHUNK_SIZE} words  (overlap {settings.CHUNK_OVERLAP})
     Confidence    : {settings.CONFIDENCE_THRESHOLD} (technical: {settings.CONFIDENCE_THRESHOLD_TECHNICAL})
     ChromaDB      : {settings.CHROMA_DIR}

  🚀 Starting server...
     URL   → http://localhost:{args.port}
     UI    → http://localhost:{args.port}/ui
     Docs  → http://localhost:{args.port}/docs
     Press Ctrl+C to stop
""")

# ── Launch uvicorn ─────────────────────────────────────────────────────────
import uvicorn
uvicorn.run(
    "app.main:app",
    host    = args.host,
    port    = args.port,
    reload  = False,
    log_level = "info"
)
