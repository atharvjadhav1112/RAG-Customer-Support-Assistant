"""
app/main.py
─────────────────────────────────────────────────────────────
FastAPI Application Entry Point

Responsibilities:
  - Create and configure the FastAPI app instance
  - Register middleware (CORS, logging)
  - Mount the API router
  - Serve the frontend UI (static HTML)
  - Lifespan: pre-warm embedding model + pre-ingest sample PDF

Run:
    python -m uvicorn app.main:app --reload --port 8000
    OR
    python run.py
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routes import router
from app.utils.helpers import get_logger

# Root logger configuration
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
    datefmt = "%H:%M:%S"
)
logger = get_logger("app.main")


# ══════════════════════════════════════════════════════════════════════════
#  LIFESPAN — startup / shutdown
# ══════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup (before first request) and once at shutdown.
    Pre-warms the embedding model and pre-ingests any PDFs in /data/.
    """
    logger.info("=" * 60)
    logger.info("  RAG Customer Support Assistant — Starting")
    logger.info("=" * 60)
    logger.info(f"  LLM backend:       {'OpenAI ' + settings.LLM_MODEL if settings.use_openai_llm else 'smart-mock'}")
    logger.info(f"  Embedding backend: {'OpenAI Ada' if settings.use_openai_embeddings else 'local ' + settings.EMBEDDING_MODEL_LOCAL}")
    logger.info(f"  ChromaDB path:     {settings.CHROMA_DIR}")
    logger.info(f"  Confidence thr.:   {settings.CONFIDENCE_THRESHOLD}")

    # Pre-warm embedding model (runs in background thread)
    async def _prewarm():
        try:
            from app.services.retriever import get_vector_store
            store = await asyncio.to_thread(get_vector_store)
            count = await asyncio.to_thread(store.count)
            logger.info(f"  ChromaDB ready: {count} chunks indexed")
        except Exception as exc:
            logger.warning(f"  Pre-warm warning: {exc}")

    await _prewarm()
    logger.info("  Server ready — http://localhost:8000")
    logger.info("  API Docs  — http://localhost:8000/docs")
    logger.info("  UI        — http://localhost:8000/ui")
    logger.info("=" * 60)

    yield   # ← application runs here

    logger.info("RAG Assistant shutting down.")


# ══════════════════════════════════════════════════════════════════════════
#  APP FACTORY
# ══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "RAG Customer Support Assistant",
    description = (
        "Production-level RAG system with:\n"
        "- LangGraph workflow orchestration\n"
        "- ChromaDB vector store\n"
        "- Human-in-the-Loop escalation\n"
        "- OpenAI GPT / local SentenceTransformers"
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)


# ── CORS ───────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
    allow_credentials = True,
)


# ── Request timing middleware ──────────────────────────────────────────────
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = int((time.perf_counter() - start) * 1000)
    response.headers["X-Response-Time-Ms"] = str(elapsed)
    return response


# ── Global exception handler ───────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code = 500,
        content     = {"detail": "Internal server error. Please try again."}
    )


# ── Include API router ─────────────────────────────────────────────────────
app.include_router(router, prefix="")


# ── Root redirect ──────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "RAG Customer Support Assistant", "ui": "/ui", "docs": "/docs"}


# ── Serve frontend (static HTML) ───────────────────────────────────────────
_FRONTEND = Path(__file__).parent.parent / "frontend"

if _FRONTEND.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(_FRONTEND)),
        name="static"
    )

    @app.get("/ui", include_in_schema=False)
    async def serve_ui():
        index = _FRONTEND / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return JSONResponse({"detail": "Frontend not found"}, status_code=404)
