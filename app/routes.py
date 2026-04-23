"""
app/routes.py
─────────────────────────────────────────────────────────────
FastAPI Router — all endpoints.

Endpoints:
  POST /upload          — ingest a PDF into the knowledge base
  POST /query           — run a customer query through the RAG pipeline
  GET  /kb/stats        — knowledge base statistics
  GET  /kb/documents    — list ingested documents
  GET  /hitl/queue      — pending HITL tickets
  GET  /hitl/all        — all HITL tickets
  GET  /hitl/{id}       — single ticket lookup
  POST /hitl/{id}/resolve — human agent submits an answer

All endpoints are async; heavy work (embedding, LLM) runs in a
thread pool via asyncio.to_thread so the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.schemas import (
    HealthResponse,
    HITLQueueResponse,
    HITLResolveRequest,
    HITLResolveResponse,
    HITLTicket,
    IngestResponse,
    KBStatsResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from app.services.graph import run_graph
from app.services.hitl import get_hitl_service
from app.services.ingestion import PDFIngestionService
from app.services.chunking import ChunkingService
from app.services.retriever import get_vector_store
from app.utils.helpers import Timer, get_logger

logger = get_logger(__name__)

router = APIRouter()


# ══════════════════════════════════════════════════════════════════════════
#  HEALTH
# ══════════════════════════════════════════════════════════════════════════

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health — KB stats, config summary, timestamp."""
    store  = await asyncio.to_thread(get_vector_store)
    stats  = await asyncio.to_thread(store.stats)
    return HealthResponse(
        status    = "healthy",
        kb        = KBStatsResponse(**stats),
        config    = settings.summary(),
        timestamp = time.time()
    )


# ══════════════════════════════════════════════════════════════════════════
#  INGESTION
# ══════════════════════════════════════════════════════════════════════════

@router.post("/upload", response_model=IngestResponse, tags=["Knowledge Base"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF and ingest it into the ChromaDB vector store.

    - Extracts text with PyPDF + pdfplumber fallback
    - Chunks into ≤500-word windows with 50-word overlap
    - Embeds with SentenceTransformers (or OpenAI if configured)
    - Upserts into ChromaDB collection
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted."
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    # Run the ingestion pipeline in a thread (CPU-bound)
    try:
        result = await asyncio.to_thread(
            _run_ingestion, content, file.filename
        )
    except Exception as exc:
        logger.error(f"Ingestion error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        )

    return result


def _run_ingestion(content: bytes, filename: str) -> IngestResponse:
    """Synchronous ingestion pipeline (runs in thread pool)."""
    ingestor = PDFIngestionService()
    chunker  = ChunkingService(
        chunk_size    = settings.CHUNK_SIZE,
        chunk_overlap = settings.CHUNK_OVERLAP
    )
    store = get_vector_store()

    pages  = ingestor.extract(content=content, filename=filename)
    if not pages:
        return IngestResponse(
            status          = "error",
            filename        = filename,
            message         = "No text could be extracted from the PDF."
        )

    chunks = chunker.chunk(pages)
    if not chunks:
        return IngestResponse(
            status          = "error",
            filename        = filename,
            pages_extracted = len(pages),
            message         = "Chunking produced zero usable chunks."
        )

    stored = store.upsert(chunks)
    return IngestResponse(
        status          = "success",
        filename        = filename,
        pages_extracted = len(pages),
        chunks_created  = stored,
    )


# ══════════════════════════════════════════════════════════════════════════
#  QUERY
# ══════════════════════════════════════════════════════════════════════════

@router.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_endpoint(req: QueryRequest):
    """
    Submit a customer support question.

    The request flows through the full LangGraph pipeline:
      Input → Intent → Retrieval → LLM → Confidence → Route → Output
    """
    if not req.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty."
        )

    session_id = req.session_id or f"sess_{uuid.uuid4().hex[:8]}"
    timer      = Timer()

    # Execute graph in thread (blocks on LLM call)
    final_state = await asyncio.to_thread(
        run_graph,
        req.query,
        session_id,
        req.history or []
    )

    elapsed = timer.elapsed_ms()

    # Build SourceChunk list from retrieved chunks
    raw_chunks  = final_state.get("retrieved_chunks") or []
    source_list = [
        SourceChunk(
            chunk_id = c.chunk_id,
            source   = c.source,
            page     = c.page,
            score    = c.score,
            preview  = c.preview,
        )
        for c in raw_chunks
    ]

    # Derive primary source document name
    primary_source = source_list[0].source if source_list else ""

    llm_result = final_state.get("llm_result")

    return QueryResponse(
        answer         = final_state.get("final_answer") or "No answer generated.",
        confidence     = final_state.get("confidence") or 0.0,
        intent         = final_state.get("intent") or "unknown",
        source         = primary_source,
        sources        = source_list,
        needs_hitl     = final_state.get("needs_hitl", False),
        hitl_ticket_id = final_state.get("hitl_ticket_id"),
        routing_reason = final_state.get("routing_reason"),
        session_id     = session_id,
        response_ms    = elapsed,
        model_used     = llm_result.model_used if llm_result else "unknown",
    )


# ══════════════════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════

@router.get("/kb/stats", response_model=KBStatsResponse, tags=["Knowledge Base"])
async def kb_stats():
    """Return total chunk count, document list, and document count."""
    store = await asyncio.to_thread(get_vector_store)
    stats = await asyncio.to_thread(store.stats)
    return KBStatsResponse(**stats)


@router.get("/kb/documents", tags=["Knowledge Base"])
async def kb_documents():
    """List all source documents currently indexed."""
    store   = await asyncio.to_thread(get_vector_store)
    sources = await asyncio.to_thread(store.list_sources)
    return {"documents": sources, "count": len(sources)}


# ══════════════════════════════════════════════════════════════════════════
#  HITL
# ══════════════════════════════════════════════════════════════════════════

@router.get("/hitl/queue", response_model=HITLQueueResponse, tags=["HITL"])
async def hitl_queue():
    """Return all pending escalation tickets."""
    svc     = get_hitl_service()
    pending = await asyncio.to_thread(svc.get_pending)
    all_    = await asyncio.to_thread(svc.get_all)
    return HITLQueueResponse(
        tickets = [HITLTicket(**t) for t in pending],
        total   = len(all_),
        pending = len(pending)
    )


@router.get("/hitl/all", tags=["HITL"])
async def hitl_all():
    """Return all escalation tickets (all statuses)."""
    svc   = get_hitl_service()
    items = await asyncio.to_thread(svc.get_all)
    return {
        "tickets": items,
        "total":   len(items),
        "stats":   await asyncio.to_thread(svc.stats)
    }


@router.get("/hitl/{ticket_id}", response_model=HITLTicket, tags=["HITL"])
async def hitl_get(ticket_id: str):
    """Fetch a single HITL ticket by its ID."""
    svc    = get_hitl_service()
    ticket = await asyncio.to_thread(svc.get_ticket, ticket_id)
    if not ticket:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket '{ticket_id}' not found."
        )
    return HITLTicket(**ticket)


@router.post(
    "/hitl/{ticket_id}/resolve",
    response_model=HITLResolveResponse,
    tags=["HITL"]
)
async def hitl_resolve(ticket_id: str, req: HITLResolveRequest):
    """
    Human agent submits an answer for an escalated ticket.

    The resolved response is stored and can be served to the waiting
    user if their session is still polling.
    """
    svc     = get_hitl_service()
    success = await asyncio.to_thread(
        svc.resolve_ticket, ticket_id, req.human_response, req.agent_id
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket '{ticket_id}' not found or already closed."
        )
    return HITLResolveResponse(
        success   = True,
        ticket_id = ticket_id,
        message   = "Ticket resolved successfully."
    )
