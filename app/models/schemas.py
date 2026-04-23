"""
app/models/schemas.py
─────────────────────────────────────────────────────────────
Pydantic v2 schemas for all API request and response objects.
These are the single source of truth for data contracts.
"""

from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════
#  INGESTION
# ══════════════════════════════════════════════════════════════════════════

class IngestResponse(BaseModel):
    """Returned after a PDF is uploaded and processed."""
    status:          Literal["success", "error"]
    filename:        str
    pages_extracted: int = 0
    chunks_created:  int = 0
    message:         str = ""

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "filename": "policy.pdf",
                "pages_extracted": 12,
                "chunks_created": 47,
                "message": ""
            }
        }


# ══════════════════════════════════════════════════════════════════════════
#  QUERY
# ══════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    """Incoming customer query."""
    query:      str = Field(..., min_length=1, max_length=2000, description="Customer's question")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation continuity")
    history:    Optional[List[ConversationTurn]] = Field(default_factory=list, description="Prior turns")

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is your refund policy?",
                "session_id": "sess_abc123"
            }
        }


class ConversationTurn(BaseModel):
    """A single turn in conversation history."""
    role:    Literal["user", "assistant"]
    content: str


# Rebuild after ConversationTurn is defined
QueryRequest.model_rebuild()


class SourceChunk(BaseModel):
    """A retrieved chunk used as evidence for an answer."""
    chunk_id:  str
    source:    str
    page:      int
    score:     float = Field(..., ge=0.0, le=1.0)
    preview:   str   = Field("", description="First 120 chars of chunk text")


class QueryResponse(BaseModel):
    """Full response returned to the client."""
    answer:         str
    confidence:     float = Field(..., ge=0.0, le=1.0)
    intent:         str
    source:         str   = Field("", description="Primary source document")
    sources:        List[SourceChunk] = Field(default_factory=list)
    needs_hitl:     bool  = False
    hitl_ticket_id: Optional[str] = None
    routing_reason: Optional[str] = None
    session_id:     str
    response_ms:    int   = 0
    model_used:     str   = ""
    cached:         bool  = False

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "You can return items within 30 days of purchase.",
                "confidence": 0.91,
                "intent": "faq",
                "source": "policy.pdf",
                "sources": [],
                "needs_hitl": False,
                "session_id": "sess_abc123",
                "response_ms": 1240,
                "model_used": "gpt-4o-mini"
            }
        }


# ══════════════════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════

class KBStatsResponse(BaseModel):
    """Knowledge base health and statistics."""
    total_chunks:   int
    document_count: int
    documents:      List[str]


# ══════════════════════════════════════════════════════════════════════════
#  HITL (Human-in-the-Loop)
# ══════════════════════════════════════════════════════════════════════════

class HITLTicket(BaseModel):
    """Represents one HITL escalation entry."""
    ticket_id:       str
    session_id:      str
    query:           str
    context_preview: str
    reason:          str
    status:          Literal["pending", "resolved", "timeout"]
    created_at:      str
    resolved_at:     Optional[str]  = None
    human_response:  Optional[str]  = None
    agent_id:        Optional[str]  = None


class HITLResolveRequest(BaseModel):
    """Payload for a human agent to close a ticket."""
    human_response: str = Field(..., min_length=1, description="Agent's answer")
    agent_id:       str = Field("agent_1", description="Identifier of the responding agent")


class HITLResolveResponse(BaseModel):
    success:    bool
    ticket_id:  str
    message:    str


class HITLQueueResponse(BaseModel):
    tickets: List[HITLTicket]
    total:   int
    pending: int


# ══════════════════════════════════════════════════════════════════════════
#  HEALTH
# ══════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status:     Literal["healthy", "degraded", "unhealthy"]
    kb:         KBStatsResponse
    config:     dict
    timestamp:  float
