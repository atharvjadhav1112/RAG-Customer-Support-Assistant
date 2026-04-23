"""
app/services/graph.py
─────────────────────────────────────────────────────────────
LangGraph Workflow Engine

This is the central orchestration layer. It defines:
  - GraphState TypedDict  — the shared state flowing between nodes
  - 7 Node functions      — pure functions transforming state
  - 2 Router functions    — determine which edge to follow
  - build_graph()         — compiles the StateGraph

Graph topology:
                          ┌──────────────┐
                          │  input_node  │
                          └──────┬───────┘
                                 │
                          ┌──────▼───────┐
                          │ intent_router│
                          └──┬───┬───┬───┘
                    greeting │   │faq│   │ escalate
               ┌─────────────┘   │   └───┼────────────┐
               ▼                 ▼       ▼            ▼
        greeting_node    retrieval_node        hitl_node
                                 │                  ▲
                          llm_node                  │
                                 │          low confidence
                       confidence_eval_node ────────┘
                                 │
                              high conf
                                 │
                         output_node → END
"""

from __future__ import annotations

from typing import List, Literal, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from app.config import settings
from app.services.rag_pipeline import (
    LLMResult,
    call_llm,
    classify_intent,
    compute_confidence,
    format_context,
)
from app.services.retriever import RetrievedChunk, get_vector_store
from app.services.hitl import get_hitl_service
from app.utils.helpers import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  GRAPH STATE
# ══════════════════════════════════════════════════════════════════════════

class GraphState(TypedDict):
    """
    The single mutable object passed between every node.
    Nodes return a PARTIAL dict — only the keys they modify.
    LangGraph merges these back into the running state automatically.
    """
    # ── Input ──────────────────────────────────────────────────────────
    query:      str
    session_id: str
    history:    List[dict]

    # ── Intermediate ───────────────────────────────────────────────────
    intent:          Optional[str]              # faq|complaint|technical|escalate|greeting
    retrieved_chunks: Optional[List[RetrievedChunk]]
    context_string:  Optional[str]

    # ── LLM output ─────────────────────────────────────────────────────
    llm_result:       Optional[LLMResult]
    confidence:       Optional[float]
    final_answer:     Optional[str]

    # ── Control ────────────────────────────────────────────────────────
    needs_hitl:       bool
    hitl_ticket_id:   Optional[str]
    routing_reason:   Optional[str]
    error:            Optional[str]


# ══════════════════════════════════════════════════════════════════════════
#  NODE FUNCTIONS  (pure: GraphState → partial GraphState)
# ══════════════════════════════════════════════════════════════════════════

def input_node(state: GraphState) -> dict:
    """
    NODE 1 — Input
    Sanitise and validate the incoming query.
    Truncates, strips, and sets safe defaults.
    """
    query = (state.get("query") or "").strip()
    if not query:
        query = "Hello"
    if len(query) > 1500:
        query = query[:1500] + "…"

    logger.info(f"[input_node] query='{query[:80]}' session={state.get('session_id')}")
    return {
        "query":       query,
        "needs_hitl":  False,
        "history":     state.get("history") or [],
    }


def intent_router_node(state: GraphState) -> dict:
    """
    NODE 2 — Intent Classification
    Determines which branch of the graph to follow.
    No DB access — fast keyword matching.
    """
    intent = classify_intent(state["query"])
    logger.info(f"[intent_router] intent='{intent}'")
    return {"intent": intent}


def retrieval_node(state: GraphState) -> dict:
    """
    NODE 3 — Retrieval
    Queries ChromaDB for the top-K most relevant chunks.
    Technical queries use a larger K for broader coverage.
    """
    store = get_vector_store()
    k     = 8 if state.get("intent") == "technical" else settings.RETRIEVAL_K

    chunks = store.retrieve(state["query"], k=k)
    context = format_context(chunks)

    logger.info(f"[retrieval_node] {len(chunks)} chunks retrieved (k={k})")
    return {
        "retrieved_chunks": chunks,
        "context_string":   context,
    }


def llm_node(state: GraphState) -> dict:
    """
    NODE 4 — LLM Generation
    Calls the language model with context + query → answer.
    Handles OpenAI and mock paths transparently.
    """
    result = call_llm(
        query   = state["query"],
        context = state.get("context_string") or "No relevant context found.",
        history = state.get("history") or []
    )
    logger.info(f"[llm_node] model='{result.model_used}' tokens={result.tokens}")
    return {"llm_result": result}


def confidence_eval_node(state: GraphState) -> dict:
    """
    NODE 5 — Confidence Evaluation
    Scores answer quality. Decides whether HITL escalation is needed.
    """
    chunks     = state.get("retrieved_chunks") or []
    llm_result = state.get("llm_result")
    answer     = llm_result.answer if llm_result else ""

    confidence = compute_confidence(chunks, answer)
    logger.info(f"[confidence_eval] score={confidence:.3f}")

    return {
        "confidence":  confidence,
        "final_answer": answer,
    }


def hitl_node(state: GraphState) -> dict:
    """
    NODE 6 — Human-in-the-Loop Escalation
    Creates a HITL ticket and returns a placeholder answer.
    In production: graph would pause here and resume after agent resolves.
    """
    svc = get_hitl_service()
    ticket_id = svc.create_ticket(
        session_id = state.get("session_id", "unknown"),
        query      = state["query"],
        context    = state.get("context_string") or "",
        reason     = state.get("routing_reason") or "low_confidence"
    )

    placeholder = (
        f"Your query has been escalated to our support team "
        f"(Ticket ID: {ticket_id}). "
        f"A human agent will respond within 5 minutes. "
        f"Reason: {state.get('routing_reason', 'requires human review')}."
    )

    logger.info(f"[hitl_node] ticket created: {ticket_id}")
    return {
        "needs_hitl":     True,
        "hitl_ticket_id": ticket_id,
        "final_answer":   placeholder,
        "confidence":     0.0,
    }


def greeting_node(state: GraphState) -> dict:
    """
    NODE 7 — Greeting Handler
    Short-circuits retrieval for simple greetings.
    """
    answer = (
        "Hello! 👋 I'm your AI-powered customer support assistant. "
        "I can help with product questions, policies, returns, technical issues, and more. "
        "What can I help you with today?"
    )
    logger.info("[greeting_node] greeting response sent")
    return {
        "final_answer":     answer,
        "confidence":       1.0,
        "retrieved_chunks": [],
        "needs_hitl":       False,
    }


def output_node(state: GraphState) -> dict:
    """
    NODE 8 — Output
    Final pass-through node. Could add logging, formatting, or caching here.
    """
    logger.info(
        f"[output_node] answer_len={len(state.get('final_answer',''))} "
        f"confidence={state.get('confidence',0):.3f} "
        f"hitl={state.get('needs_hitl')}"
    )
    return {}   # no changes — just signals end of graph


# ══════════════════════════════════════════════════════════════════════════
#  ROUTER FUNCTIONS  (state → branch key)
# ══════════════════════════════════════════════════════════════════════════

def route_by_intent(
    state: GraphState
) -> Literal["retrieval", "hitl", "greeting"]:
    """
    After intent_router_node:
      - greeting  → greeting_node (no retrieval needed)
      - escalate  → hitl_node (user explicitly asked for human)
      - everything else → retrieval_node
    """
    intent = state.get("intent", "faq")

    if intent == "greeting":
        return "greeting"

    if intent == "escalate":
        # Mutate state directly before routing (LangGraph allows this)
        state["routing_reason"] = "user_requested_human_agent"
        return "hitl"

    return "retrieval"


def route_by_confidence(
    state: GraphState
) -> Literal["output", "hitl"]:
    """
    After confidence_eval_node:
      - No chunks found              → hitl (knowledge gap)
      - Confidence < threshold       → hitl (uncertain answer)
      - Confidence >= threshold      → output (deliver answer)

    Technical queries use a stricter threshold (0.80 vs 0.72).
    """
    confidence = state.get("confidence", 0.0)
    chunks     = state.get("retrieved_chunks") or []
    intent     = state.get("intent", "faq")

    # Hard rule: no context at all → must escalate
    if not chunks:
        state["routing_reason"] = "no_relevant_chunks_found"
        return "hitl"

    threshold = (
        settings.CONFIDENCE_THRESHOLD_TECHNICAL
        if intent == "technical"
        else settings.CONFIDENCE_THRESHOLD
    )

    if confidence >= threshold:
        return "output"

    state["routing_reason"] = (
        f"confidence_{confidence:.3f}_below_threshold_{threshold}"
    )
    return "hitl"


# ══════════════════════════════════════════════════════════════════════════
#  GRAPH COMPILATION
# ══════════════════════════════════════════════════════════════════════════

def build_graph() -> "CompiledStateGraph":
    """
    Assemble and compile the full LangGraph StateGraph.

    Graph edges:
        input_node → intent_router_node
        intent_router_node →[intent] greeting_node | hitl_node | retrieval_node
        retrieval_node → llm_node
        llm_node → confidence_eval_node
        confidence_eval_node →[confidence] output_node | hitl_node
        hitl_node     → output_node
        greeting_node → output_node
        output_node   → END
    """
    g = StateGraph(GraphState)

    # ── Register nodes ───────────────────────────────────────────────────
    g.add_node("input_node",          input_node)
    g.add_node("intent_router_node",  intent_router_node)
    g.add_node("retrieval_node",      retrieval_node)
    g.add_node("llm_node",            llm_node)
    g.add_node("confidence_eval_node", confidence_eval_node)
    g.add_node("hitl_node",           hitl_node)
    g.add_node("greeting_node",       greeting_node)
    g.add_node("output_node",         output_node)

    # ── Entry point ──────────────────────────────────────────────────────
    g.set_entry_point("input_node")

    # ── Fixed edges ──────────────────────────────────────────────────────
    g.add_edge("input_node",          "intent_router_node")
    g.add_edge("retrieval_node",      "llm_node")
    g.add_edge("llm_node",            "confidence_eval_node")
    g.add_edge("hitl_node",           "output_node")
    g.add_edge("greeting_node",       "output_node")
    g.add_edge("output_node",         END)

    # ── Conditional edges ────────────────────────────────────────────────
    g.add_conditional_edges(
        "intent_router_node",
        route_by_intent,
        {
            "retrieval": "retrieval_node",
            "hitl":      "hitl_node",
            "greeting":  "greeting_node",
        }
    )

    g.add_conditional_edges(
        "confidence_eval_node",
        route_by_confidence,
        {
            "output": "output_node",
            "hitl":   "hitl_node",
        }
    )

    compiled = g.compile()
    logger.info("LangGraph compiled successfully")
    return compiled


# ══════════════════════════════════════════════════════════════════════════
#  PUBLIC EXECUTION FUNCTION
# ══════════════════════════════════════════════════════════════════════════

# Lazy singleton
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_graph(
    query:      str,
    session_id: str,
    history:    List[dict] | None = None
) -> GraphState:
    """
    Execute the full RAG pipeline for one user query.

    Args:
        query:      Sanitised user question
        session_id: Unique session identifier
        history:    Prior conversation turns

    Returns:
        Final GraphState after all nodes have executed.
    """
    graph = get_graph()

    initial: GraphState = {
        "query":            query,
        "session_id":       session_id,
        "history":          history or [],
        "intent":           None,
        "retrieved_chunks": None,
        "context_string":   None,
        "llm_result":       None,
        "confidence":       None,
        "final_answer":     None,
        "needs_hitl":       False,
        "hitl_ticket_id":   None,
        "routing_reason":   None,
        "error":            None,
    }

    try:
        final_state: GraphState = graph.invoke(initial)
    except Exception as exc:
        logger.error(f"Graph execution error: {exc}", exc_info=True)
        final_state = {
            **initial,
            "final_answer": (
                "I encountered an internal error processing your request. "
                "Please try again or contact support."
            ),
            "error":      str(exc),
            "confidence": 0.0,
        }

    return final_state
