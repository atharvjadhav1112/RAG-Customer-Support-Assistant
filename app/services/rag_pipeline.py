"""
app/services/rag_pipeline.py
─────────────────────────────────────────────────────────────
RAG Pipeline Service

Responsibilities:
  1. Classify the intent of a user query
  2. Build a grounded prompt from retrieved context
  3. Call LLM (OpenAI or smart mock) to generate an answer
  4. Compute a confidence score from retrieval quality + answer tone

Design notes:
  - System prompt enforces grounding — LLM cannot hallucinate
  - Mock LLM extracts top-matching sentences so output is non-trivial
  - Confidence = weighted combination of retrieval score + LLM self-signal
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from app.config import settings
from app.services.retriever import RetrievedChunk
from app.utils.helpers import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  INTENT CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════

# Map of intent → list of trigger regex patterns
_INTENT_RULES: dict[str, List[str]] = {
    "greeting": [
        r"\b(hi|hello|hey|good morning|good afternoon|good evening|greetings)\b"
    ],
    "escalate": [
        r"\b(speak to|talk to|connect me|human|manager|supervisor|real person|agent"
        r"|complaint department|lawsuit|legal action)\b"
    ],
    "complaint": [
        r"\b(broken|terrible|awful|horrible|worst|useless|scam|fraud|angry|furious"
        r"|unacceptable|disgusting|hate|pathetic|ridiculous|never works|waste of money)\b"
    ],
    "technical": [
        r"\b(error|bug|crash|exception|not working|failed|install|configure|setup"
        r"|api|endpoint|integration|debug|code|stack|0x[0-9a-fA-F]+|error code)\b"
    ],
    "faq": [
        r"\b(what|how|when|where|who|why|can|do|does|is|are|will|would|should|could"
        r"|tell me|explain|describe|list)\b"
    ],
}


def classify_intent(query: str) -> str:
    """
    Rule-based intent classification.
    Returns one of: greeting | escalate | complaint | technical | faq
    """
    q = query.lower().strip()
    for intent, patterns in _INTENT_RULES.items():
        for pattern in patterns:
            if re.search(pattern, q, re.IGNORECASE):
                return intent
    return "faq"   # default


# ══════════════════════════════════════════════════════════════════════════
#  PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are a precise and helpful customer support assistant.

STRICT RULES:
1. Answer ONLY using the provided Context section.
2. If the context is insufficient, respond exactly:
   "I don't have enough information to answer this reliably."
3. Be concise and clear (under 200 words unless detail is critical).
4. For multi-step answers, use numbered steps.
5. End with: (Source: <filename>, page <N>) — use the most relevant chunk.
6. NEVER invent facts, prices, dates, or policies not in the context."""


def build_prompt(
    query: str,
    context: str,
    history: List[dict]
) -> List[dict]:
    """
    Construct the messages array for the LLM.

    Args:
        query:   Sanitised user query
        context: Formatted string of retrieved chunks
        history: Last N conversation turns [{role, content}, ...]

    Returns:
        List of OpenAI-format message dicts
    """
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # Include up to last 3 turns for multi-turn coherence
    for turn in history[-3:]:
        if turn.get("role") in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn["content"]})

    user_content = (
        f"Context:\n"
        f"{'─' * 60}\n"
        f"{context}\n"
        f"{'─' * 60}\n\n"
        f"Customer question: {query}\n\n"
        f"Answer based strictly on the context above:"
    )
    messages.append({"role": "user", "content": user_content})
    return messages


def format_context(chunks: List[RetrievedChunk]) -> str:
    """Convert a list of retrieved chunks into a single context string."""
    if not chunks:
        return "No relevant context found."
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Source: {chunk.source}, Page {chunk.page} "
            f"(relevance={chunk.score:.2f})\n{chunk.text}"
        )
    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════
#  LLM CALL
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class LLMResult:
    answer:     str
    model_used: str
    tokens:     int = 0


def call_llm(
    query: str,
    context: str,
    history: List[dict]
) -> LLMResult:
    """
    Generate an answer via OpenAI (if key available) or smart mock.
    """
    if settings.use_openai_llm:
        return _openai_call(query, context, history)
    return _mock_call(query, context)


def _openai_call(query: str, context: str, history: List[dict]) -> LLMResult:
    """Real OpenAI GPT call."""
    try:
        from openai import OpenAI
        client   = OpenAI(api_key=settings.OPENAI_API_KEY)
        messages = build_prompt(query, context, history)

        resp   = client.chat.completions.create(
            model       = settings.LLM_MODEL,
            messages    = messages,
            temperature = settings.LLM_TEMPERATURE,
            max_tokens  = settings.LLM_MAX_TOKENS,
        )
        return LLMResult(
            answer     = resp.choices[0].message.content.strip(),
            model_used = settings.LLM_MODEL,
            tokens     = resp.usage.total_tokens
        )
    except Exception as exc:
        logger.error(f"OpenAI call failed: {exc} — falling back to mock")
        return _mock_call(query, context)


def _mock_call(query: str, context: str) -> LLMResult:
    """
    Smart mock LLM — no API key required.
    Scores each sentence in the context by word-overlap with the query,
    picks the top 3, and stitches them into a coherent answer.
    """
    if not context or "No relevant context found" in context:
        return LLMResult(
            answer=(
                "I don't have enough information to answer this reliably. "
                "Let me connect you with a support agent who can help further."
            ),
            model_used="smart-mock"
        )

    # Strip the metadata prefix lines from context
    clean_context = re.sub(r"\[\d+\] Source:.*?\n", "", context)

    # Tokenise query (remove stop words)
    stop = {"what","is","the","a","an","how","do","does","can","i","my",
            "your","?","about","of","to","in","for","on","with","are","was"}
    q_words = {w.lower() for w in query.split()} - stop

    # Score sentences
    scored: list[tuple[float, str]] = []
    for sent in re.split(r"(?<=[.!?])\s+", clean_context):
        sent = sent.strip()
        if len(sent) < 25:
            continue
        s_words = {w.lower() for w in sent.split()}
        if not s_words:
            continue
        overlap = len(q_words & s_words)
        score   = overlap / max(len(q_words), 1)
        if score > 0:
            scored.append((score, sent))

    scored.sort(key=lambda x: -x[0])
    top = [s for _, s in scored[:4]]

    if not top:
        # Fall back to first two sentences of context
        raw_sents = re.split(r"(?<=[.!?])\s+", clean_context)
        top = [s.strip() for s in raw_sents[:2] if len(s.strip()) > 25]

    if not top:
        return LLMResult(
            answer="I found some context but couldn't extract a clear answer. A support agent will assist you.",
            model_used="smart-mock"
        )

    answer = " ".join(top)

    # Append source citation from first chunk marker
    src_match = re.search(r"\[1\] Source: ([^,]+), Page (\d+)", context)
    if src_match:
        answer += f"\n\n(Source: {src_match.group(1)}, page {src_match.group(2)})"

    return LLMResult(answer=answer, model_used="smart-mock (set OPENAI_API_KEY for GPT)")


# ══════════════════════════════════════════════════════════════════════════
#  CONFIDENCE SCORING
# ══════════════════════════════════════════════════════════════════════════

_UNCERTAINTY_PHRASES = [
    "don't have enough", "cannot answer", "not sure", "unclear",
    "no information", "couldn't find", "unable to", "insufficient"
]


def compute_confidence(
    chunks: List[RetrievedChunk],
    answer: str
) -> float:
    """
    Estimate answer confidence on [0.0, 1.0].

    Formula:
        confidence = 0.6 * top_score + 0.4 * avg_score − uncertainty_penalty

    Args:
        chunks: Retrieved chunks used to generate the answer
        answer: The generated answer text

    Returns:
        Float in [0.0, 1.0]
    """
    if not chunks:
        return 0.0

    top_score = max(c.score for c in chunks)
    avg_score = sum(c.score for c in chunks) / len(chunks)

    # Penalise if LLM admitted uncertainty
    penalty = 0.25 if any(p in answer.lower() for p in _UNCERTAINTY_PHRASES) else 0.0

    score = 0.60 * top_score + 0.40 * avg_score - penalty
    return round(max(0.0, min(1.0, score)), 4)
