"""
app/services/hitl.py
─────────────────────────────────────────────────────────────
Human-in-the-Loop (HITL) Service

Responsibilities:
  1. Accept an escalation request from the LangGraph workflow
  2. Persist it to a JSON queue file
  3. Allow human agents to resolve tickets via API
  4. Provide queue status for monitoring

Design notes:
  - JSON file storage (no Redis dependency — swap easily)
  - Thread-safe reads/writes via _lock (single-process safe)
  - Each ticket holds the original query, context preview, and reason
  - Timeout logic: if unresolved after HITL_TIMEOUT_SECONDS → "timeout"
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.config import settings
from app.utils.helpers import get_logger, make_ticket_id

logger = get_logger(__name__)

# ── Storage file ───────────────────────────────────────────────────────────
_QUEUE_FILE = settings.DATA_DIR / "hitl_queue.json"
_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _load() -> Dict[str, dict]:
    """Read the queue from disk. Returns {} if file missing or corrupt."""
    with _lock:
        if not _QUEUE_FILE.exists():
            return {}
        try:
            return json.loads(_QUEUE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}


def _save(queue: Dict[str, dict]) -> None:
    """Persist the queue to disk atomically."""
    with _lock:
        _QUEUE_FILE.write_text(
            json.dumps(queue, indent=2, default=str),
            encoding="utf-8"
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ══════════════════════════════════════════════════════════════════════════
#  HITL SERVICE
# ══════════════════════════════════════════════════════════════════════════

class HITLService:
    """
    Manages the human-review escalation queue.

    Typical flow:
        ticket_id = svc.create_ticket(session_id, query, context, reason)
        # ... agent reviews ...
        svc.resolve_ticket(ticket_id, "Here is the answer.", "agent_1")
    """

    def create_ticket(
        self,
        session_id: str,
        query: str,
        context: str,
        reason: str
    ) -> str:
        """
        Create and persist a new HITL escalation ticket.

        Args:
            session_id: User session that triggered escalation
            query:      Original user query
            context:    Retrieved context string (preview stored)
            reason:     Why escalation was triggered

        Returns:
            ticket_id string (e.g. "tkt_a1b2c3d4")
        """
        ticket_id = make_ticket_id()
        queue     = _load()
        queue[ticket_id] = {
            "ticket_id":       ticket_id,
            "session_id":      session_id,
            "query":           query,
            "context_preview": context[:400] if context else "",
            "reason":          reason,
            "status":          "pending",
            "created_at":      _now_iso(),
            "resolved_at":     None,
            "human_response":  None,
            "agent_id":        None,
        }
        _save(queue)
        logger.info(f"HITL ticket created: {ticket_id} | reason='{reason}'")
        return ticket_id

    def resolve_ticket(
        self,
        ticket_id: str,
        human_response: str,
        agent_id: str = "agent_1"
    ) -> bool:
        """
        Resolve an open ticket with a human answer.

        Returns:
            True if ticket existed and was updated, False otherwise.
        """
        queue = _load()
        if ticket_id not in queue:
            logger.warning(f"Resolve failed — ticket not found: {ticket_id}")
            return False

        queue[ticket_id].update({
            "status":         "resolved",
            "human_response": human_response,
            "agent_id":       agent_id,
            "resolved_at":    _now_iso(),
        })
        _save(queue)
        logger.info(f"HITL ticket resolved: {ticket_id} by {agent_id}")
        return True

    def get_ticket(self, ticket_id: str) -> Optional[dict]:
        """Return one ticket by ID, or None if not found."""
        return _load().get(ticket_id)

    def get_pending(self) -> List[dict]:
        """Return all tickets with status='pending'."""
        return [t for t in _load().values() if t["status"] == "pending"]

    def get_all(self) -> List[dict]:
        """Return all tickets regardless of status."""
        return list(_load().values())

    def poll_for_resolution(
        self,
        ticket_id: str,
        timeout: int | None = None
    ) -> Optional[str]:
        """
        Blocking poll for a human response.
        Checks every 2 seconds until resolved or timeout.

        Args:
            ticket_id: Ticket to poll
            timeout:   Max seconds to wait (defaults to settings.HITL_TIMEOUT_SECONDS)

        Returns:
            human_response string if resolved, None if timeout.
        """
        timeout  = timeout or settings.HITL_TIMEOUT_SECONDS
        deadline = time.time() + timeout

        logger.info(f"Polling for HITL resolution: {ticket_id} (timeout={timeout}s)")
        while time.time() < deadline:
            ticket = self.get_ticket(ticket_id)
            if ticket and ticket["status"] == "resolved":
                logger.info(f"HITL resolved within poll window: {ticket_id}")
                return ticket["human_response"]
            time.sleep(2)

        # Mark as timed out
        queue = _load()
        if ticket_id in queue and queue[ticket_id]["status"] == "pending":
            queue[ticket_id]["status"] = "timeout"
            _save(queue)
        logger.warning(f"HITL poll timeout: {ticket_id}")
        return None

    def stats(self) -> dict:
        all_tickets = self.get_all()
        by_status   = {"pending": 0, "resolved": 0, "timeout": 0}
        for t in all_tickets:
            by_status[t["status"]] = by_status.get(t["status"], 0) + 1
        return {"total": len(all_tickets), **by_status}


# ── Module-level singleton ────────────────────────────────────────────────
_hitl_service: HITLService | None = None

def get_hitl_service() -> HITLService:
    global _hitl_service
    if _hitl_service is None:
        _hitl_service = HITLService()
    return _hitl_service
