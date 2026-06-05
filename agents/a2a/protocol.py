"""
Agent-to-Agent (A2A) message contract.

Every exchange between the planning agent and the execution agent is a typed,
validated A2AMessage. No raw strings cross an agent boundary. Each message is
conversation-scoped so an entire pipeline run can be reconstructed from the log,
which is a hard requirement for audit in regulated environments.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

AgentId = Literal["planning_agent", "execution_agent"]
MessageType = Literal[
    "task_assignment",   # planner -> executor: do this analysis
    "result_return",     # executor -> planner: here are the results
    "audit_request",     # planner -> executor: re-verify this output
    "audit_response",    # executor -> planner: verification result
    "error",             # either direction: a failure occurred
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class A2AMessage(BaseModel):
    """A single, validated message exchanged between agents."""

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=_utcnow)

    sender_id: AgentId
    receiver_id: AgentId
    message_type: MessageType

    task_description: str
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Code, results, file paths, audit findings — structured, never free text.",
    )

    audit_required: bool = True
    conversation_id: str = Field(
        ..., description="Ties every message in one pipeline run together."
    )
    parent_message_id: Optional[str] = None

    def child(
        self,
        *,
        sender_id: AgentId,
        receiver_id: AgentId,
        message_type: MessageType,
        task_description: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        audit_required: bool = True,
    ) -> "A2AMessage":
        """Create a reply that inherits this message's conversation lineage."""
        return A2AMessage(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            task_description=task_description or self.task_description,
            payload=payload or {},
            audit_required=audit_required,
            conversation_id=self.conversation_id,
            parent_message_id=self.message_id,
        )

    def to_log_record(self) -> Dict[str, Any]:
        """Flat, JSON-serializable representation for the observability layer."""
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp.isoformat(),
            "conversation_id": self.conversation_id,
            "parent_message_id": self.parent_message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type,
            "task_description": self.task_description[:200],
            "payload_keys": sorted(self.payload.keys()),
            "audit_required": self.audit_required,
        }
