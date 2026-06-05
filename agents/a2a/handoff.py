"""
Handoff helpers that construct correctly-typed A2A messages for the two
canonical transitions: planner assigning work, and executor returning results.
Centralizing construction here guarantees the message contract is used uniformly.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List

from .protocol import A2AMessage


class A2AHandoff:
    """Factory for the standard planner<->executor message exchanges."""

    @staticmethod
    def assign_task(
        description: str,
        code: str,
        expected_outputs: List[str],
        conversation_id: str | None = None,
    ) -> A2AMessage:
        """Planning agent -> execution agent."""
        return A2AMessage(
            sender_id="planning_agent",
            receiver_id="execution_agent",
            message_type="task_assignment",
            task_description=description,
            payload={"code": code, "expected_outputs": expected_outputs},
            conversation_id=conversation_id or str(uuid.uuid4()),
        )

    @staticmethod
    def return_results(
        parent: A2AMessage, results: Dict[str, Any], audit_passed: bool
    ) -> A2AMessage:
        """Execution agent -> planning agent."""
        return parent.child(
            sender_id="execution_agent",
            receiver_id="planning_agent",
            message_type="result_return",
            payload={"results": results, "audit_passed": audit_passed},
            audit_required=False,
        )

    @staticmethod
    def raise_error(parent: A2AMessage, error: str, sender: str = "execution_agent") -> A2AMessage:
        """Either agent -> the other: structured error report."""
        return parent.child(
            sender_id=sender,            # type: ignore[arg-type]
            receiver_id="planning_agent" if sender == "execution_agent" else "execution_agent",
            message_type="error",
            payload={"error": error},
            audit_required=False,
        )
