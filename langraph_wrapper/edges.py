"""Conditional routing for the audit node."""
from __future__ import annotations

from langgraph.graph import END

MAX_ITERATIONS = 3


def route_after_audit(state: dict) -> str:
    """Replan on audit failure (bounded), interpret on success, else end."""
    if state.get("audit_passed"):
        return "interpret"
    if state.get("iteration_count", 0) < MAX_ITERATIONS:
        return "replan"
    return END
