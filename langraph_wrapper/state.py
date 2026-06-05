"""Shared state object threaded through the LangGraph state machine."""
from __future__ import annotations

from typing import List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    task: str
    planned_code: Optional[str]
    rag_context: Optional[str]
    executed_results: Optional[dict]
    audit_passed: Optional[bool]
    audit_failures: List[str]
    interpretation: Optional[str]
    iteration_count: int
    a2a_messages: List[dict]
