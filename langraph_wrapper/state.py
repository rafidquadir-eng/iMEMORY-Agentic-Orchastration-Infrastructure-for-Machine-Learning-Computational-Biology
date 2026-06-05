"""Shared state object threaded through the LangGraph state machine."""
from __future__ import annotations

from typing import Dict, List, Optional, TypedDict


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
    # Discovery pipeline extensions
    pipeline_results: Optional[dict]   # full pipeline output from executor
    report_path: Optional[str]         # path to generated markdown report
    figure_paths: Optional[dict]       # {"pca": path, "recovery": path, ...}
    discovery_task: Optional[bool]     # True if this is a full pipeline run
    _session_id: Optional[str]         # observability session ID
    _task_message: Optional[object]
    _result_message: Optional[object]
