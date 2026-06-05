"""
The iMEMORY orchestration as a four-node LangGraph StateGraph:

    plan_task -> execute_code -> audit_results -> (replan | interpret_bio | END)

This graph IS the existing two-agent pipeline; LangGraph only formalizes the
control flow and makes the self-correcting replan edge explicit.
"""
from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, StateGraph

from agents.orchestrator import PlanningAgent
from agents.executor import ExecutionAgent
from rag.document_loader import BioRAGPipeline
from .edges import route_after_audit
from .nodes import make_nodes
from .state import AgentState


def create_imemory_graph(
    planner: PlanningAgent,
    executor: ExecutionAgent,
    rag: BioRAGPipeline,
    exec_context: Dict[str, Any],
):
    nodes = make_nodes(planner, executor, rag, exec_context)
    g = StateGraph(AgentState)
    g.add_node("plan_task", nodes["plan_task"])
    g.add_node("execute_code", nodes["execute_code"])
    g.add_node("audit_results", nodes["audit_results"])
    g.add_node("interpret_bio", nodes["interpret_bio"])

    g.set_entry_point("plan_task")
    g.add_edge("plan_task", "execute_code")
    g.add_edge("execute_code", "audit_results")
    g.add_conditional_edges(
        "audit_results",
        route_after_audit,
        {"replan": "plan_task", "interpret": "interpret_bio", END: END},
    )
    g.add_edge("interpret_bio", END)
    return g.compile()
