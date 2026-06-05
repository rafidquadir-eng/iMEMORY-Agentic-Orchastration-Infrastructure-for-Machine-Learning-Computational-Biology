"""
The iMEMORY orchestration as a six-node LangGraph StateGraph:

    plan_task -> execute_pipeline -> audit_results
      -> (replan: plan_task | pass: interpret_bio)
      -> interpret_bio -> generate_report -> generate_viz -> END

The conditional edge from audit_results is unchanged from v1. The three new
downstream nodes (interpret, report, viz) extend the linear success path.
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

    g.add_node("plan_task",        nodes["plan_task"])
    g.add_node("execute_pipeline", nodes["execute_pipeline"])
    g.add_node("audit_results",    nodes["audit_results"])
    g.add_node("interpret_bio",    nodes["interpret_bio"])
    g.add_node("generate_report",  nodes["generate_report"])
    g.add_node("generate_viz",     nodes["generate_viz"])

    g.set_entry_point("plan_task")
    g.add_edge("plan_task",        "execute_pipeline")
    g.add_edge("execute_pipeline", "audit_results")
    g.add_conditional_edges(
        "audit_results",
        route_after_audit,
        {"replan": "plan_task", "interpret": "interpret_bio", END: END},
    )
    g.add_edge("interpret_bio",   "generate_report")
    g.add_edge("generate_report", "generate_viz")
    g.add_edge("generate_viz",    END)

    return g.compile()
