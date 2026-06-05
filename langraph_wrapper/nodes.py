"""
Node implementations for the iMEMORY StateGraph.

Each node wraps existing agent logic — the graph formalizes structure, it does
not reimplement the agents. Nodes are constructed with the live agents and the
execution context via `make_nodes`, keeping the graph definition clean.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

from agents.orchestrator import PlanningAgent
from agents.executor import ExecutionAgent
from rag.document_loader import BioRAGPipeline
from rag.retriever import format_context


def make_nodes(
    planner: PlanningAgent,
    executor: ExecutionAgent,
    rag: BioRAGPipeline,
    exec_context: Dict[str, Any],
) -> Dict[str, Callable[[dict], dict]]:

    def plan_task_node(state: dict) -> dict:
        chunks = rag.retrieve(state["task"], n=5)
        ctx = format_context(chunks)
        msg = planner.plan_task(state["task"], rag_context=ctx)
        state["rag_context"] = ctx
        state["planned_code"] = msg.payload["code"]
        state.setdefault("a2a_messages", []).append(msg.to_log_record())
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        state["_task_message"] = msg  # carried for the executor node
        return state

    def execute_code_node(state: dict) -> dict:
        task_msg = state["_task_message"]
        result_msg = executor.execute(task_msg, context=exec_context)
        state["executed_results"] = result_msg.payload.get("results", {})
        state["audit_passed"] = result_msg.payload.get("audit_passed", False)
        state["a2a_messages"].append(result_msg.to_log_record())
        state["_result_message"] = result_msg
        return state

    def audit_results_node(state: dict) -> dict:
        # Audit already ran inside the executor; surface failures into state.
        if not state.get("audit_passed"):
            state.setdefault("audit_failures", []).append(
                f"iteration {state.get('iteration_count')} failed structural audit"
            )
        return state

    def interpret_bio_node(state: dict) -> dict:
        state["interpretation"] = planner.interpret(state["_result_message"])
        return state

    return {
        "plan_task": plan_task_node,
        "execute_code": execute_code_node,
        "audit_results": audit_results_node,
        "interpret_bio": interpret_bio_node,
    }
