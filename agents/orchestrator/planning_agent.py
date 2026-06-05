"""
Planning agent (the orchestrator).

Responsibilities:
  1. Decompose a scientific question into a concrete analysis task.
  2. Pull relevant domain context from the RAG layer.
  3. Emit a typed A2A task_assignment for the execution agent.
  4. Interpret returned results biologically.

The agent is built directly on the Anthropic Messages API. Every model call is
wrapped by the observability tracer so cost and latency are captured.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from anthropic import Anthropic

from agents.a2a import A2AHandoff, A2AMessage
from observability.tracer import AgentTracer

_DEFAULT_MODEL = os.environ.get("IMEMORY_ORCHESTRATOR_MODEL", "claude-opus-4-20250514")

_PLANNER_SYSTEM = """You are the planning agent for iMEMORY, a computational-biology
target-discovery platform. You decompose a scientific question into a single, concrete,
executable analysis task for a coding agent. You think in terms of graph-tensor inputs,
embedding-space operations, and falsifiable readouts. You are precise about expected
outputs so the result can be audited automatically. You never write the full
implementation yourself — you specify the task and the acceptance criteria."""

_INTERPRET_SYSTEM = """You are the planning agent interpreting returned analysis results
for iMEMORY. Summarize what the result means biologically, state whether it supports the
hypothesis, and flag any caveat a reviewer should know. Be concise and rigorous."""


class PlanningAgent:
    def __init__(
        self,
        client: Optional[Anthropic] = None,
        model: str = _DEFAULT_MODEL,
        tracer: Optional[AgentTracer] = None,
    ):
        self.client = client or Anthropic()
        self.model = model
        self.tracer = tracer or AgentTracer()

    # -- internal: a single traced model call -------------------------------
    def _call(self, system: str, user: str, max_tokens: int = 1024) -> str:
        t0 = time.time()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency_ms = (time.time() - t0) * 1000
        self.tracer.trace(
            agent="planning_agent",
            task=user,
            tokens_in=resp.usage.input_tokens,
            tokens_out=resp.usage.output_tokens,
            latency_ms=latency_ms,
            audit_passed=True,
            model=self.model,
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    # -- public API ---------------------------------------------------------
    def plan_task(
        self, question: str, rag_context: str = "", expected_outputs: Optional[List[str]] = None
    ) -> A2AMessage:
        """Turn a scientific question into a typed task_assignment for the executor."""
        prompt = (
            f"Scientific question:\n{question}\n\n"
            f"Relevant domain context (retrieved):\n{rag_context or '(none)'}\n\n"
            "Produce: (1) a one-paragraph task description, and (2) a short Python analysis "
            "plan the coding agent should implement. Keep the plan deterministic and testable."
        )
        plan = self._call(_PLANNER_SYSTEM, prompt)
        return A2AHandoff.assign_task(
            description=question,
            code=plan,
            expected_outputs=expected_outputs or ["results_table.csv", "summary.json"],
        )

    def interpret(self, result_message: A2AMessage) -> str:
        """Interpret an execution agent's returned results."""
        results = result_message.payload.get("results", {})
        prompt = (
            f"Task: {result_message.task_description}\n\n"
            f"Returned results (structured): {results}\n\n"
            "Interpret biologically and state support for the hypothesis."
        )
        return self._call(_INTERPRET_SYSTEM, prompt, max_tokens=600)
