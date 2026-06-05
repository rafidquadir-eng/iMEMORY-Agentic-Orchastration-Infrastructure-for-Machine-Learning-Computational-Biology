"""
Execution agent.

In production this role is performed by a Sonnet-class coding agent attached to
the iMEMORY codebase inside an IDE (Google Antigravity). It receives a typed A2A
task_assignment, writes and runs the analysis code against the graph store and
data, self-audits the output, and returns a typed result_return message.

In this public showcase the execution agent runs a bounded, deterministic
analysis on synthetic embeddings so the end-to-end loop is demonstrable without
the proprietary codebase. The `_run_analysis` method is the seam where the
Antigravity IDE agent plugs in for real workloads.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict

from anthropic import Anthropic

from agents.a2a import A2AHandoff, A2AMessage
from observability.audit_trail import AuditTrail
from observability.tracer import AgentTracer
from imemory.vector_ops import nominate_targets

_DEFAULT_MODEL = os.environ.get("IMEMORY_EXECUTOR_MODEL", "claude-sonnet-4-20250514")


class ExecutionAgent:
    def __init__(
        self,
        client: Anthropic | None = None,
        model: str = _DEFAULT_MODEL,
        tracer: AgentTracer | None = None,
        audit: AuditTrail | None = None,
    ):
        self.client = client or Anthropic()
        self.model = model
        self.tracer = tracer or AgentTracer()
        self.audit = audit or AuditTrail()

    def execute(self, task: A2AMessage, context: Dict[str, Any] | None = None) -> A2AMessage:
        """Run the assigned task and return a typed result, having self-audited it."""
        context = context or {}
        t0 = time.time()
        try:
            results = self._run_analysis(task, context)
            audit_passed, failures = self.audit.verify(
                results, expected_outputs=task.payload.get("expected_outputs", [])
            )
        except Exception as exc:  # noqa: BLE001 — surface as structured error
            self.tracer.trace(
                agent="execution_agent",
                task=task.task_description,
                tokens_in=0,
                tokens_out=0,
                latency_ms=(time.time() - t0) * 1000,
                audit_passed=False,
                model=self.model,
            )
            return A2AHandoff.raise_error(task, error=str(exc))

        self.tracer.trace(
            agent="execution_agent",
            task=task.task_description,
            tokens_in=context.get("tokens_in", 0),
            tokens_out=context.get("tokens_out", 0),
            latency_ms=(time.time() - t0) * 1000,
            audit_passed=audit_passed,
            model=self.model,
        )
        return A2AHandoff.return_results(task, results=results, audit_passed=audit_passed)

    # -- production seam ----------------------------------------------------
    def _run_analysis(self, task: A2AMessage, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deterministic synthetic analysis for the showcase.

        Replace the body of this method with a call into the Antigravity
        IDE-attached coding agent (or a sandboxed subprocess) for real workloads.
        Here: nominate candidate targets from a synthetic embedding matrix.
        """
        embeddings = context.get("embeddings")
        gene_labels = context.get("gene_labels")
        disease_centroid = context.get("disease_centroid")
        if embeddings is None or gene_labels is None or disease_centroid is None:
            raise ValueError("Execution context missing embeddings/labels/centroid.")

        nominations = nominate_targets(embeddings, gene_labels, disease_centroid, top_k=2)
        return {
            "nominated_targets": nominations,            # e.g. [("Novel Marker A", 0.78), ...]
            "n_genes_scored": len(gene_labels),
            "method": "embedding-distance nomination (synthetic showcase)",
        }
