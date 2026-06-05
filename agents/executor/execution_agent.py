"""
agents/executor/execution_agent.py  (v2 — agentic tool-use loop)

The ExecutionAgent now runs the full drug discovery pipeline via Anthropic
API tool_use. Claude autonomously decides which pipeline stages to call and
in what order; each tool call is routed to the real pipeline functions via
dispatch_tool. Results flow back through A2A to the planning agent.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from anthropic import Anthropic

from agents.a2a import A2AHandoff, A2AMessage
from agents.tools.discovery_tools import (
    PIPELINE_TOOLS,
    PipelineContext,
    dispatch_tool,
)
from observability.audit_trail import AuditTrail
from observability.tracer import AgentTracer

_DEFAULT_MODEL = os.environ.get("IMEMORY_EXECUTOR_MODEL", "claude-sonnet-4-20250514")

EXECUTOR_SYSTEM_PROMPT = (
    "You are the iMEMORY execution agent. You run a drug discovery pipeline for "
    "ATNR lupus nephritis by calling the available tools in the correct scientific "
    "order: first build cohorts, then compute the therapeutic signature, then screen "
    "LINCS, then generate molecules, then score recovery, then generate visualizations. "
    "Always call all tools unless the task explicitly says otherwise. "
    "Report your findings after completing the pipeline."
)


def _context_to_results(ctx: PipelineContext) -> Dict[str, Any]:
    """Flatten PipelineContext into a JSON-serializable results dict."""
    targets = []
    if ctx.targets:
        for item in ctx.targets:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                targets.append((str(item[0]), float(item[1])))
            else:
                targets.append(str(item))

    lincs_hits = []
    for h in ctx.lincs_hits:
        lincs_hits.append({
            "name":              str(getattr(h, "pert_name", getattr(h, "compound_id", "?"))),
            "reversal_potential": float(getattr(h, "reversal_potential", 0.0)),
            "z_sum":             float(getattr(h, "z_sum", 0.0)) if getattr(h, "z_sum", None) is not None else None,
            "smiles":            str(getattr(h, "smiles", getattr(h, "smiles_seed", ""))),
        })

    candidates = []
    for sc in ctx.scored_candidates:
        candidates.append({
            "id":               sc.candidate_id,
            "smiles":           sc.smiles,
            "recovery_fraction": float(sc.recovery.recovery_fraction),
            "qed":              float(sc.qed),
            "novelty":          float(sc.novelty),
            "angle_before_deg": float(sc.recovery.mean_angle_before_deg),
            "angle_after_deg":  float(sc.recovery.mean_angle_after_deg),
        })

    return {
        "nominated_targets":  targets,
        "lincs_hits":         lincs_hits,
        "lincs_mode":         ctx.lincs_mode,
        "generation_mode":    ctx.generation_mode,
        "ranked_candidates":  candidates,
        "converged":          ctx.converged,
        "n_iterations":       len(ctx.iterations),
        "figure_paths":       ctx.figure_paths,
    }


class ExecutionAgent:
    def __init__(
        self,
        client: Optional[Anthropic] = None,
        model: str = _DEFAULT_MODEL,
        tracer: Optional[AgentTracer] = None,
        audit: Optional[AuditTrail] = None,
    ):
        self.client = client or Anthropic()
        self.model  = model
        self.tracer = tracer or AgentTracer()
        self.audit  = audit or AuditTrail()

    def execute(self, task: A2AMessage, context: Optional[Dict[str, Any]] = None) -> A2AMessage:
        """Run the assigned task via the agentic tool-use loop. Returns typed A2A message."""
        context = context or {}
        t0 = time.time()

        # Build PipelineContext from whatever the caller passed in.
        pipeline_ctx = PipelineContext(
            gene_labels      = context.get("gene_labels", []),
            W_tau            = context.get("W_tau"),
            synthetic_library = context.get("synthetic_library"),
        )
        # Pre-fill embeddings if supplied (avoids redundant rebuild in tool).
        if "diseased_embeddings" in context:
            pipeline_ctx.diseased_embeddings = context["diseased_embeddings"]
        if "healthy_embeddings" in context:
            pipeline_ctx.healthy_embeddings  = context["healthy_embeddings"]

        total_in, total_out = 0, 0
        try:
            results, total_in, total_out = self._run_agentic_pipeline(task, pipeline_ctx)
            audit_passed, failures = self.audit.verify(
                results,
                expected_outputs=task.payload.get("expected_outputs", []),
            )
        except Exception as exc:  # noqa: BLE001
            self.tracer.trace(
                agent="execution_agent", task=task.task_description,
                tokens_in=0, tokens_out=0,
                latency_ms=(time.time() - t0) * 1000,
                audit_passed=False, model=self.model,
            )
            return A2AHandoff.raise_error(task, error=str(exc))

        self.tracer.trace(
            agent="execution_agent", task=task.task_description,
            tokens_in=total_in, tokens_out=total_out,
            latency_ms=(time.time() - t0) * 1000,
            audit_passed=audit_passed, model=self.model,
        )
        return A2AHandoff.return_results(task, results=results, audit_passed=audit_passed)

    def _run_agentic_pipeline(
        self, task: A2AMessage, ctx: PipelineContext
    ) -> tuple[Dict[str, Any], int, int]:
        """Anthropic tool-use agentic loop. Returns (results_dict, tokens_in, tokens_out)."""
        messages = [{"role": "user", "content": task.task_description}]
        total_in, total_out = 0, 0

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=EXECUTOR_SYSTEM_PROMPT,
                tools=PIPELINE_TOOLS,
                messages=messages,
            )
            total_in  += response.usage.input_tokens
            total_out += response.usage.output_tokens

            # Extend conversation with assistant turn.
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                break

            tool_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_blocks:
                break

            tool_results = []
            for block in tool_blocks:
                t0_tool = time.time()
                try:
                    result  = dispatch_tool(block.name, block.input, ctx)
                    content = json.dumps(result, default=str)
                except Exception as exc:  # noqa: BLE001
                    content = json.dumps({"error": str(exc)})
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     content,
                })
                self.tracer.trace(
                    agent=f"execution_agent.{block.name}",
                    task=block.name,
                    tokens_in=0, tokens_out=0,
                    latency_ms=(time.time() - t0_tool) * 1000,
                    audit_passed=True, model=self.model,
                )

            messages.append({"role": "user", "content": tool_results})

        return _context_to_results(ctx), total_in, total_out


# ---------------------------------------------------------------------------
# Extend AuditTrail to check discovery-specific outputs
# ---------------------------------------------------------------------------
class AuditTrail(AuditTrail):  # type: ignore[no-redef]
    def verify(self, results, expected_outputs=None):
        failures = []
        if not isinstance(results, dict) or not results:
            return False, ["results_empty_or_malformed"]

        if "nominated_targets" in results and not results["nominated_targets"]:
            failures.append("no_targets_nominated")

        if "ranked_candidates" in results and not results["ranked_candidates"]:
            failures.append("no_candidates_scored")

        if "figure_paths" in results and not results["figure_paths"]:
            failures.append("no_visualizations_generated")

        # Legacy check
        if results.get("n_genes_scored", -1) == 0:
            failures.append("zero_genes_scored")

        return (len(failures) == 0), failures
