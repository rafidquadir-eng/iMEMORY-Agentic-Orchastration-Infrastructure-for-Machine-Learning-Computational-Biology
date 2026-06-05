"""
agents/orchestrator/planning_agent.py  (v2 — +discovery planning + report generation)

Planning agent: decomposes scientific questions into typed A2A task assignments,
interprets returned results biologically, plans full drug-discovery runs, and
generates structured markdown reports.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
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

_DISCOVERY_PLANNER_SYSTEM = """You are the iMEMORY planning agent specialising in the
full drug-discovery pipeline for the ATNR SLE (trained-immunity) endotype. You emit a
complete pipeline instruction covering: patient cohort embedding via HetGAT, therapeutic
signature computation, LINCS L1000 screening, de novo molecule generation via SELFIES,
iMEMORY recovery scoring, visualisation generation, and report production. You are
explicit about acceptance criteria so the execution agent's output can be audited."""

_REPORT_SYSTEM = """You are a scientific writer for iMEMORY, a deep-learning
target-discovery platform. Write a structured, professional markdown report from
the provided pipeline results. Be quantitative, precise, and include all requested
sections. Use markdown tables where specified. Flag synthetic-data limitations clearly."""


class PlanningAgent:
    def __init__(
        self,
        client: Optional[Anthropic] = None,
        model: str = _DEFAULT_MODEL,
        tracer: Optional[AgentTracer] = None,
    ):
        self.client = client or Anthropic()
        self.model  = model
        self.tracer = tracer or AgentTracer()

    # ── internal: single traced model call ───────────────────────────────────
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
            agent="planning_agent", task=user,
            tokens_in=resp.usage.input_tokens,
            tokens_out=resp.usage.output_tokens,
            latency_ms=latency_ms, audit_passed=True, model=self.model,
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    # ── public API — existing ────────────────────────────────────────────────
    def plan_task(
        self,
        question: str,
        rag_context: str = "",
        expected_outputs: Optional[List[str]] = None,
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

    # ── public API — new: discovery pipeline ────────────────────────────────
    def plan_discovery_task(
        self, cohort_description: str, rag_context: str = ""
    ) -> A2AMessage:
        """Specialised task planner for the full drug discovery pipeline."""
        pipeline_instruction = (
            f"Run the full iMEMORY drug discovery pipeline on the following cohort:\n"
            f"{cohort_description}\n\n"
            f"Domain context:\n{rag_context or '(none)'}\n\n"
            "Pipeline stages (execute in order):\n"
            "  1. build_patient_cohorts — embed diseased and healthy patients via HetGAT\n"
            "  2. compute_therapeutic_signature — Δ, inverse-project, nominate targets\n"
            "  3. screen_lincs — query SigCom LINCS L1000 for disease reversers\n"
            "  4. generate_molecules — de novo candidates via SELFIES + RDKit\n"
            "  5. score_recovery — iMEMORY recovery scoring (cosine angular displacement)\n"
            "  6. generate_visualizations — save 5 matplotlib figures to outputs/figures/\n"
            "\nAcceptance criteria (all must be satisfied):\n"
            "  - nominated_targets: non-empty list of (label, score) pairs\n"
            "  - ranked_candidates: non-empty list with recovery_fraction fields\n"
            "  - figure_paths: dict with at least one path to a saved PNG\n"
        )
        self._call(_DISCOVERY_PLANNER_SYSTEM, pipeline_instruction, max_tokens=512)
        return A2AHandoff.assign_task(
            description=(
                f"Run the full iMEMORY drug discovery pipeline on the ATNR SLE cohort: "
                "embed patients, compute the therapeutic signature, screen LINCS, "
                "generate de novo candidates, score recovery, generate visualizations, "
                "and produce a structured discovery report."
            ),
            code=pipeline_instruction,
            expected_outputs=["targets", "lincs_hits", "ranked_candidates",
                              "figure_paths", "report"],
        )

    def generate_report(
        self,
        pipeline_results: Dict[str, Any],
        interpretation: str = "",
        session_id: str = "",
    ) -> str:
        """Write a full structured markdown report from pipeline results. Returns file path."""
        ts   = datetime.now(timezone.utc).isoformat()
        sid  = session_id or "unknown"
        mode = pipeline_results.get("lincs_mode", "unknown")
        gmode= pipeline_results.get("generation_mode", "unknown")

        targets_md = ""
        for item in pipeline_results.get("nominated_targets", []):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                targets_md += f"| {item[0]} | {item[1]:.4f} |\n"

        lincs_md = ""
        for h in pipeline_results.get("lincs_hits", [])[:10]:
            name = h.get("name", "?")
            z    = h.get("z_sum")
            rev  = h.get("reversal_potential", 0.0)
            smi  = h.get("smiles", "")[:30]
            lincs_md += f"| {name} | {z} | {rev:.4f} | {smi} |\n"

        cands_md = ""
        for c in pipeline_results.get("ranked_candidates", [])[:10]:
            cands_md += (
                f"| {c.get('id','?')} | {c.get('smiles','')[:25]} | "
                f"{c.get('qed',0):.3f} | {c.get('novelty',0):.3f} | "
                f"{c.get('recovery_fraction',0):.4f} |\n"
            )

        figs_md = ""
        for k, v in pipeline_results.get("figure_paths", {}).items():
            figs_md += f"- **{k}**: `{v}`\n"

        best_cand = ""
        ranked = pipeline_results.get("ranked_candidates", [])
        if ranked:
            b = ranked[0]
            best_cand = (
                f"**Best candidate:** `{b.get('id','?')}` — "
                f"recovery={b.get('recovery_fraction',0):.4f}, "
                f"QED={b.get('qed',0):.3f}, "
                f"angle {b.get('angle_before_deg',0):.1f}° → {b.get('angle_after_deg',0):.1f}°\n\n"
                f"SMILES: `{b.get('smiles','')}`"
            )

        prompt = (
            f"Write the following iMEMORY drug discovery report in full markdown.\n\n"
            f"SESSION: {sid}\nTIMESTAMP: {ts}\nLINCS_MODE: {mode}\nGEN_MODE: {gmode}\n\n"
            f"BIOLOGICAL INTERPRETATION:\n{interpretation or '(none provided)'}\n\n"
            f"NOMINATED TARGETS TABLE:\n| Gene | Priority Score |\n|---|---|\n{targets_md}\n\n"
            f"LINCS HITS TABLE:\n| Compound | z_sum | Reversal | SMILES |\n|---|---|---|---|\n{lincs_md}\n\n"
            f"CANDIDATES TABLE:\n| ID | SMILES | QED | Novelty | Recovery |\n|---|---|---|---|---|\n{cands_md}\n\n"
            f"BEST CANDIDATE:\n{best_cand}\n\n"
            f"FIGURES:\n{figs_md}\n\n"
            f"PIPELINE RESULTS SUMMARY:\n"
            f"  converged={pipeline_results.get('converged')}, "
            f"  n_iterations={pipeline_results.get('n_iterations')}, "
            f"  n_targets={len(pipeline_results.get('nominated_targets',[]))}, "
            f"  n_lincs_hits={len(pipeline_results.get('lincs_hits',[]))}, "
            f"  n_candidates={len(pipeline_results.get('ranked_candidates',[]))}\n\n"
            "Produce the full structured report with all required sections. "
            "Be quantitative and include all tables above."
        )
        report_md = self._call(_REPORT_SYSTEM, prompt, max_tokens=3000)

        os.makedirs("./outputs", exist_ok=True)
        report_path = f"./outputs/report_{sid}.md"
        with open(report_path, "w") as f:
            f.write(report_md)

        return report_path
