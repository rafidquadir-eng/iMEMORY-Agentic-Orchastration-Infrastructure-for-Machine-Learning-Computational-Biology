"""
demo/demo_agentic_pipeline.py

Primary demo: the FULL iMEMORY drug discovery pipeline orchestrated end-to-end
by the agent infrastructure.

    PlanningAgent (Claude Opus) decomposes the task
      -> ExecutionAgent (Claude Sonnet) runs tool_use loop:
           build_patient_cohorts -> compute_therapeutic_signature
           -> screen_lincs -> generate_molecules -> score_recovery
           -> generate_visualizations
      -> LangGraph: audit -> interpret -> generate_report -> generate_viz
      -> Outputs: markdown report + 5 PNG figures

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python demo/demo_agentic_pipeline.py

No proprietary weights, no real patient data — all computation is synthetic or
uses public APIs (SigCom LINCS L1000, SELFIES/RDKit).
"""
from __future__ import annotations

import os
import sys

import numpy as np

# ── Output directories ───────────────────────────────────────────────────────
os.makedirs("./outputs/figures", exist_ok=True)

# ── Import full stack ────────────────────────────────────────────────────────
from agents.orchestrator import PlanningAgent
from agents.executor import ExecutionAgent
from rag.document_loader import BioRAGPipeline
from langraph_wrapper.graph import create_imemory_graph
from observability.tracer import AgentTracer

from data.synthetic.generate_lincs_library import generate_library
from discovery.lincs_connectivity import inject_planted_reversers
from imemory.hetgat_ops import init_random_weights, multi_head_hetgat_update
from imemory.graph_builder import build_synthetic_patient_graph, MODALITIES
from imemory.therapeutic_signature import compute_therapeutic_delta, inverse_project

# ── Gene panel (128 real HGNC symbols) — identical to demo_full_pipeline ────
_RAW_GENES = [
    "TLR2", "BCL2", "EZH2",
    "STAT1", "STAT2", "IRF7", "IRF9", "MX1", "MX2", "OAS1", "OAS2", "OAS3", "OASL",
    "ISG15", "IFI6", "IFI27", "IFI44", "IFI44L", "IFIT1", "IFIT2", "IFIT3",
    "IFITM1", "IFITM3", "RSAD2", "USP18", "HERC5", "HERC6", "DDX58", "IFIH1",
    "XAF1", "SIGLEC1", "LY6E", "EIF2AK2", "GBP1", "GBP5", "CXCL10", "CXCL9",
    "CCL2", "CCL5", "IL6", "IL10", "IL1B", "TNF", "IFNG", "IFNA1", "IFNB1",
    "TLR4", "TLR7", "TLR9", "MYD88", "NFKB1", "RELA", "NLRP3", "CASP1",
    "C1QA", "C1QB", "C3", "C4A", "CR2", "FCGR1A", "FCGR2A", "FCGR3A",
    "CD19", "MS4A1", "CD27", "CD38", "CD40", "CD40LG", "CD80", "CD86",
    "BLK", "BANK1", "TNFSF13B", "TNFRSF13B", "PRDM1", "XBP1", "IRF4", "IRF5",
    "IRF8", "AICDA", "CD14", "CD68", "ITGAM", "ITGAX", "CSF1R", "FCN1",
    "S100A8", "S100A9", "S100A12", "LYZ", "CD163", "MRC1", "CLEC7A", "NOD2",
    "TREM1", "TREM2", "HLA-DRA", "HLA-DRB1", "CD274", "PDCD1", "CTLA4", "FOXP3",
    "IL2RA", "CD3D", "CD3E", "CD4", "CD8A", "GZMB", "PRF1", "TBX21", "GATA3",
    "RORC", "IFNAR1", "IFNAR2", "JAK1", "JAK2", "TYK2", "SOCS1", "SOCS3",
    "PTPN22", "KMT2A", "EP300", "CREBBP", "DNMT1", "DNMT3A", "TET2", "HDAC1",
    "SETD2",
]
GENE_PANEL = _RAW_GENES[:128]
assert len(GENE_PANEL) == 128

_PLANTED_SEED_SMILES = [
    "c1ccc2c(c1)nc(n2)NC(=O)C",
    "O=C(N)c1ccc(cc1)S(=O)(=O)N",
    "c1ccc(cc1)C(=O)Nc1ccncc1",
    "C1CCN(CC1)C(=O)c1ccccc1",
    "Nc1nc(cs1)c1ccccc1",
]


def _embed(graph, weights) -> np.ndarray:
    h_i        = graph["nodes"]["patient"]
    neighbor_h = [graph["nodes"][m] for m in MODALITIES]
    return multi_head_hetgat_update(
        h_i, neighbor_h,
        weights["W_tau_self"], weights["W_tau_neighbor"], weights["a_tau"],
    )


def build_cohorts(n_diseased: int = 30, n_healthy: int = 30):
    weights  = init_random_weights(n_heads=4, dim=128, seed=42)
    diseased = np.stack([
        _embed(build_synthetic_patient_graph(f"D-{i:03d}", seed=1000 + i), weights)
        for i in range(n_diseased)
    ])
    healthy  = np.stack([
        _embed(build_synthetic_patient_graph(f"H-{i:03d}", seed=5000 + i), weights)
        for i in range(n_healthy)
    ])
    rng  = np.random.default_rng(3)
    axis = rng.normal(size=128).astype("float32")
    axis /= np.linalg.norm(axis)
    diseased = diseased + 3.0 * axis
    return diseased, healthy, weights["W_tau_self"][0]


def main() -> None:
    print("=" * 62)
    print("  iMEMORY Agentic Pipeline — Full Orchestrated Demo")
    print("=" * 62)

    # 1. Build cohorts + synthetic LINCS library (pipeline seed data)
    print("\n[1/4] Building synthetic patient cohorts via HetGAT...")
    diseased, healthy, W_tau = build_cohorts(n_diseased=30, n_healthy=30)
    delta                    = compute_therapeutic_delta(
        diseased.mean(axis=0), healthy.mean(axis=0)
    )
    disease_feature_sig = inverse_project(delta, W_tau)

    print("[2/4] Preparing synthetic LINCS fallback library (30,000 cpds)...")
    library = generate_library(n=30_000, seed=99)
    inject_planted_reversers(
        library, disease_feature_sig, n=5, seed_smiles=_PLANTED_SEED_SMILES
    )

    # 2. Wire the agent infrastructure
    print("[3/4] Wiring agents and LangGraph...")
    tracer   = AgentTracer()
    planner  = PlanningAgent(tracer=tracer)
    executor = ExecutionAgent(tracer=tracer)
    rag      = BioRAGPipeline()

    exec_context = {
        "gene_labels":          GENE_PANEL,
        "W_tau":                W_tau,
        "synthetic_library":    library,
        "diseased_embeddings":  diseased,
        "healthy_embeddings":   healthy,
        "n_diseased":           30,
        "n_healthy":            30,
    }

    app = create_imemory_graph(planner, executor, rag, exec_context)

    # 3. Run the graph
    print("[4/4] Invoking LangGraph (plan → execute → audit → interpret → report → viz)...\n")
    initial_state = {
        "task": (
            "Run the full iMEMORY drug discovery pipeline on the ATNR SLE cohort: "
            "embed patients, compute the therapeutic signature, screen LINCS, "
            "generate de novo candidates, score recovery, generate visualizations, "
            "and produce a structured discovery report."
        ),
        "discovery_task":  True,
        "iteration_count": 0,
        "a2a_messages":    [],
        "audit_failures":  [],
        "_session_id":     tracer.session_id,
    }

    final = app.invoke(initial_state)

    # 4. Print results
    print("\n" + "=" * 62)
    print("  PIPELINE COMPLETE")
    print("=" * 62)

    print("\n=== A2A MESSAGE LOG ===")
    for rec in final.get("a2a_messages", []):
        if "sender_id" in rec:
            print(f"  {rec['sender_id']} -> {rec['receiver_id']} [{rec['message_type']}]")
        elif "event" in rec:
            print(f"  [event] {rec['event']}")

    print("\n=== REPORT ===")
    report_path = final.get("report_path")
    print(f"  {report_path}")

    print("\n=== FIGURES ===")
    fig_paths = final.get("figure_paths") or {}
    if fig_paths:
        for k, p in fig_paths.items():
            print(f"  {k}: {p}")
    else:
        print("  (no figures generated)")

    print("\n=== OBSERVABILITY SUMMARY ===")
    summary = tracer.session_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print(f"\n✓ Full orchestrated pipeline complete.")
    if report_path:
        print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
