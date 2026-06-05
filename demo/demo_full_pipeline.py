"""
demo/demo_full_pipeline.py

End-to-end closed-loop drug discovery demo, wired to REAL components:

    HetGAT embedding (synthetic patients, synthetic weights)
      -> therapeutic signature (Δ, inverse projection, target nomination)
      -> REAL LINCS L1000 screen via SigCom API (synthetic fallback if offline)
      -> REAL de novo generation via SELFIES+RDKit (fallback if not installed)
      -> iMEMORY recovery scoring (cosine angular displacement; read-across effects)
      -> rank -> closed loop (refine if recovery below threshold)

The 128 features are REAL HGNC immune/SLE gene symbols, so the LINCS query
returns real reverser compounds. Only the iMEMORY weights and patient expression
values are synthetic (proprietary components excluded from the public repo).
The demo prints which real paths fired (lincs_mode, generation_mode).

Optional deps for the fully-real path:
    pip install requests selfies rdkit
Run:
    python demo/demo_full_pipeline.py
"""
from __future__ import annotations

import numpy as np

from imemory.hetgat_ops import init_random_weights, multi_head_hetgat_update
from imemory.graph_builder import build_synthetic_patient_graph, MODALITIES
from imemory.therapeutic_signature import compute_therapeutic_delta, inverse_project
from discovery.closed_loop import ClosedLoopDiscovery
from discovery.lincs_connectivity import inject_planted_reversers
from data.synthetic.generate_lincs_library import generate_library

# --- Real 128-gene immune / SLE / interferon / epigenetic panel (HGNC) -------
# First three are the iMEMORY-nominated targets so nomination surfaces them.
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
assert len(GENE_PANEL) == 128, f"Gene panel must be 128, got {len(GENE_PANEL)}"

_PLANTED_SEED_SMILES = [
    "c1ccc2c(c1)nc(n2)NC(=O)C",
    "O=C(N)c1ccc(cc1)S(=O)(=O)N",
    "c1ccc(cc1)C(=O)Nc1ccncc1",
    "C1CCN(CC1)C(=O)c1ccccc1",
    "Nc1nc(cs1)c1ccccc1",
]


def _embed(graph, weights) -> np.ndarray:
    h_i = graph["nodes"]["patient"]
    neighbor_h = [graph["nodes"][m] for m in MODALITIES]
    return multi_head_hetgat_update(
        h_i, neighbor_h,
        weights["W_tau_self"], weights["W_tau_neighbor"], weights["a_tau"],
    )


def build_cohorts(n_diseased: int = 30, n_healthy: int = 30):
    weights = init_random_weights(n_heads=4, dim=128, seed=42)
    diseased = np.stack([_embed(build_synthetic_patient_graph(f"D-{i:03d}", seed=1000 + i), weights)
                         for i in range(n_diseased)])
    healthy = np.stack([_embed(build_synthetic_patient_graph(f"H-{i:03d}", seed=5000 + i), weights)
                        for i in range(n_healthy)])
    # Impose a disease axis so the therapeutic signature is non-trivial.
    rng = np.random.default_rng(3)
    axis = rng.normal(size=128).astype("float32")
    axis /= np.linalg.norm(axis)
    diseased = diseased + 3.0 * axis
    return diseased, healthy, weights["W_tau_self"][0]


def main() -> None:
    print("Building synthetic cohorts via HetGAT forward pass...")
    diseased, healthy, W_tau = build_cohorts()

    # Disease feature signature (for synthetic-fallback planted reversers).
    delta = compute_therapeutic_delta(diseased.mean(axis=0), healthy.mean(axis=0))
    disease_feature_signature = inverse_project(delta, W_tau)

    print("Preparing synthetic LINCS fallback library (used only if API offline)...")
    library = generate_library(n=30_000, seed=99)
    inject_planted_reversers(
        library, disease_feature_signature, n=5, seed_smiles=_PLANTED_SEED_SMILES
    )

    print("Running closed-loop discovery (attempting REAL SigCom LINCS API)...\n")
    pipeline = ClosedLoopDiscovery(
        W_tau=W_tau,
        gene_labels=GENE_PANEL,
        recovery_threshold=0.5,
        max_iterations=3,
    )
    result = pipeline.run(
        diseased_embeddings=diseased,
        healthy_embeddings=healthy,
        synthetic_library=library,
        top_targets=5,
        top_hits=10,
        candidates_per_round=8,
    )

    print(f"LINCS mode:      {result.lincs_mode}      (real = live SigCom API)")
    print(f"Generation mode: {result.generation_mode}  (selfies = real SELFIES+RDKit)\n")

    print("=== NOMINATED TARGETS (inverse-projected Δ) ===")
    for label, score in result.nominated_targets:
        print(f"  {label:12s}  priority={score:.3f}")

    print("\n=== TOP LINCS REVERSAL HITS ===")
    for h in result.lincs_hits[:5]:
        name = getattr(h, "pert_name", getattr(h, "compound_id", "?"))
        rev = getattr(h, "reversal_potential", 0.0)
        z = getattr(h, "z_sum", None)
        ztxt = f"  zSum={z:+.3f}" if z is not None else ""
        print(f"  {str(name):28s}  reversal={rev:.3f}{ztxt}")

    print("\n=== CLOSED-LOOP ITERATIONS ===")
    for it in result.iterations:
        print(f"  iter {it.iteration}: {it.n_candidates} candidates, "
              f"best recovery={it.best_recovery:+.3f} ({it.best_candidate_id})")
    print(f"  converged: {result.converged}")

    print("\n=== FINAL RANKED DE NOVO CANDIDATES (by iMEMORY recovery) ===")
    for sc in result.ranked_candidates[:5]:
        r = sc.recovery
        print(f"  {sc.candidate_id:12s}  recovery={r.recovery_fraction:+.3f}  "
              f"QED={sc.qed:.2f}  novelty={sc.novelty:.2f}  "
              f"{r.mean_angle_before_deg:.1f}->{r.mean_angle_after_deg:.1f} deg  "
              f"SMILES={sc.smiles}")

    print("\nDone. LINCS + chemistry + recovery math are real; embedding space is synthetic.")


if __name__ == "__main__":
    main()
