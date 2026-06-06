"""
agents/tools/discovery_tools.py

PIPELINE_TOOLS — Anthropic API tool definitions for every pipeline stage.
PipelineContext — mutable session object shared across tool calls.
dispatch_tool   — routes tool_name → real pipeline function.

These are the seams where Claude's tool_use decisions land in real computation.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# A.  PIPELINE_TOOLS — Anthropic tool definitions
# ---------------------------------------------------------------------------
PIPELINE_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "build_patient_cohorts",
        "description": (
            "Load or train the iMEMORY HetGAT on a 5-cluster structured synthetic "
            "cohort (healthy, SLEDAI 5/10/13 responders, SLEDAI 13 non-responders). "
            "Returns trained diseased (SLEDAI 13 NR) and healthy embeddings from the "
            "learned 128-d embedding space, plus the trained W_tau projection matrix."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "force_retrain": {"type": "boolean", "default": False},
            },
            "required": [],
        },
    },
    {
        "name": "compute_therapeutic_signature",
        "description": (
            "Compute therapeutic delta Δ = v_diseased - v_healthy, apply cosine "
            "angular displacement analysis, inverse-project through W_tau to the "
            "gene feature space, and return ranked target nominations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "top_targets": {"type": "integer", "default": 5},
            },
            "required": [],
        },
    },
    {
        "name": "screen_lincs",
        "description": (
            "Screen the disease feature signature against the LINCS L1000 library "
            "via the SigCom LINCS API (real, with synthetic fallback). Returns "
            "compounds ranked by reversal potential (negative connectivity)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "top_hits": {"type": "integer", "default": 10},
            },
            "required": [],
        },
    },
    {
        "name": "generate_molecules",
        "description": (
            "Generate de novo candidate molecules via SELFIES+RDKit mutation, seeded "
            "by the top LINCS hit SMILES. Returns candidates with QED, MW, logP, "
            "TPSA, novelty properties."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n_candidates": {"type": "integer", "default": 8},
                "diversity":    {"type": "number", "default": 1.0},
            },
            "required": [],
        },
    },
    {
        "name": "score_recovery",
        "description": (
            "Run iMEMORY in-silico knockout recovery scoring on all generated "
            "candidates. For each candidate, forward-projects its read-across "
            "feature signature into embedding space, applies it to diseased patients, "
            "and measures angular recovery toward the healthy centroid via cosine "
            "angular displacement. Returns candidates ranked by recovery fraction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "recovery_threshold": {"type": "number", "default": 0.5},
            },
            "required": [],
        },
    },
    {
        "name": "generate_visualizations",
        "description": (
            "Generate and save 6 matplotlib figures: (1) PCA scatter of diseased vs "
            "healthy embeddings, (2) candidate recovery bar chart colored by QED, "
            "(3) QED vs novelty scatter sized by recovery, (4) LINCS reversal "
            "potential histogram, (5) closed-loop convergence curve, "
            "(6) all-patient cluster embedding PCA scatter (5 clinical clusters). "
            "Returns dict of figure paths."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "output_dir": {"type": "string", "default": "./outputs/figures"},
                "session_id": {"type": "string", "default": ""},
            },
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# B.  PipelineContext — mutable session state shared across tool calls
# ---------------------------------------------------------------------------
@dataclass
class PipelineContext:
    gene_labels: List[str]
    W_tau: Optional[np.ndarray] = None
    synthetic_library: Optional[Dict[str, Any]] = None

    diseased_embeddings: Optional[np.ndarray] = None
    healthy_embeddings: Optional[np.ndarray] = None
    delta: Optional[np.ndarray] = None
    disease_feature_signature: Optional[np.ndarray] = None
    targets: Optional[List] = None

    lincs_hits: List = field(default_factory=list)
    lincs_mode: str = "none"
    _hit_sigs: Dict[str, np.ndarray] = field(default_factory=dict)
    _hit_smiles: Dict[str, str] = field(default_factory=dict)

    candidates: List = field(default_factory=list)
    generation_mode: str = "none"

    scored_candidates: List = field(default_factory=list)
    iterations: List = field(default_factory=list)
    converged: bool = False

    figure_paths: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# C.  dispatch_tool — routes tool name → real pipeline function
# ---------------------------------------------------------------------------
def dispatch_tool(
    tool_name: str, tool_input: Dict[str, Any], ctx: PipelineContext
) -> Dict[str, Any]:
    """Route an Anthropic tool_use block to the real pipeline function."""
    if tool_name == "build_patient_cohorts":
        return _build_patient_cohorts(tool_input, ctx)
    elif tool_name == "compute_therapeutic_signature":
        return _compute_therapeutic_signature(tool_input, ctx)
    elif tool_name == "screen_lincs":
        return _screen_lincs(tool_input, ctx)
    elif tool_name == "generate_molecules":
        return _generate_molecules(tool_input, ctx)
    elif tool_name == "score_recovery":
        return _score_recovery(tool_input, ctx)
    elif tool_name == "generate_visualizations":
        return _generate_visualizations(tool_input, ctx)
    else:
        raise ValueError(f"Unknown tool: {tool_name}")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def _build_patient_cohorts(inp: Dict, ctx: PipelineContext) -> Dict:
    """
    Load trained HetGAT embeddings from checkpoint (training automatically
    if the checkpoint is absent). Populates ctx with trained diseased/healthy
    embeddings and the learned W_tau projection matrix.
    """
    # If caller pre-supplied trained embeddings (e.g. from demo script), use them.
    if (ctx.diseased_embeddings is not None
            and ctx.healthy_embeddings is not None
            and ctx.W_tau is not None
            and not inp.get("force_retrain", False)):
        training_source = "pre-supplied (caller)"
        best_val_acc = "n/a"
    else:
        from training.train import ensure_trained
        trained = ensure_trained(verbose=True)
        ctx.diseased_embeddings = trained["diseased_embeddings"]
        ctx.healthy_embeddings  = trained["healthy_embeddings"]
        ctx.W_tau               = trained["W_tau"]
        training_source = "trained HetGAT (synthetic structured cohort, 5 clusters)"
        # Try to read best_val_acc from checkpoint
        try:
            import torch
            from training.train import CHECKPOINT_PATH
            ck = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
            best_val_acc = round(float(ck.get("best_val_acc", 0.0)), 4)
        except Exception:  # noqa: BLE001
            best_val_acc = "loaded from checkpoint"

    v_d = ctx.diseased_embeddings.mean(axis=0)
    v_h = ctx.healthy_embeddings.mean(axis=0)
    axis_norm = float(np.linalg.norm(v_d - v_h))

    return {
        "n_diseased":           int(ctx.diseased_embeddings.shape[0]),
        "n_healthy":            int(ctx.healthy_embeddings.shape[0]),
        "embedding_dim":        int(ctx.diseased_embeddings.shape[1]),
        "disease_axis_norm":    round(axis_norm, 4),
        "cluster_description":  "SLEDAI 13 non-responders (diseased) vs healthy",
        "training_source":      training_source,
        "best_val_acc":         best_val_acc,
    }


def _compute_therapeutic_signature(inp: Dict, ctx: PipelineContext) -> Dict:
    from imemory.therapeutic_signature import (
        compute_therapeutic_delta,
        inverse_project,
        nominate_targets_from_delta,
        cosine_angular_displacement,
    )

    top_k = int(inp.get("top_targets", 5))
    v_d = ctx.diseased_embeddings.mean(axis=0)
    v_h = ctx.healthy_embeddings.mean(axis=0)

    ctx.delta = compute_therapeutic_delta(v_d, v_h)
    ctx.disease_feature_signature = inverse_project(ctx.delta, ctx.W_tau)
    ctx.targets = nominate_targets_from_delta(
        ctx.delta, ctx.W_tau, ctx.gene_labels, top_k
    )

    # Population angular profile across patients
    angles = [
        float(np.degrees(np.arccos(
            np.clip(
                float(np.dot(d / (np.linalg.norm(d) + 1e-12),
                             v_h / (np.linalg.norm(v_h) + 1e-12))),
                -1.0, 1.0)
        )))
        for d in ctx.diseased_embeddings
    ]
    mean_angle = float(np.mean(angles))

    return {
        "targets": [(label, round(float(score), 4)) for label, score in ctx.targets],
        "delta_l2_norm": round(float(np.linalg.norm(ctx.delta)), 4),
        "mean_angular_displacement_deg": round(mean_angle, 2),
    }


def _screen_lincs(inp: Dict, ctx: PipelineContext) -> Dict:
    top_hits = int(inp.get("top_hits", 10))

    # Try real SigCom API first
    try:
        from discovery.lincs_api import query_lincs_reversers, LincsUnavailable
        hits = query_lincs_reversers(
            ctx.disease_feature_signature, ctx.gene_labels, limit=top_hits
        )
        hit_sigs, hit_smiles = {}, {}
        for h in hits:
            hit_sigs[h.pert_name] = (
                -h.reversal_potential * ctx.disease_feature_signature
            ).astype("float32")
            if h.smiles:
                hit_smiles[h.pert_name] = h.smiles
        ctx.lincs_hits  = hits
        ctx.lincs_mode  = "real"
        ctx._hit_sigs   = hit_sigs
        ctx._hit_smiles = hit_smiles
        hit_rows = [
            {
                "pert_name": h.pert_name,
                "z_sum": round(float(h.z_sum), 4) if h.z_sum is not None else None,
                "reversal_potential": round(float(h.reversal_potential), 4),
                "smiles": h.smiles or "",
            }
            for h in hits[:top_hits]
        ]
    except Exception:  # noqa: BLE001
        from discovery.lincs_connectivity import screen_lincs_library
        assert ctx.synthetic_library is not None, "Need synthetic_library for fallback."
        hits = screen_lincs_library(
            ctx.disease_feature_signature, ctx.synthetic_library,
            top_k=top_hits, weighted=True
        )
        hit_sigs   = {h.compound_id: ctx.synthetic_library[h.compound_id] for h in hits}
        hit_smiles = {h.compound_id: h.smiles_seed for h in hits if h.smiles_seed}
        ctx.lincs_hits  = hits
        ctx.lincs_mode  = "synthetic"
        ctx._hit_sigs   = hit_sigs
        ctx._hit_smiles = hit_smiles
        hit_rows = [
            {
                "pert_name": h.compound_id,
                "z_sum": None,
                "reversal_potential": round(float(h.reversal_potential), 4),
                "smiles": h.smiles_seed or "",
            }
            for h in hits[:top_hits]
        ]

    return {
        "n_hits":     len(hit_rows),
        "lincs_mode": ctx.lincs_mode,
        "top_hits":   hit_rows,
    }


def _generate_molecules(inp: Dict, ctx: PipelineContext) -> Dict:
    n_cands   = int(inp.get("n_candidates", 8))
    diversity = float(inp.get("diversity", 1.0))
    seed_smiles = [s for s in ctx._hit_smiles.values() if s]

    try:
        from discovery.denovo_generation import SelfiesGenerator
        gen   = SelfiesGenerator(seed=23)
        cands = gen.generate(seed_smiles=seed_smiles, n_candidates=n_cands,
                             diversity=diversity)
        mode  = "selfies"
    except Exception:  # noqa: BLE001
        from discovery.denovo_generation import Candidate
        seeds = seed_smiles or ["O=C(N)c1ccccc1"]
        cands = [
            Candidate(
                candidate_id=f"IAB-GEN-{i:03d}",
                smiles=seeds[i % len(seeds)] + "C" * (1 + i % 3),
                parent_smiles=seeds[i % len(seeds)],
            )
            for i in range(n_cands)
        ]
        mode = "fallback"

    ctx.candidates      = cands
    ctx.generation_mode = mode

    rows = [
        {
            "id":      getattr(c, "candidate_id", str(i)),
            "smiles":  c.smiles,
            "qed":     round(float(getattr(c, "qed", 0.0)), 4),
            "mw":      round(float(getattr(c, "mw", 0.0)), 2),
            "logp":    round(float(getattr(c, "logp", 0.0)), 3),
            "novelty": round(float(getattr(c, "novelty", 1.0)), 4),
        }
        for i, c in enumerate(cands)
    ]
    return {"n_candidates": len(rows), "generation_mode": mode, "candidates": rows}


def _score_recovery(inp: Dict, ctx: PipelineContext) -> Dict:
    threshold = float(inp.get("recovery_threshold", 0.5))
    from discovery.insilico_knockout import (
        apply_signature_perturbation, effect_from_signature,
        read_across_signature, recovery_score,
    )
    from discovery.closed_loop import ScoredCandidate, LoopIteration

    healthy_centroid = ctx.healthy_embeddings.mean(axis=0)
    scored = []
    for cand in ctx.candidates:
        sigs = list(ctx._hit_sigs.values())
        if not sigs:
            effect = np.zeros(128, dtype="float32")
        else:
            cand_fp = getattr(cand, "fingerprint", None)
            if cand_fp is not None:
                try:
                    from rdkit import Chem
                    from rdkit.Chem import AllChem
                    hit_fps, ordered_sigs = [], []
                    for hid, sig in ctx._hit_sigs.items():
                        smi = ctx._hit_smiles.get(hid)
                        if not smi:
                            continue
                        mol = Chem.MolFromSmiles(smi)
                        if mol is None:
                            continue
                        hit_fps.append(
                            AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        )
                        ordered_sigs.append(sig)
                    if hit_fps:
                        feat_sig = read_across_signature(cand_fp, hit_fps, ordered_sigs)
                        effect   = effect_from_signature(feat_sig, ctx.W_tau)
                    else:
                        effect = effect_from_signature(
                            np.mean(sigs, axis=0).astype("float32"), ctx.W_tau
                        )
                except Exception:  # noqa: BLE001
                    effect = effect_from_signature(
                        np.mean(sigs, axis=0).astype("float32"), ctx.W_tau
                    )
            else:
                effect = effect_from_signature(
                    np.mean(sigs, axis=0).astype("float32"), ctx.W_tau
                )

        perturbed = apply_signature_perturbation(ctx.diseased_embeddings, effect, step=1.0)
        rec = recovery_score(
            ctx.diseased_embeddings, perturbed, healthy_centroid,
            label=getattr(cand, "candidate_id", "?")
        )
        scored.append(ScoredCandidate(
            candidate_id=getattr(cand, "candidate_id", "?"),
            smiles=cand.smiles,
            recovery=rec,
            qed=float(getattr(cand, "qed", 0.0)),
            novelty=float(getattr(cand, "novelty", 1.0)),
        ))

    scored.sort(key=lambda s: s.recovery.recovery_fraction, reverse=True)
    ctx.scored_candidates = scored
    ctx.converged = bool(scored and scored[0].recovery.recovery_fraction >= threshold)

    ctx.iterations.append(LoopIteration(
        iteration=len(ctx.iterations) + 1,
        n_candidates=len(scored),
        best_recovery=scored[0].recovery.recovery_fraction if scored else 0.0,
        best_candidate_id=scored[0].candidate_id if scored else "",
    ))

    ranked_rows = [
        {
            "id":                 sc.candidate_id,
            "smiles":             sc.smiles,
            "recovery_fraction":  round(float(sc.recovery.recovery_fraction), 4),
            "qed":                round(sc.qed, 4),
            "novelty":            round(sc.novelty, 4),
            "angle_before_deg":   round(float(sc.recovery.mean_angle_before_deg), 2),
            "angle_after_deg":    round(float(sc.recovery.mean_angle_after_deg), 2),
        }
        for sc in scored[:10]
    ]
    best_rf = scored[0].recovery.recovery_fraction if scored else 0.0
    return {
        "converged":           ctx.converged,
        "best_recovery":       round(float(best_rf), 4),
        "n_iterations_needed": len(ctx.iterations),
        "ranked":              ranked_rows,
    }


def _generate_visualizations(inp: Dict, ctx: PipelineContext) -> Dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from sklearn.decomposition import PCA

    output_dir = inp.get("output_dir", "./outputs/figures")
    session_id = inp.get("session_id", "")
    os.makedirs(output_dir, exist_ok=True)

    def _fpath(name: str) -> str:
        suffix = f"_{session_id}" if session_id else ""
        return os.path.join(output_dir, f"{name}{suffix}.png")

    paths: Dict[str, str] = {}

    # ── Fig 1: PCA scatter ──────────────────────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        all_emb = np.vstack([ctx.diseased_embeddings, ctx.healthy_embeddings])
        pca = PCA(n_components=2)
        proj = pca.fit_transform(all_emb)
        n_d = len(ctx.diseased_embeddings)
        ax.scatter(proj[:n_d, 0], proj[:n_d, 1], c="tomato",
                   label=f"Diseased (n={n_d})", alpha=0.6, s=40, zorder=3)
        ax.scatter(proj[n_d:, 0], proj[n_d:, 1], c="steelblue",
                   label=f"Healthy (n={len(ctx.healthy_embeddings)})",
                   alpha=0.6, s=40, zorder=3)
        if ctx.scored_candidates:
            best = ctx.scored_candidates[0]
            # project best candidate mean effect as a star
            sigs = list(ctx._hit_sigs.values())
            if sigs:
                from discovery.insilico_knockout import (
                    apply_signature_perturbation, effect_from_signature,
                )
                effect   = effect_from_signature(np.mean(sigs, axis=0).astype("float32"), ctx.W_tau)
                perturbed = apply_signature_perturbation(ctx.diseased_embeddings, effect, step=1.0)
                mean_p   = perturbed.mean(axis=0, keepdims=True)
                proj_p   = pca.transform(mean_p)
                ax.scatter(proj_p[:, 0], proj_p[:, 1], marker="*", s=250,
                           c="limegreen", zorder=5, label="Best cand (mean effect)")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        ax.set_title("Patient Embedding Space (PCA) — Diseased vs Healthy")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = _fpath("pca_embeddings")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths["pca"] = p
    except Exception as e:
        paths["pca_error"] = str(e)

    # ── Fig 2: Recovery bar chart ───────────────────────────────────────────
    try:
        ranked = ctx.scored_candidates[:10]
        if ranked:
            fig, ax = plt.subplots(figsize=(9, max(3, len(ranked) * 0.55)))
            rfs  = [s.recovery.recovery_fraction for s in ranked]
            qeds = [s.qed for s in ranked]
            cmap = cm.get_cmap("RdYlGn")
            colors = [cmap(q) for q in qeds]
            labels = [s.smiles[:30] + ("…" if len(s.smiles) > 30 else "")
                      for s in ranked]
            y = range(len(ranked))
            ax.barh(y, rfs, color=colors, edgecolor="white", height=0.65)
            ax.set_yticks(list(y))
            ax.set_yticklabels(labels, fontsize=7)
            ax.axvline(0, color="black", lw=0.8)
            ax.axvline(0.5, color="green", lw=1, ls="--", alpha=0.5, label="threshold 0.5")
            ax.set_xlim(-0.5, 1.0)
            ax.set_xlabel("Recovery fraction")
            ax.set_title("Candidate Recovery Scores (angular shift toward healthy)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            p = _fpath("recovery_scores")
            fig.savefig(p, dpi=130)
            plt.close(fig)
            paths["recovery"] = p
    except Exception as e:
        paths["recovery_error"] = str(e)

    # ── Fig 3: Molecular properties scatter ─────────────────────────────────
    try:
        ranked = ctx.scored_candidates
        if ranked:
            qeds  = np.array([s.qed for s in ranked])
            novs  = np.array([s.novelty for s in ranked])
            rfs   = np.array([s.recovery.recovery_fraction for s in ranked])
            sizes = np.clip(rfs, 0.1, 0.5) * 500
            cmap  = cm.get_cmap("RdYlGn")
            fig, ax = plt.subplots(figsize=(7, 5))
            sc = ax.scatter(qeds, novs, c=rfs, s=sizes, cmap=cmap,
                            vmin=-0.2, vmax=1.0, alpha=0.8, edgecolors="white")
            plt.colorbar(sc, ax=ax, label="Recovery fraction")
            for i, s in enumerate(ranked[:3]):
                ax.annotate(s.candidate_id, (qeds[i], novs[i]),
                            fontsize=7, ha="left", va="bottom")
            ax.set_xlabel("QED")
            ax.set_ylabel("Novelty")
            ax.set_title("De Novo Candidate Properties")
            fig.tight_layout()
            p = _fpath("molecular_properties")
            fig.savefig(p, dpi=130)
            plt.close(fig)
            paths["properties"] = p
    except Exception as e:
        paths["properties_error"] = str(e)

    # ── Fig 4: LINCS reversal histogram ─────────────────────────────────────
    try:
        if ctx.lincs_hits:
            revs = [float(getattr(h, "reversal_potential", 0.0)) for h in ctx.lincs_hits]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(revs, bins=min(20, len(revs)), color="steelblue",
                    edgecolor="white", alpha=0.85)
            ax.axvline(0.7, color="tomato", lw=1.5, ls="--",
                       label="Strong reverser threshold (0.7)")
            ax.set_xlabel("Reversal potential")
            ax.set_ylabel("Count")
            ax.set_title("LINCS Reversal Potential Distribution")
            ax.legend(fontsize=8)
            fig.tight_layout()
            p = _fpath("lincs_connectivity")
            fig.savefig(p, dpi=130)
            plt.close(fig)
            paths["lincs"] = p
    except Exception as e:
        paths["lincs_error"] = str(e)

    # ── Fig 5: Convergence curve ─────────────────────────────────────────────
    try:
        if ctx.iterations:
            iters = [it.iteration for it in ctx.iterations]
            bests = [it.best_recovery for it in ctx.iterations]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(iters, bests, "o-", color="steelblue", lw=2, ms=8)
            ax.axhline(0.5, color="green", lw=1, ls="--", label="Threshold 0.5")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Best recovery fraction")
            ax.set_title("Closed-Loop Convergence")
            ax.set_xticks(iters)
            ax.legend(fontsize=8)
            fig.tight_layout()
            p = _fpath("convergence")
            fig.savefig(p, dpi=130)
            plt.close(fig)
            paths["convergence"] = p
    except Exception as e:
        paths["convergence_error"] = str(e)

    # ── Fig 6: All-patient cluster embedding PCA ────────────────────────────
    try:
        from training.train import ALL_EMBEDDINGS_PATH, ALL_LABELS_PATH
        from training.synthetic_patients import CLUSTER_NAMES
        if ALL_EMBEDDINGS_PATH.exists() and ALL_LABELS_PATH.exists():
            all_emb    = np.load(ALL_EMBEDDINGS_PATH)
            all_labels = np.load(ALL_LABELS_PATH)
            pca6 = PCA(n_components=2)
            proj6 = pca6.fit_transform(all_emb)

            cluster_colors = {0: "steelblue", 1: "seagreen", 2: "gold",
                              3: "tomato",    4: "darkorange"}
            fig, ax = plt.subplots(figsize=(8, 6))
            for lbl, color in cluster_colors.items():
                mask = all_labels == lbl
                name = CLUSTER_NAMES.get(lbl, str(lbl))
                ax.scatter(proj6[mask, 0], proj6[mask, 1],
                           c=color, label=f"{lbl}: {name} (n={mask.sum()})",
                           alpha=0.7, s=35, zorder=3)
            ax.set_xlabel(f"PC1 ({pca6.explained_variance_ratio_[0]:.1%} var)")
            ax.set_ylabel(f"PC2 ({pca6.explained_variance_ratio_[1]:.1%} var)")
            ax.set_title("iMEMORY 128-d Embedding Space (PCA) — 5 Clinical Clusters")
            ax.legend(fontsize=7, loc="best")
            fig.tight_layout()
            p = _fpath("cluster_embeddings")
            fig.savefig(p, dpi=130)
            plt.close(fig)
            paths["cluster_embeddings"] = p
    except Exception as e:
        paths["cluster_embeddings_error"] = str(e)

    ctx.figure_paths = paths
    return {"figure_paths": paths}
