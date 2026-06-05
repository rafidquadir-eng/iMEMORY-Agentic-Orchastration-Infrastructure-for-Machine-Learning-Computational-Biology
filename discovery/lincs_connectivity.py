"""
discovery/lincs_connectivity.py

LINCS L1000 connectivity screening.

Given a disease feature signature (the inverse-projected therapeutic delta),
screen a library of compound perturbation signatures and rank compounds by how
strongly they REVERSE the disease signature. This is the Connectivity Map (CMap)
concept: a compound whose transcriptional signature is anti-correlated with the
disease signature is a candidate to push the system back toward health.

Production note: the real iMEMORY screen queries the full LINCS L1000 corpus
(30,000+ compound/perturbagen signatures over 978 landmark genes) using the
weighted Kolmogorov-Smirnov enrichment statistic. This module implements the
cosine anti-correlation form, which is the same idea in a dependency-light,
auditable form, and exposes a weighted-enrichment variant for higher fidelity.
The synthetic library here stands in for the real L1000 data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CompoundHit:
    compound_id: str
    connectivity_score: float       # in [-1, 1]; negative = reverses disease
    reversal_potential: float       # in [0, 1]; higher = stronger candidate
    smiles_seed: Optional[str] = None


def connectivity_score(compound_signature: np.ndarray, disease_signature: np.ndarray) -> float:
    """
    Cosine connectivity between a compound signature and the disease signature.

    A NEGATIVE score means the compound's effect is anti-correlated with the
    disease direction — i.e., it reverses the disease signature (therapeutic).
    A POSITIVE score means the compound mimics the disease (to be avoided).

    Args:
        compound_signature: Compound perturbation signature, shape (D,).
        disease_signature:  Disease feature signature (inverse-projected Δ), shape (D,).

    Returns:
        Connectivity score in [-1, 1].
    """
    denom = (np.linalg.norm(compound_signature) * np.linalg.norm(disease_signature)) + 1e-12
    return float(np.dot(compound_signature, disease_signature) / denom)


def weighted_connectivity_score(
    compound_signature: np.ndarray,
    disease_signature: np.ndarray,
    top_fraction: float = 0.25,
) -> float:
    """
    CMap-style weighted connectivity: emphasize the features that move most in
    the disease signature, mimicking the landmark-gene weighting of L1000.

    Weights each feature by |disease_signature| (the most dysregulated features
    dominate), then computes the weighted cosine. Closer to the real weighted
    enrichment statistic than plain cosine.

    Args:
        compound_signature: shape (D,).
        disease_signature:  shape (D,).
        top_fraction:       Fraction of most-dysregulated features to weight.

    Returns:
        Weighted connectivity score in [-1, 1].
    """
    importance = np.abs(disease_signature)
    k = max(1, int(len(importance) * top_fraction))
    top_idx = np.argsort(importance)[-k:]
    w = np.zeros_like(importance)
    w[top_idx] = importance[top_idx]
    cs = compound_signature * w
    ds = disease_signature * w
    denom = (np.linalg.norm(cs) * np.linalg.norm(ds)) + 1e-12
    return float(np.dot(cs, ds) / denom)


def screen_lincs_library(
    disease_signature: np.ndarray,
    library: Dict[str, np.ndarray],
    smiles_seeds: Optional[Dict[str, str]] = None,
    top_k: int = 10,
    weighted: bool = True,
) -> List[CompoundHit]:
    """
    Screen the entire compound library against the disease signature and return
    the top reversal candidates (most negative connectivity).

    Args:
        disease_signature: Inverse-projected disease feature signature, shape (D,).
        library:           Mapping compound_id -> signature vector, each shape (D,).
        smiles_seeds:      Optional mapping compound_id -> seed SMILES, used to
                           seed de novo generation downstream.
        top_k:             Number of top reversal candidates to return.
        weighted:          Use the weighted connectivity statistic if True.

    Returns:
        List of CompoundHit, sorted by reversal_potential descending
        (i.e., most strongly reversing the disease signature first).
    """
    score_fn = weighted_connectivity_score if weighted else connectivity_score
    hits: List[CompoundHit] = []
    for cid, sig in library.items():
        cs = score_fn(sig, disease_signature)
        # Reversal potential: negative connectivity -> high potential. Map [-1,1] -> [0,1].
        reversal = float((-cs + 1.0) / 2.0)
        hits.append(
            CompoundHit(
                compound_id=cid,
                connectivity_score=round(cs, 4),
                reversal_potential=round(reversal, 4),
                smiles_seed=(smiles_seeds or {}).get(cid),
            )
        )
    hits.sort(key=lambda h: h.reversal_potential, reverse=True)
    return hits[:top_k]


def inject_planted_reversers(
    library: Dict[str, np.ndarray],
    reversal_axis: np.ndarray,
    n: int = 5,
    seed: int = 7,
    seed_smiles: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Add a small number of strong reverser compounds to a library, defined as
    (-reversal_axis + noise). Used by the demo so the screen returns a
    meaningful, reproducible top set rather than random noise. Clearly a
    synthetic demonstration device — real hits come from real L1000 data.

    Mutates `library` in place. Returns a mapping of the planted ids -> seed SMILES.
    """
    rng = np.random.default_rng(seed)
    axis = reversal_axis / (np.linalg.norm(reversal_axis) + 1e-12)
    seed_smiles = seed_smiles or []
    planted_smiles: Dict[str, str] = {}
    for i in range(n):
        cid = f"PLANTED-REVERSER-{i:02d}"
        sig = -axis + 0.15 * rng.normal(size=axis.shape)   # strongly anti-correlated
        library[cid] = sig.astype("float32")
        if i < len(seed_smiles):
            planted_smiles[cid] = seed_smiles[i]
    return planted_smiles
