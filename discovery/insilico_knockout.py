"""
discovery/insilico_knockout.py

In-silico recovery scoring using cosine angular displacement.

Applies a drug effect (a 128-d perturbation vector in embedding space) to the
diseased patient embeddings and measures how far they shift toward the healthy
centroid — expressed as a recovery fraction (0 = no movement, 1 = full recovery).
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def apply_signature_perturbation(
    diseased_embeddings: np.ndarray,
    effect_vector: np.ndarray,
    step: float = 1.0,
) -> np.ndarray:
    """
    Add `step * effect_vector` to each diseased patient embedding.
    Returns a new array of the same shape (does not mutate input).
    """
    return diseased_embeddings + step * effect_vector.astype("float32")


def effect_from_signature(
    feature_signature: np.ndarray,
    W_tau: np.ndarray,
) -> np.ndarray:
    """
    Project a feature-space signature through W_tau into the embedding space.
    This is the forward projection of how a drug effect (expressed as a
    gene-space vector) translates into a movement in the 128-d embedding space.
    """
    sig = feature_signature.astype("float32")
    W   = W_tau.astype("float32")
    # W_tau is (128, 128); project sig into embedding space
    effect = W @ sig
    # normalise to unit vector scaled by the original magnitude
    norm = float(np.linalg.norm(effect))
    if norm > 1e-8:
        effect = effect / norm * float(np.linalg.norm(sig))
    return effect.astype("float32")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


@dataclasses.dataclass
class RecoveryScore:
    label: str
    mean_cosine_before: float     # mean cosine sim to healthy centroid before perturbation
    mean_cosine_after:  float     # mean cosine sim to healthy centroid after perturbation
    recovery_fraction:  float     # (after - before) / (1 - before), clamped to [0, 1]
    n_patients:         int
    mean_angle_before_deg: float = 0.0   # angular distance from healthy centroid before
    mean_angle_after_deg:  float = 0.0   # angular distance from healthy centroid after


def recovery_score(
    diseased_before: np.ndarray,
    diseased_after:  np.ndarray,
    healthy_centroid: np.ndarray,
    label: str = "candidate",
) -> RecoveryScore:
    """
    Measure how much the perturbation moves diseased patients toward the
    healthy centroid in cosine space.

    recovery_fraction = (cosine_after - cosine_before) / (1 - cosine_before)
    Clamped to [0, 1]; values > 0 indicate movement toward healthy.
    """
    sims_before = np.array([_cosine(e, healthy_centroid) for e in diseased_before])
    sims_after  = np.array([_cosine(e, healthy_centroid) for e in diseased_after])

    mean_before = float(sims_before.mean())
    mean_after  = float(sims_after.mean())

    denom = 1.0 - mean_before
    if abs(denom) < 1e-8:
        frac = 0.0
    else:
        frac = float(np.clip((mean_after - mean_before) / denom, 0.0, 1.0))

    # Angular distances (degrees) from the healthy centroid
    angle_before = float(np.degrees(np.arccos(np.clip(mean_before, -1.0, 1.0))))
    angle_after  = float(np.degrees(np.arccos(np.clip(mean_after,  -1.0, 1.0))))

    return RecoveryScore(
        label=label,
        mean_cosine_before=round(mean_before, 4),
        mean_cosine_after=round(mean_after, 4),
        recovery_fraction=round(frac, 4),
        n_patients=len(diseased_before),
        mean_angle_before_deg=round(angle_before, 2),
        mean_angle_after_deg=round(angle_after,  2),
    )


# Alias for backward-compat with closed_loop.py
RecoveryResult = RecoveryScore


# ---------------------------------------------------------------------------
# Read-across similarity weighting
# ---------------------------------------------------------------------------

def read_across_signature(
    candidate_fp,
    hit_fps: List,
    hit_sigs: List[np.ndarray],
    min_similarity: float = 0.0,
) -> np.ndarray:
    """
    Weighted average of LINCS perturbation signatures, weighted by Tanimoto
    similarity between the candidate and each LINCS hit.

    Parameters
    ----------
    candidate_fp : RDKit ExplicitBitVect
        Morgan fingerprint of the de-novo candidate molecule.
    hit_fps : list of RDKit ExplicitBitVect
        Fingerprints for the top LINCS hits.
    hit_sigs : list of np.ndarray shape (128,)
        Gene-expression signatures for each LINCS hit.
    min_similarity : float
        Hits below this Tanimoto threshold are excluded.

    Returns
    -------
    np.ndarray shape (128,)
        Weighted signature (gene feature space vector).
    """
    try:
        from rdkit.DataStructs import TanimotoSimilarity
        sims = np.array([
            TanimotoSimilarity(candidate_fp, fp) for fp in hit_fps
        ], dtype="float32")
    except Exception:
        sims = np.ones(len(hit_fps), dtype="float32")

    mask = sims >= min_similarity
    if not mask.any():
        sims = np.ones(len(hit_fps), dtype="float32")
        mask = np.ones(len(hit_fps), dtype=bool)

    w = sims[mask]
    w /= w.sum() + 1e-12
    sigs = np.stack([s for s, m in zip(hit_sigs, mask) if m], axis=0)
    return (w[:, None] * sigs).sum(axis=0).astype("float32")
