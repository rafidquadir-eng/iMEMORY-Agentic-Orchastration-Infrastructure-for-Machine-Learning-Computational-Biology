"""
discovery/insilico_knockout.py

In-silico knockout validation and iMEMORY recovery scoring.

This is the validation stage and the closed-loop scoring function. A candidate's
predicted transcriptional effect is APPLIED to diseased patient embeddings, and
we measure how far the perturbed patients move toward the healthy state on the
128D hypersphere, using COSINE ANGULAR DISPLACEMENT (the iMEMORY-native distance,
robust to magnitude/batch artifacts). The fraction of the diseased-to-healthy
angular gap that the perturbation closes is the recovery score — the single
number that ranks therapies and drives the closed-loop replan decision.

The candidate effect is REAL, not a synthetic alignment proxy:
  - For a real LINCS compound: its actual L1000 signature (feature space) is
    forward-projected through W_tau into the embedding space (effect_from_signature).
  - For a de novo molecule: its effect is a Tanimoto-similarity-weighted blend of
    its parent LINCS hits' signatures (read_across_signature) — a standard
    read-across estimate — then forward-projected the same way.

Only the embedding space itself is synthetic in the public demo (proprietary
weights/data are excluded); the recovery operation and the signatures are real.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from imemory.therapeutic_signature import cosine_angular_displacement


@dataclass
class RecoveryResult:
    label: str
    recovery_fraction: float        # in (-inf, 1]; 1.0 = fully recovered to healthy
    mean_angle_before_deg: float
    mean_angle_after_deg: float
    n_patients: int


def _mean_angle_to_centroid(embeddings: np.ndarray, healthy_centroid: np.ndarray) -> float:
    """Mean angular displacement (degrees) from each patient to the healthy centroid."""
    angles = [
        cosine_angular_displacement(v, healthy_centroid)["angular_displacement_deg"]
        for v in embeddings
    ]
    return float(np.mean(angles))


# ---------------------------------------------------------------------------
# Real candidate-effect construction (no synthetic proxy)
# ---------------------------------------------------------------------------

def effect_from_signature(feature_signature: np.ndarray, W_tau: np.ndarray) -> np.ndarray:
    """
    Forward-project a compound's feature-space transcriptional signature into the
    128D embedding space to obtain its perturbation effect on patient embeddings.

    A compound that SUPPRESSES disease-up genes and INDUCES disease-down genes has
    a signature anti-correlated with the disease feature signature; projected into
    embedding space and applied to diseased patients, it moves them toward health.

    Args:
        feature_signature: Compound signature in gene/feature space, shape (128,).
                           Sign convention: the compound's induced change per feature.
        W_tau:             Projection matrix, shape (128, 128).

    Returns:
        Embedding-space effect vector, shape (128,).
    """
    assert feature_signature.shape == (128,)
    assert W_tau.shape == (128, 128)
    return (feature_signature @ W_tau.T).astype("float32")


def read_across_signature(
    candidate_fp,
    hit_fingerprints: Sequence,
    hit_signatures: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Estimate a de novo molecule's transcriptional signature by Tanimoto-weighted
    read-across from its parent LINCS hits (a standard similarity-based property
    prediction, used in regulatory read-across).

    effect_signature = Σ_i w_i * hit_signature_i,  w_i = Tanimoto(candidate, hit_i)
    normalized so weights sum to 1.

    Args:
        candidate_fp:      RDKit Morgan fingerprint of the candidate molecule.
        hit_fingerprints:  Morgan fingerprints of the parent LINCS-hit compounds.
        hit_signatures:    Feature-space signatures of those hits, each (128,).

    Returns:
        Estimated feature-space signature for the candidate, shape (128,).
    """
    from rdkit import DataStructs

    if not hit_fingerprints:
        return np.zeros(128, dtype="float32")
    weights = np.array(
        [DataStructs.TanimotoSimilarity(candidate_fp, fp) for fp in hit_fingerprints],
        dtype="float32",
    )
    if weights.sum() <= 1e-8:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    sig = np.zeros(128, dtype="float32")
    for w, hs in zip(weights, hit_signatures):
        sig += w * hs
    return sig


# ---------------------------------------------------------------------------
# Perturbation application
# ---------------------------------------------------------------------------

def apply_target_knockout(
    diseased_embeddings: np.ndarray,
    target_idx: int,
    W_tau: np.ndarray,
    knockout_strength: float = 1.0,
) -> np.ndarray:
    """
    Knock out a target feature and forward-project back into embedding space.
    (Project to feature space via W_tau, attenuate the target feature, project back.)
    """
    assert diseased_embeddings.shape[1] == 128
    assert 0 <= target_idx < 128
    feature = diseased_embeddings @ W_tau
    feature[:, target_idx] *= (1.0 - knockout_strength)
    return (feature @ W_tau.T).astype("float32")


def apply_signature_perturbation(
    diseased_embeddings: np.ndarray,
    effect_embedding: np.ndarray,
    step: float = 1.0,
) -> np.ndarray:
    """
    Apply a candidate's embedding-space effect vector to diseased patients.

    Args:
        diseased_embeddings: shape (N, 128).
        effect_embedding:    Embedding-space effect (from effect_from_signature), (128,).
        step:                Scaling of the applied effect.

    Returns:
        Perturbed embeddings, shape (N, 128).
    """
    return (diseased_embeddings + step * effect_embedding).astype("float32")


# ---------------------------------------------------------------------------
# Recovery scoring (cosine angular displacement)
# ---------------------------------------------------------------------------

def recovery_score(
    diseased_embeddings: np.ndarray,
    perturbed_embeddings: np.ndarray,
    healthy_centroid: np.ndarray,
    label: str = "candidate",
) -> RecoveryResult:
    """
    Fraction of the diseased->healthy ANGULAR gap closed by a perturbation.

    recovery_fraction = (angle_before - angle_after) / angle_before
      1.0 => perturbed patients reach the healthy centroid angle.
      0.0 => no movement.
      < 0 => moved further from health (harmful).
    """
    before = _mean_angle_to_centroid(diseased_embeddings, healthy_centroid)
    after = _mean_angle_to_centroid(perturbed_embeddings, healthy_centroid)
    frac = float((before - after) / (before + 1e-12))
    return RecoveryResult(
        label=label,
        recovery_fraction=round(frac, 4),
        mean_angle_before_deg=round(before, 3),
        mean_angle_after_deg=round(after, 3),
        n_patients=len(diseased_embeddings),
    )


def score_compound_recovery(
    diseased_embeddings: np.ndarray,
    healthy_centroid: np.ndarray,
    feature_signature: np.ndarray,
    W_tau: np.ndarray,
    label: str,
    step: float = 1.0,
) -> RecoveryResult:
    """
    End-to-end recovery for a compound given its real feature-space signature:
    forward-project the signature -> apply to diseased patients -> score recovery.
    """
    effect = effect_from_signature(feature_signature, W_tau)
    perturbed = apply_signature_perturbation(diseased_embeddings, effect, step=step)
    return recovery_score(diseased_embeddings, perturbed, healthy_centroid, label=label)


def score_target_recovery(
    diseased_embeddings: np.ndarray,
    healthy_centroid: np.ndarray,
    target_idx: int,
    target_label: str,
    W_tau: np.ndarray,
    knockout_strength: float = 1.0,
) -> RecoveryResult:
    """Knock out a target and return its recovery score."""
    perturbed = apply_target_knockout(diseased_embeddings, target_idx, W_tau, knockout_strength)
    return recovery_score(diseased_embeddings, perturbed, healthy_centroid, label=target_label)


def rank_by_recovery(results: List[RecoveryResult]) -> List[RecoveryResult]:
    """Rank candidates/targets by recovery fraction, best first."""
    return sorted(results, key=lambda r: r.recovery_fraction, reverse=True)
