"""
Vector operations and embedding-distance target nomination.

Genes whose embeddings sit closest to the disease centroid are nominated as
candidate targets. This is the synthetic analog of how iMEMORY surfaces
druggable axes; here it runs on random gene embeddings for demonstration only.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def nominate_targets(
    gene_embeddings: np.ndarray,
    gene_labels: List[str],
    centroid: np.ndarray,
    top_k: int = 2,
) -> List[Tuple[str, float]]:
    """Return the top_k (gene_label, similarity) pairs nearest the disease centroid."""
    sims = [(label, cosine_similarity(emb, centroid))
            for label, emb in zip(gene_labels, gene_embeddings)]
    sims.sort(key=lambda x: x[1], reverse=True)
    # Normalize similarity into a 0–1 score for downstream audit checks.
    top = sims[:top_k]
    return [(label, round((sim + 1) / 2, 3)) for label, sim in top]
