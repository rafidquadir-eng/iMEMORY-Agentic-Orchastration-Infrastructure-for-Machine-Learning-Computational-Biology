"""
128-dimensional embedding space utilities.

Projects a patient graph to a single embedding (mean-pool over modality nodes)
and exposes the disease centroid used as the reference point for target
nomination. Synthetic stand-in for the learned HetGAT embedding space.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def project_graph(graph: Dict) -> np.ndarray:
    """Mean-pool modality node features into one patient embedding."""
    modality_vectors = [graph["nodes"][m] for m in graph["modalities"]]
    return np.mean(modality_vectors, axis=0).astype("float32")


def disease_centroid(graphs: List[Dict]) -> np.ndarray:
    """Centroid of a set of patient embeddings — the reference for nomination."""
    embeddings = np.stack([project_graph(g) for g in graphs])
    return embeddings.mean(axis=0).astype("float32")
