"""
Synthetic multimodal patient-graph builder.

iMEMORY's production HetGAT integrates seven omics modalities into one
heterogeneous graph per patient. This module builds a structurally-faithful
synthetic version for the demo: modality nodes connected to a patient node, with
random feature vectors. No real data is used.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

MODALITIES: List[str] = [
    "gene_expr", "scRNA", "methylation", "chromatin", "miRNA", "SNP", "metabolomics",
]


def build_synthetic_patient_graph(
    patient_id: str, n_features: int = 128, seed: int | None = None
) -> Dict:
    rng = np.random.default_rng(seed)
    nodes = {
        modality: rng.normal(size=n_features).astype("float32") for modality in MODALITIES
    }
    nodes["patient"] = np.mean(list(nodes.values()), axis=0).astype("float32")
    edges = [("patient", m) for m in MODALITIES]  # star topology, patient at center
    return {"patient_id": patient_id, "nodes": nodes, "edges": edges, "modalities": MODALITIES}
