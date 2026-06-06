"""
Structured synthetic patient generator for HetGAT training.

5 clinically defined clusters with biologically inspired gene expression
prototypes. Patients within a cluster = prototype + Gaussian noise.
Cluster separation ensures the trained HetGAT learns meaningful 128-d
embeddings where similar clinical states cluster together.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Gene panel — same 128 HGNC symbols used throughout the repo
# ---------------------------------------------------------------------------
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
GENE_PANEL: List[str] = _RAW_GENES[:128]
assert len(GENE_PANEL) == 128

MODALITIES = [
    "gene_expr", "scRNA", "methylation", "chromatin", "miRNA", "SNP", "metabolomics"
]

# ---------------------------------------------------------------------------
# Clinical cluster definitions
# ---------------------------------------------------------------------------
CLUSTER_DEFS = {
    "healthy":           {"label": 0, "n": 60,  "sledai": 0,  "responder": True},
    "sledai_5_resp":     {"label": 1, "n": 50,  "sledai": 5,  "responder": True},
    "sledai_10_resp":    {"label": 2, "n": 40,  "sledai": 10, "responder": True},
    "sledai_13_nonresp": {"label": 3, "n": 30,  "sledai": 13, "responder": False},
    "sledai_13_resp":    {"label": 4, "n": 40,  "sledai": 13, "responder": True},
}

CLUSTER_NAMES = {v["label"]: k for k, v in CLUSTER_DEFS.items()}

# Gene group indices
_gidx = {g: i for i, g in enumerate(GENE_PANEL)}


def _gene_idx(names: List[str]) -> List[int]:
    """Return indices for named genes that exist in GENE_PANEL."""
    return [_gidx[n] for n in names if n in _gidx]


IFN_GENES    = _gene_idx(["STAT1", "STAT2", "IRF7", "IRF9", "MX1", "MX2", "OAS1", "OAS2",
                           "ISG15", "IFI44", "IFI44L", "IFIT1", "IFIT2", "IFIT3",
                           "IFITM1", "RSAD2", "HERC5", "HERC6", "GBP1", "GBP5",
                           "DDX58", "IFIH1", "LY6E", "XAF1"])
INNATE_GENES = _gene_idx(["TLR2", "TLR4", "TLR7", "TLR9", "MYD88", "NLRP3", "CASP1",
                           "NOD2", "CLEC7A", "TREM1", "S100A8", "S100A9", "S100A12"])
BCELL_GENES  = _gene_idx(["BCL2", "CD19", "MS4A1", "CD27", "CD38", "CD40", "CD40LG",
                           "BLK", "BANK1", "TNFSF13B", "PRDM1", "XBP1", "AICDA"])
EPIGENETIC   = _gene_idx(["EZH2", "KMT2A", "EP300", "CREBBP", "DNMT1", "DNMT3A",
                           "TET2", "HDAC1", "SETD2"])
COMPLEMENT   = _gene_idx(["C1QA", "C1QB", "C3", "C4A", "CR2"])
REGULATORY   = _gene_idx(["FOXP3", "IL2RA", "SOCS1", "SOCS3", "PTPN22", "IL10",
                           "CD163", "MRC1", "TREM2"])
CHEMOKINES   = _gene_idx(["CXCL10", "CXCL9", "CCL2", "CCL5", "IL6", "IL1B", "TNF",
                           "IFNG", "NFKB1", "RELA"])
MYELOID      = _gene_idx(["CD14", "CD68", "ITGAM", "ITGAX", "CSF1R", "FCN1", "LYZ"])


def _build_prototype(cluster_name: str) -> np.ndarray:
    """
    Build a 128-d prototype gene expression vector for each clinical cluster.
    Biologically inspired but synthetically constructed — no real patient data.
    Values are in log2-like scale relative to baseline (0.0).

    The gene-group level assignments drive cluster separation; specific
    gene-level therapeutic signature targets emerge from the trained model's
    W_tau inverse projection — not hardcoded here.
    """
    proto = np.zeros(128, dtype=np.float32)

    if cluster_name == "healthy":
        for i in COMPLEMENT: proto[i] = +0.3
        for i in REGULATORY: proto[i] = +0.4

    elif cluster_name == "sledai_5_resp":
        for i in IFN_GENES:    proto[i] = +0.6
        for i in INNATE_GENES: proto[i] = +0.4
        for i in BCELL_GENES:  proto[i] = +0.5
        for i in COMPLEMENT:   proto[i] = -0.3
        for i in REGULATORY:   proto[i] = -0.2

    elif cluster_name == "sledai_10_resp":
        for i in IFN_GENES:    proto[i] = +1.2
        for i in INNATE_GENES: proto[i] = +0.9
        for i in BCELL_GENES:  proto[i] = +1.0
        for i in EPIGENETIC:   proto[i] = +0.7
        for i in CHEMOKINES:   proto[i] = +0.8
        for i in COMPLEMENT:   proto[i] = -0.7
        for i in REGULATORY:   proto[i] = -0.6

    elif cluster_name == "sledai_13_nonresp":
        # High-disease non-responder: elevated inflammatory gene programs
        # (interferon signaling, innate sensing, B-cell survival, epigenetic
        # remodeling), depleted complement and regulatory programs.
        # Consistent with published SLE biology. The specific gene-level
        # therapeutic signature is recovered by the trained model's inverse
        # projection and is not pre-specified here.
        for i in IFN_GENES:    proto[i] = +2.8
        for i in INNATE_GENES: proto[i] = +2.5
        for i in BCELL_GENES:  proto[i] = +2.6
        for i in EPIGENETIC:   proto[i] = +2.0
        for i in CHEMOKINES:   proto[i] = +2.3
        for i in MYELOID:      proto[i] = +1.8
        for i in COMPLEMENT:   proto[i] = -1.5
        for i in REGULATORY:   proto[i] = -1.4

    elif cluster_name == "sledai_13_resp":
        for i in IFN_GENES:    proto[i] = +2.0
        for i in INNATE_GENES: proto[i] = +1.6
        for i in BCELL_GENES:  proto[i] = +1.8
        for i in EPIGENETIC:   proto[i] = +1.3
        for i in CHEMOKINES:   proto[i] = +1.5
        for i in MYELOID:      proto[i] = +1.2
        for i in COMPLEMENT:   proto[i] = -0.9
        for i in REGULATORY:   proto[i] = -0.5

    return proto


def build_structured_cohort(
    noise_std: float = 0.35,
    seed: int = 42,
    modality_noise_scale: Dict[str, float] | None = None,
) -> Tuple[List[Dict], np.ndarray]:
    """
    Generate 220 synthetic patients across 5 clusters with structured
    gene expression patterns.

    Returns:
        patient_graphs: List of patient graph dicts (same schema as
                        imemory/graph_builder.py) — one per patient.
        labels:         Integer cluster label array, shape (220,).
    """
    rng = np.random.default_rng(seed)
    mod_scale = modality_noise_scale or {
        "gene_expr":    1.0,
        "scRNA":        1.2,
        "methylation":  0.8,
        "chromatin":    0.9,
        "miRNA":        1.3,
        "SNP":          0.4,
        "metabolomics": 1.1,
    }

    all_graphs: List[Dict] = []
    all_labels: List[int] = []

    for cluster_name, cfg in CLUSTER_DEFS.items():
        proto = _build_prototype(cluster_name)
        n     = cfg["n"]
        label = cfg["label"]

        for i in range(n):
            patient_id = f"{cluster_name}_{i:03d}"
            nodes: Dict[str, np.ndarray] = {}

            for mod in MODALITIES:
                scale = mod_scale[mod]
                mod_rng = np.random.default_rng(hash(mod) % 10000 + seed)
                mod_offset = mod_rng.normal(0, 0.2, size=128).astype("float32")
                feat = (proto + mod_offset
                        + rng.normal(0, noise_std * scale, 128).astype("float32"))
                nodes[mod] = feat

            nodes["patient"] = np.mean(
                [nodes[m] for m in MODALITIES], axis=0
            ).astype("float32")

            all_graphs.append({
                "patient_id": patient_id,
                "cluster":    cluster_name,
                "label":      label,
                "sledai":     cfg["sledai"],
                "responder":  cfg["responder"],
                "nodes":      nodes,
                "edges":      [("patient", m) for m in MODALITIES],
                "modalities": MODALITIES,
            })
            all_labels.append(label)

    # Shuffle while keeping graph↔label alignment
    perm = rng.permutation(len(all_graphs))
    all_graphs = [all_graphs[i] for i in perm]
    all_labels_arr = np.array(all_labels)[perm]

    return all_graphs, all_labels_arr


def graphs_to_tensors(graphs: List[Dict]):
    """
    Convert patient graph dicts to stacked tensors for the PyTorch HetGAT.

    Returns:
        patient_t : (N, 128) float32 tensor
        mod_t     : Dict[mod_name -> (N, 128) float32 tensor]
    """
    import torch
    patient_t = torch.tensor(
        np.stack([g["nodes"]["patient"] for g in graphs]), dtype=torch.float32
    )
    mod_t = {
        mod: torch.tensor(
            np.stack([g["nodes"][mod] for g in graphs]), dtype=torch.float32
        )
        for mod in MODALITIES
    }
    return patient_t, mod_t


def get_diseased_healthy_graphs(
    graphs: List[Dict], all_labels: np.ndarray
) -> Tuple[List[Dict], List[Dict]]:
    """
    Return the diseased (SLEDAI 13 NR, label=3) and healthy (label=0) subsets.
    """
    diseased = [g for g, lbl in zip(graphs, all_labels) if lbl == 3]
    healthy  = [g for g, lbl in zip(graphs, all_labels) if lbl == 0]
    return diseased, healthy
