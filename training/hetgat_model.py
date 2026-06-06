"""
Trainable iMEMORY HetGAT — pure PyTorch, no torch_geometric.

Implements the same 5-step heterogeneous graph attention operations
described in imemory/hetgat_ops.py, but as differentiable nn.Module
operations so the weights can be learned via backpropagation.

Step 1: Per-node-type linear projection    (W_tau @ h_i)
Step 2: Scalar attention weights           (a_tau^T [z_i || z_j], softmax)
Step 3: Message scaling                    (alpha_ij * z_j)
Step 4: Neighborhood aggregation           (sum of scaled messages)
Step 5: Non-linear activation              (ELU element-wise)
Multi-head: K heads averaged (not concatenated) — preserves dim=128 exactly.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITIES = [
    "gene_expr", "scRNA", "methylation", "chromatin", "miRNA", "SNP", "metabolomics"
]
N_MODALITIES = len(MODALITIES)


class HetGATLayer(nn.Module):
    """
    One layer of the Heterogeneous Graph Attention Network.

    Operates on the star-topology patient graph: the patient node attends
    over its 7 modality neighbors, producing an updated patient embedding.
    Dimension is strictly preserved at `dim` throughout (multi-head averaging).
    """

    def __init__(self, dim: int = 128, n_heads: int = 4):
        super().__init__()
        self.dim     = dim
        self.n_heads = n_heads

        # Step 1: Per-type linear projections (W_tau) — the trained weights
        # that give the embedding space biological meaning.
        self.proj_patient  = nn.Linear(dim, dim, bias=False)
        self.proj_modality = nn.Linear(dim, dim, bias=False)

        # Step 2: Attention vectors — (n_heads, 2*dim)
        self.attn_vectors = nn.Parameter(torch.empty(n_heads, 2 * dim))
        nn.init.xavier_uniform_(self.attn_vectors.unsqueeze(0))

        # Step 5 output projection (after aggregation + ELU)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        z_patient:    torch.Tensor,            # (B, dim)
        z_modalities: Dict[str, torch.Tensor], # {mod: (B, dim)}
    ) -> torch.Tensor:
        """Returns updated patient embedding h_prime, shape (B, dim)."""
        # Step 1: project all features
        z_i = self.proj_patient(z_patient)                         # (B, dim)
        z_j_list = [
            self.proj_modality(z_modalities[mod]) for mod in MODALITIES
        ]                                                            # N_mods × (B, dim)
        z_j_stack = torch.stack(z_j_list, dim=1)                   # (B, N, dim)

        # Steps 2–4 per attention head, then average (Step multi-head)
        head_outputs = []
        for h in range(self.n_heads):
            a = self.attn_vectors[h]                                 # (2*dim,)

            # Step 2: attention scores over modalities
            z_i_exp = z_i.unsqueeze(1).expand(-1, N_MODALITIES, -1)  # (B, N, dim)
            concat   = torch.cat([z_i_exp, z_j_stack], dim=-1)       # (B, N, 2*dim)
            scores   = (concat * a).sum(-1)                           # (B, N)

            # Step 2b: softmax normalization
            alphas = F.softmax(scores, dim=-1)                        # (B, N)

            # Steps 3+4: weighted aggregation
            aggregated = (alphas.unsqueeze(-1) * z_j_stack).sum(1)   # (B, dim)
            head_outputs.append(aggregated)

        # Multi-head AVERAGING — strictly preserves dim=128
        S_i = torch.stack(head_outputs, dim=0).mean(0)              # (B, dim)

        # Step 5: ELU + output projection
        h_prime = F.elu(self.out_proj(S_i))                         # (B, dim)
        return h_prime


class iMEMORYHetGAT(nn.Module):
    """
    Full 2-layer iMEMORY HetGAT with classification head.

    Architecture:
        Input: patient (B,128) + 7 modality nodes (B,128) each
        Layer 1: HetGATLayer  → (B, 128)
        Layer 2: HetGATLayer  → (B, 128)   [patient embedding]
        Classifier: LayerNorm → Linear(128,64) → ELU → Dropout → Linear(64,5)
    """

    def __init__(self, dim: int = 128, n_heads: int = 4, n_classes: int = 5):
        super().__init__()
        self.dim = dim

        # Per-modality input projection (applied before the GAT layers)
        self.input_proj_patient = nn.Linear(dim, dim, bias=False)
        self.input_proj_mod = nn.ModuleDict({
            mod: nn.Linear(dim, dim, bias=False) for mod in MODALITIES
        })

        self.layer1 = HetGATLayer(dim, n_heads)
        self.layer2 = HetGATLayer(dim, n_heads)
        self.norm   = nn.LayerNorm(dim)

        # Classification head (used during training; not for embeddings)
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(dim // 2, n_classes),
        )

    def forward(
        self,
        patient_features:  torch.Tensor,            # (B, 128)
        modality_features: Dict[str, torch.Tensor], # {mod: (B, 128)}
    ):
        """
        Returns:
            embedding: (B, 128) — the trained patient embedding
            logits:    (B, n_classes) — for cross-entropy loss
        """
        h_patient = self.input_proj_patient(patient_features)
        h_mods    = {
            mod: self.input_proj_mod[mod](modality_features[mod])
            for mod in MODALITIES
        }

        h1 = self.layer1(h_patient, h_mods)

        # Modality residual: patient context fed back (detached to avoid loop)
        h_mods_updated = {
            mod: h_mods[mod] + 0.1 * h1.detach() for mod in MODALITIES
        }

        h2        = self.layer2(h1, h_mods_updated)
        embedding = self.norm(h2)                    # (B, 128)
        logits    = self.classifier(embedding)       # (B, n_classes)
        return embedding, logits

    def get_W_tau(self) -> np.ndarray:
        """
        Return the patient-node projection weight matrix W_tau as a numpy array.
        Used by therapeutic_signature.inverse_project() to map the 128-d
        therapeutic delta back to gene feature space. Shape: (128, 128).
        """
        return self.layer1.proj_patient.weight.detach().cpu().numpy()

    def embed(
        self,
        patient_features:  torch.Tensor,
        modality_features: Dict[str, torch.Tensor],
    ) -> np.ndarray:
        """Inference-only: return embeddings as numpy array (no gradient)."""
        self.eval()
        with torch.no_grad():
            emb, _ = self.forward(patient_features, modality_features)
        return emb.cpu().numpy()
