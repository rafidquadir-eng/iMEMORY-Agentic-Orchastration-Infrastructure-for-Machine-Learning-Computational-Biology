"""
imemory/hetgat_ops.py

The five discrete vector operations of the iMEMORY HetGAT forward pass,
implemented in pure NumPy so they are readable without a GPU or PyTorch
installation. Each function is a direct implementation of one step —
the docstring references the architectural step number from the design spec.

Key constraint: the 128-dimensional space is STRICTLY PRESERVED at every step.
  - W_tau is always 128x128 (square): projection never changes dimensionality.
  - Multi-head attention uses AVERAGING, not concatenation, at the final layer.
  - Messages, aggregates, and activations all remain in R^128.

In production, these operations run as parallelized tensor ops via
PyTorch Geometric on the 8xA100 cluster. The NumPy implementations here
are mathematically identical — only the execution engine differs.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Step 1: Node-type-specific linear projection
# ---------------------------------------------------------------------------

def linear_project(h_i: np.ndarray, W_tau: np.ndarray) -> np.ndarray:
    """
    Step 1 — Node-Type Specific Linear Projection.

    Projects a node's raw 128D feature vector into the shared representation
    space using a weight matrix unique to that node type (tau).

    Because W_tau is a SQUARE 128x128 matrix, the output z_i remains strictly
    in R^128. This is a pure linear transformation — rotation and scaling —
    with no dimensionality change.

    Args:
        h_i:    Node feature vector, shape (128,). Raw biological features.
        W_tau:  Type-specific projection matrix, shape (128, 128). Trainable.

    Returns:
        z_i:    Projected node embedding, shape (128,). Shared-space ready.
    """
    assert h_i.shape == (128,), f"Expected (128,), got {h_i.shape}"
    assert W_tau.shape == (128, 128), f"Expected (128, 128), got {W_tau.shape}"
    z_i = W_tau @ h_i                    # (128, 128) x (128,) -> (128,)
    assert z_i.shape == (128,)
    return z_i


# ---------------------------------------------------------------------------
# Step 2: Scalar attention weights
# ---------------------------------------------------------------------------

def attention_score(
    z_i: np.ndarray,
    z_j: np.ndarray,
    a_tau: np.ndarray,
) -> float:
    """
    Step 2a — Raw Attention Score (pre-softmax).

    Computes the unnormalized attention score e_ij between a target node i
    and a neighbor node j by:
      1. Concatenating their 128D vectors into a temporary 256D vector.
      2. Taking the dot product with a trainable 256D attention vector a_tau.
      The result is a single scalar — the biological affinity weight.

    Args:
        z_i:    Projected embedding of target node i, shape (128,).
        z_j:    Projected embedding of neighbor node j, shape (128,).
        a_tau:  Type-specific attention vector, shape (256,). Trainable.

    Returns:
        e_ij:   Scalar attention score (float). Not yet normalized.
    """
    assert z_i.shape == (128,)
    assert z_j.shape == (128,)
    assert a_tau.shape == (256,), f"Expected (256,), got {a_tau.shape}"
    concat = np.concatenate([z_i, z_j])  # (256,) — temporary concatenation
    e_ij = float(a_tau @ concat)         # (256,)^T . (256,) -> scalar
    return e_ij


def softmax_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Step 2b — Softmax Normalization over the Neighborhood.

    Converts raw scores across all neighbors N(i) into probability weights
    that sum to 1. Each alpha_ij is a pure scalar in [0, 1] representing the
    proportion of biological attention node i assigns to neighbor j.

    Uses the numerically stable exp(x - max(x)) formulation.

    Args:
        scores: Raw attention scores e_ij for all neighbors, shape (N,).

    Returns:
        alphas: Normalized attention weights, shape (N,). Sum == 1.0.
    """
    shifted = scores - scores.max()      # numerical stability
    exp_s = np.exp(shifted)
    return exp_s / exp_s.sum()


def compute_attention_weights(
    z_i: np.ndarray,
    neighbor_embeddings: List[np.ndarray],
    a_tau: np.ndarray,
) -> np.ndarray:
    """
    Full Step 2: compute and normalize attention weights for all neighbors.

    Args:
        z_i:                 Target node embedding, shape (128,).
        neighbor_embeddings: List of projected neighbor embeddings, each (128,).
        a_tau:               Attention vector, shape (256,).

    Returns:
        alphas:  Normalized attention weights, shape (N,). Sum == 1.0.
    """
    scores = np.array([
        attention_score(z_i, z_j, a_tau) for z_j in neighbor_embeddings
    ])
    return softmax_normalize(scores)


# ---------------------------------------------------------------------------
# Step 3: Message scaling
# ---------------------------------------------------------------------------

def scale_message(alpha_ij: float, z_j: np.ndarray) -> np.ndarray:
    """
    Step 3 — Message Scaling.

    Scales the neighbor's 128D projected embedding by the scalar attention
    weight alpha_ij. Every one of the 128 values in z_j is multiplied by the
    same scalar. The message m_ij remains strictly in R^128.

    This is element-wise scalar-vector multiplication.

    Args:
        alpha_ij: Scalar attention weight in [0, 1] from Step 2.
        z_j:      Neighbor projected embedding, shape (128,).

    Returns:
        m_ij:     Scaled message, shape (128,).
    """
    m_ij = alpha_ij * z_j               # scalar * (128,) -> (128,)
    assert m_ij.shape == (128,)
    return m_ij


# ---------------------------------------------------------------------------
# Step 4: Neighborhood aggregation
# ---------------------------------------------------------------------------

def aggregate(messages: List[np.ndarray]) -> np.ndarray:
    """
    Step 4 — Neighborhood Aggregation.

    Sums all scaled 128D messages from node i's neighborhood.
    Pure vector addition in R^128. Because adding vectors together does not
    alter their dimensionality, the aggregated sum S_i remains in R^128.

    Args:
        messages: List of scaled messages m_ij, each shape (128,).

    Returns:
        S_i:      Aggregated neighborhood message, shape (128,).
    """
    assert messages, "No messages to aggregate."
    S_i = np.zeros(128, dtype=np.float32)
    for m in messages:
        assert m.shape == (128,)
        S_i += m                         # vector addition, dimensionality preserved
    return S_i


# ---------------------------------------------------------------------------
# Step 5: Non-linear activation
# ---------------------------------------------------------------------------

def elu_activate(S_i: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Step 5 — Non-Linear Activation (ELU).

    Applies Exponential Linear Unit element-wise across all 128 positions
    independently. This introduces the non-linearity that allows the network
    to learn complex biological thresholds.

    ELU: f(x) = x if x > 0, else alpha * (exp(x) - 1)

    The output h'_i remains in R^128 — dimensionality is unchanged.

    Args:
        S_i:    Aggregated neighborhood message, shape (128,).
        alpha:  ELU slope for negative values (default 1.0).

    Returns:
        h_i_prime: Updated node embedding, shape (128,). Ready for next layer.
    """
    positive = S_i * (S_i > 0)
    negative = alpha * (np.exp(np.minimum(S_i, 0)) - 1) * (S_i <= 0)
    h_i_prime = positive + negative
    assert h_i_prime.shape == (128,)
    return h_i_prime


# ---------------------------------------------------------------------------
# Full single-node forward pass (Steps 1-5)
# ---------------------------------------------------------------------------

def hetgat_node_update(
    h_i: np.ndarray,
    neighbor_h: List[np.ndarray],
    W_tau_self: np.ndarray,
    W_tau_neighbor: np.ndarray,
    a_tau: np.ndarray,
) -> np.ndarray:
    """
    Complete five-step HetGAT update for a single node.

    Runs Steps 1-5 in sequence, preserving R^128 throughout.

    Args:
        h_i:              Target node raw features, shape (128,).
        neighbor_h:       List of neighbor raw features, each (128,).
        W_tau_self:       Projection matrix for the target node type (128x128).
        W_tau_neighbor:   Projection matrix for the neighbor node type (128x128).
        a_tau:            Attention vector for this edge relation type (256,).

    Returns:
        h_i_prime:        Updated node embedding, shape (128,).
    """
    # Step 1: Project all nodes into shared space.
    z_i = linear_project(h_i, W_tau_self)
    z_neighbors = [linear_project(h_j, W_tau_neighbor) for h_j in neighbor_h]

    # Step 2: Compute normalized attention weights.
    alphas = compute_attention_weights(z_i, z_neighbors, a_tau)

    # Step 3: Scale each neighbor message.
    messages = [scale_message(float(alpha), z_j)
                for alpha, z_j in zip(alphas, z_neighbors)]

    # Step 4: Aggregate.
    S_i = aggregate(messages)

    # Step 5: Non-linear activation.
    h_i_prime = elu_activate(S_i)

    return h_i_prime


# ---------------------------------------------------------------------------
# Multi-head attention with AVERAGING (not concatenation)
# ---------------------------------------------------------------------------

def multi_head_hetgat_update(
    h_i: np.ndarray,
    neighbor_h: List[np.ndarray],
    W_tau_self_heads: List[np.ndarray],
    W_tau_neighbor_heads: List[np.ndarray],
    a_tau_heads: List[np.ndarray],
) -> np.ndarray:
    """
    Multi-head HetGAT update with AVERAGING across heads.

    CRITICAL ARCHITECTURAL DECISION:
    Standard GNNs concatenate K attention heads, which would expand the
    embedding from 128D to K*128D. iMEMORY uses element-wise AVERAGING
    instead, compressing K parallel 128D outputs back to a single 128D vector.

    This preserves the strict 128D structure across the entire depth of the
    foundation model — no dimensionality creep between layers.

    Args:
        h_i:                     Target node features, shape (128,).
        neighbor_h:              Neighbor features, each (128,).
        W_tau_self_heads:        List of K projection matrices (each 128x128).
        W_tau_neighbor_heads:    List of K projection matrices (each 128x128).
        a_tau_heads:             List of K attention vectors (each 256,).

    Returns:
        h_i_prime:  Averaged multi-head output, shape (128,). Strictly R^128.
    """
    K = len(W_tau_self_heads)
    assert K == len(W_tau_neighbor_heads) == len(a_tau_heads), "Head count mismatch."

    head_outputs = []
    for k in range(K):
        h_k = hetgat_node_update(
            h_i,
            neighbor_h,
            W_tau_self_heads[k],
            W_tau_neighbor_heads[k],
            a_tau_heads[k],
        )
        head_outputs.append(h_k)

    # Average across heads — NOT concatenate. Output remains R^128.
    stacked = np.stack(head_outputs, axis=0)   # (K, 128)
    h_i_prime = stacked.mean(axis=0)           # (128,)
    assert h_i_prime.shape == (128,)
    return h_i_prime


# ---------------------------------------------------------------------------
# Synthetic weight initialization (demo only — not trained weights)
# ---------------------------------------------------------------------------

def init_random_weights(
    n_heads: int = 4,
    dim: int = 128,
    seed: int = 42,
) -> Dict:
    """
    Initialize random HetGAT weights for demonstration purposes.
    These are NOT the trained iMEMORY weights — those are proprietary.
    This function exists solely so the demo pipeline runs end-to-end.

    Args:
        n_heads: Number of attention heads.
        dim:     Embedding dimension (must be 128).
        seed:    RNG seed for reproducibility.

    Returns:
        Dictionary of randomly initialized weight tensors.
    """
    assert dim == 128, "iMEMORY architecture requires dim=128."
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(dim)
    return {
        "W_tau_self":     [rng.normal(scale=scale, size=(dim, dim)).astype("float32")
                           for _ in range(n_heads)],
        "W_tau_neighbor": [rng.normal(scale=scale, size=(dim, dim)).astype("float32")
                           for _ in range(n_heads)],
        "a_tau":          [rng.normal(scale=scale, size=(2 * dim,)).astype("float32")
                           for _ in range(n_heads)],
    }
