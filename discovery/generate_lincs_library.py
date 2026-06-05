"""
data/synthetic/generate_lincs_library.py

Generate a synthetic LINCS L1000-style compound signature library for the demo.

The real LINCS L1000 corpus contains 30,000+ compound/perturbagen signatures
over 978 landmark genes. This generator produces a structurally analogous
synthetic library — N compound signatures in the 128-dimensional biological
feature space used by iMEMORY's inverse projection — so the connectivity screen
runs end-to-end without the real (large, separately-licensed) dataset.

No real compound data is used. Signatures are random draws. The demo injects a
handful of planted reverser compounds at run time (see
discovery.lincs_connectivity.inject_planted_reversers) so the screen returns a
meaningful, reproducible top set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

OUT_DIR = Path(__file__).parent
N_COMPOUNDS = 30_000
DIM = 128


def generate_library(n: int = N_COMPOUNDS, dim: int = DIM, seed: int = 99) -> Dict[str, np.ndarray]:
    """
    Build an in-memory synthetic compound signature library.

    Returns:
        Mapping compound_id -> signature vector (shape (dim,), float32).
    """
    rng = np.random.default_rng(seed)
    library: Dict[str, np.ndarray] = {}
    for i in range(n):
        # Sparse-ish signatures: most features near zero, a few strongly perturbed,
        # mimicking how a real perturbagen moves only part of the transcriptome.
        sig = rng.normal(scale=0.3, size=dim).astype("float32")
        n_strong = rng.integers(3, 12)
        idx = rng.choice(dim, size=n_strong, replace=False)
        sig[idx] += rng.normal(scale=1.5, size=n_strong).astype("float32")
        library[f"LINCS-{i:05d}"] = sig
    return library


def save_library(library: Dict[str, np.ndarray], path: Path | None = None) -> Path:
    """Persist the library to a compressed .npz for reuse across demo runs."""
    path = path or (OUT_DIR / "lincs_library.npz")
    ids = list(library.keys())
    matrix = np.stack([library[i] for i in ids])
    np.savez_compressed(path, ids=np.array(ids), signatures=matrix)
    return path


def load_library(path: Path | None = None) -> Dict[str, np.ndarray]:
    """Load a previously saved library."""
    path = path or (OUT_DIR / "lincs_library.npz")
    data = np.load(path, allow_pickle=True)
    return {cid: sig for cid, sig in zip(data["ids"], data["signatures"])}


def main() -> None:
    lib = generate_library()
    out = save_library(lib)
    print(f"Wrote synthetic LINCS library: {len(lib)} compounds x {DIM} features -> {out}")


if __name__ == "__main__":
    main()
