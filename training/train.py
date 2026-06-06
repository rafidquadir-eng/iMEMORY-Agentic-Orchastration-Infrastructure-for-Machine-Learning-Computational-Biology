"""
HetGAT training loop.

Trains the iMEMORYHetGAT on the 5-cluster structured synthetic cohort.
Saves the trained model checkpoint and pre-computed embeddings for the
diseased (SLEDAI 13 NR) and healthy clusters, which are loaded by the
downstream discovery pipeline.

Expected runtime: ~2-4 minutes on CPU, ~20 seconds on GPU.
Expected validation accuracy: 90-97% (clusters are well-separated).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from training.synthetic_patients import (
    MODALITIES,
    build_structured_cohort,
    get_diseased_healthy_graphs,
    graphs_to_tensors,
)
from training.hetgat_model import iMEMORYHetGAT

# ---------------------------------------------------------------------------
# Checkpoint paths
# ---------------------------------------------------------------------------
CHECKPOINT_DIR      = Path("data/checkpoints")
CHECKPOINT_PATH     = CHECKPOINT_DIR / "hetgat_model.pt"
EMBEDDINGS_DISEASED = CHECKPOINT_DIR / "embeddings_diseased.npy"
EMBEDDINGS_HEALTHY  = CHECKPOINT_DIR / "embeddings_healthy.npy"
ALL_EMBEDDINGS_PATH = CHECKPOINT_DIR / "all_embeddings.npy"
ALL_LABELS_PATH     = CHECKPOINT_DIR / "all_labels.npy"
W_TAU_PATH          = CHECKPOINT_DIR / "W_tau.npy"


def run_training(
    n_epochs:   int   = 120,
    lr:         float = 1e-3,
    batch_size: int   = 32,
    seed:       int   = 42,
    verbose:    bool  = True,
) -> Dict:
    """
    Train the HetGAT, save checkpoint and embeddings.
    Returns a dict with training history and final metrics.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build structured synthetic data ──────────────────────────────────────
    if verbose:
        print("Building structured synthetic cohort (5 clusters)...")
    graphs, labels = build_structured_cohort(seed=seed)
    patient_t, mod_t = graphs_to_tensors(graphs)
    labels_t = torch.tensor(labels, dtype=torch.long)

    # ── Train / validation split (stratified) ────────────────────────────────
    idx = np.arange(len(graphs))
    train_idx, val_idx = train_test_split(
        idx, test_size=0.2, stratify=labels, random_state=seed
    )

    def _batch_iter(indices, shuffle: bool = True):
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch = indices[start: start + batch_size]
            yield (
                patient_t[batch].to(device),
                {mod: mod_t[mod][batch].to(device) for mod in MODALITIES},
                labels_t[batch].to(device),
            )

    # ── Model, optimizer, loss ────────────────────────────────────────────────
    model     = iMEMORYHetGAT(dim=128, n_heads=4, n_classes=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    history: Dict = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state   = None

    if verbose:
        print(f"Training on {device} | {len(train_idx)} train, {len(val_idx)} val")
        print(f"Epochs: {n_epochs} | Batch size: {batch_size} | LR: {lr}")
        print("-" * 55)

    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        # ── Training pass ──────────────────────────────────────────────────
        model.train()
        train_losses = []
        for p, m, y in _batch_iter(train_idx, shuffle=True):
            optimizer.zero_grad()
            _, logits = model(p, m)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # ── Validation pass ───────────────────────────────────────────────
        model.eval()
        val_losses, val_preds, val_true = [], [], []
        with torch.no_grad():
            for p, m, y in _batch_iter(val_idx, shuffle=False):
                _, logits = model(p, m)
                val_losses.append(criterion(logits, y).item())
                val_preds.extend(logits.argmax(-1).cpu().tolist())
                val_true.extend(y.cpu().tolist())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        val_acc    = accuracy_score(val_true, val_preds)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"Epoch {epoch:3d}/{n_epochs} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | "
                  f"val_acc={val_acc:.3f}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\nTraining complete in {elapsed:.1f}s | Best val_acc={best_val_acc:.3f}")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": best_state,
            "model_config":     {"dim": 128, "n_heads": 4, "n_classes": 5},
            "best_val_acc":     best_val_acc,
            "history":          history,
        },
        CHECKPOINT_PATH,
    )
    if verbose:
        print(f"Model checkpoint saved: {CHECKPOINT_PATH}")

    # ── Save all embeddings for visualisation + downstream pipeline ───────────
    _save_cluster_embeddings(model, graphs, labels, patient_t, mod_t, device, verbose)

    return {
        "best_val_acc":     best_val_acc,
        "elapsed_seconds":  elapsed,
        "history":          history,
        "checkpoint_path":  str(CHECKPOINT_PATH),
    }


def _save_cluster_embeddings(
    model, graphs, labels, patient_t, mod_t, device, verbose=True
):
    """Save pre-computed embeddings (all patients + diseased/healthy) and W_tau."""
    # ── All 220 patients ──────────────────────────────────────────────────────
    all_emb = model.embed(
        patient_t.to(device),
        {mod: mod_t[mod].to(device) for mod in MODALITIES},
    )
    np.save(ALL_EMBEDDINGS_PATH, all_emb)
    np.save(ALL_LABELS_PATH, labels)

    # ── Diseased (label=3) and healthy (label=0) subsets ─────────────────────
    diseased_idx = np.where(labels == 3)[0]
    healthy_idx  = np.where(labels == 0)[0]

    def _embed_idx(idx):
        pt = patient_t[idx].to(device)
        mt = {mod: mod_t[mod][idx].to(device) for mod in MODALITIES}
        return model.embed(pt, mt)

    diseased_emb = _embed_idx(diseased_idx)
    healthy_emb  = _embed_idx(healthy_idx)
    W_tau        = model.get_W_tau()

    np.save(EMBEDDINGS_DISEASED, diseased_emb)
    np.save(EMBEDDINGS_HEALTHY,  healthy_emb)
    np.save(W_TAU_PATH,          W_tau)

    if verbose:
        print(f"Saved all embeddings:      {ALL_EMBEDDINGS_PATH}  "
              f"shape={all_emb.shape}")
        print(f"Saved diseased embeddings: {EMBEDDINGS_DISEASED}  "
              f"shape={diseased_emb.shape}")
        print(f"Saved healthy embeddings:  {EMBEDDINGS_HEALTHY}  "
              f"shape={healthy_emb.shape}")
        print(f"Saved W_tau:               {W_TAU_PATH}  "
              f"shape={W_tau.shape}")


def load_trained_embeddings() -> Dict:
    """
    Load pre-computed embeddings and W_tau from checkpoint.
    Raises FileNotFoundError if training has not been run yet.
    """
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {CHECKPOINT_PATH}. "
            "Run `python -m training` first."
        )
    result = {
        "diseased_embeddings": np.load(EMBEDDINGS_DISEASED),
        "healthy_embeddings":  np.load(EMBEDDINGS_HEALTHY),
        "W_tau":               np.load(W_TAU_PATH),
    }
    if ALL_EMBEDDINGS_PATH.exists():
        result["all_embeddings"] = np.load(ALL_EMBEDDINGS_PATH)
        result["all_labels"]     = np.load(ALL_LABELS_PATH)
    return result


def ensure_trained(verbose: bool = True) -> Dict:
    """
    Check for checkpoint; train if missing. Returns loaded embeddings.
    Called by the discovery pipeline to guarantee trained embeddings exist.
    """
    if CHECKPOINT_PATH.exists():
        if verbose:
            print("Trained model checkpoint found. Loading embeddings...")
        return load_trained_embeddings()
    if verbose:
        print("No checkpoint found. Training HetGAT now...")
    run_training(verbose=verbose)
    return load_trained_embeddings()


if __name__ == "__main__":
    run_training(verbose=True)
