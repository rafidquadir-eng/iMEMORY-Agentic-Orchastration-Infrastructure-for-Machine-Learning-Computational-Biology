"""
discovery/lincs_api.py

Real SigCom LINCS L1000 connectivity client.

Screens an iMEMORY therapeutic signature against the real LINCS L1000 chemical
perturbation library via the SigCom LINCS API (Ma'ayan Lab, ISMMS) and returns
compounds that REVERSE the disease signature (negative two-sided enrichment).

Workflow (see docs/lincs_api.md):
  1. Decompose the therapeutic signature into UP and DOWN gene sets.
  2. Resolve gene symbols -> entity UUIDs   (metadata API /entities/find).
  3. Two-sided rank enrichment              (data API /enrich/ranktwosided).
  4. Resolve result signature UUIDs -> perturbagen metadata (metadata API).

Requires `requests` and network access. If either is unavailable, the client
raises LincsUnavailable, and the caller (closed_loop) falls back to the
synthetic screen in lincs_connectivity.py so the demo still runs offline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

DATA_API = "https://maayanlab.cloud/sigcom-lincs/data-api/api/v1"
METADATA_API = "https://maayanlab.cloud/sigcom-lincs/metadata-api"

# Substrings used to auto-detect the L1000 chemical-perturbation library.
_CHEM_LIBRARY_HINTS = ("l1000", "chem", "pert")


class LincsUnavailable(RuntimeError):
    """Raised when the LINCS API cannot be reached or `requests` is missing."""


@dataclass
class LincsHit:
    signature_uuid: str
    pert_name: str
    pert_id: Optional[str]
    z_sum: float                 # negative => reverses disease (therapeutic)
    reversal_potential: float    # [0,1]; higher = stronger reverser
    smiles: Optional[str] = None


def _session():
    try:
        import requests  # local import keeps requests optional
    except Exception as exc:  # noqa: BLE001
        raise LincsUnavailable("`requests` is not installed.") from exc
    return requests


# ---------------------------------------------------------------------------
# Signature decomposition
# ---------------------------------------------------------------------------

def decompose_signature(
    signed_signature: np.ndarray,
    gene_labels: List[str],
    n_up: int = 100,
    n_down: int = 100,
) -> Tuple[List[str], List[str]]:
    """
    Split a signed gene signature into UP and DOWN gene-symbol sets.

    UP   = genes with the most POSITIVE values (elevated in disease).
    DOWN = genes with the most NEGATIVE values (suppressed in disease).

    Args:
        signed_signature: Signed per-gene signature (disease direction), shape (G,).
        gene_labels:      HGNC gene symbols, length G (must align with signature).
        n_up, n_down:     Number of genes to take for each set.

    Returns:
        (up_genes, down_genes) as lists of gene symbols.
    """
    assert len(gene_labels) == len(signed_signature)
    order = np.argsort(signed_signature)            # ascending
    down_idx = order[:n_down]                        # most negative
    up_idx = order[-n_up:]                            # most positive
    up_genes = [gene_labels[i] for i in up_idx]
    down_genes = [gene_labels[i] for i in down_idx]
    return up_genes, down_genes


# ---------------------------------------------------------------------------
# API steps
# ---------------------------------------------------------------------------

def resolve_genes(symbols: List[str], timeout: int = 30) -> Dict[str, str]:
    """Step 1: resolve gene symbols -> entity UUIDs via the metadata API."""
    requests = _session()
    payload = {
        "filter": {
            "where": {"meta.symbol": {"inq": symbols}},
            "fields": ["id", "meta.symbol"],
        }
    }
    try:
        r = requests.post(f"{METADATA_API}/entities/find", json=payload, timeout=timeout)
        r.raise_for_status()
        rows = r.json()
    except Exception as exc:  # noqa: BLE001
        raise LincsUnavailable(f"entities/find failed: {exc}") from exc

    mapping: Dict[str, str] = {}
    for row in rows:
        sym = (row.get("meta") or {}).get("symbol")
        uuid = row.get("id")
        if sym and uuid:
            mapping[sym] = uuid
    return mapping


def find_chem_library_uuid(timeout: int = 30) -> str:
    """Find the L1000 chemical-perturbation consensus library UUID."""
    requests = _session()
    try:
        r = requests.get(f"{METADATA_API}/libraries", timeout=timeout)
        r.raise_for_status()
        libraries = r.json()
    except Exception as exc:  # noqa: BLE001
        raise LincsUnavailable(f"libraries fetch failed: {exc}") from exc

    for lib in libraries:
        name = ((lib.get("meta") or {}).get("name") or "").lower()
        if all(h in name for h in _CHEM_LIBRARY_HINTS):
            return lib["id"]
    # Fallback: first library that mentions L1000.
    for lib in libraries:
        name = ((lib.get("meta") or {}).get("name") or "").lower()
        if "l1000" in name:
            return lib["id"]
    raise LincsUnavailable("Could not locate an L1000 chemical-perturbation library.")


def rank_twosided(
    up_uuids: List[str],
    down_uuids: List[str],
    library_uuid: str,
    limit: int = 100,
    timeout: int = 60,
) -> List[dict]:
    """Step 2: two-sided rank enrichment via the data API."""
    requests = _session()
    payload = {
        "up_entities": up_uuids,
        "down_entities": down_uuids,
        "limit": limit,
        "database": library_uuid,
    }
    try:
        r = requests.post(f"{DATA_API}/enrich/ranktwosided", json=payload, timeout=timeout)
        r.raise_for_status()
        body = r.json()
    except Exception as exc:  # noqa: BLE001
        raise LincsUnavailable(f"enrich/ranktwosided failed: {exc}") from exc
    # Result key has historically been "results"; tolerate a bare list too.
    return body.get("results", body if isinstance(body, list) else [])


def resolve_signatures(sig_uuids: List[str], timeout: int = 60) -> Dict[str, dict]:
    """Step 3: resolve signature UUIDs -> perturbagen metadata."""
    requests = _session()
    payload = {"filter": {"where": {"id": {"inq": sig_uuids}}}}
    try:
        r = requests.post(f"{METADATA_API}/signatures/find", json=payload, timeout=timeout)
        r.raise_for_status()
        rows = r.json()
    except Exception as exc:  # noqa: BLE001
        raise LincsUnavailable(f"signatures/find failed: {exc}") from exc
    return {row["id"]: (row.get("meta") or {}) for row in rows if row.get("id")}


# ---------------------------------------------------------------------------
# Full query
# ---------------------------------------------------------------------------

def query_lincs_reversers(
    signed_signature: np.ndarray,
    gene_labels: List[str],
    n_up: int = 100,
    n_down: int = 100,
    limit: int = 50,
    library_uuid: Optional[str] = None,
) -> List[LincsHit]:
    """
    Full real-LINCS connectivity query: signature -> ranked reverser compounds.

    Args:
        signed_signature: Signed disease gene signature, shape (G,).
        gene_labels:      HGNC gene symbols aligned to the signature.
        n_up, n_down:     Genes per direction submitted to the enrichment.
        limit:            Max signatures to return.
        library_uuid:     Optional explicit library UUID; auto-detected if None.

    Returns:
        List of LincsHit sorted by reversal_potential (best reverser first).

    Raises:
        LincsUnavailable: if the API/network/requests is unavailable.
    """
    up_genes, down_genes = decompose_signature(signed_signature, gene_labels, n_up, n_down)
    gene_map = resolve_genes(up_genes + down_genes)
    up_uuids = [gene_map[g] for g in up_genes if g in gene_map]
    down_uuids = [gene_map[g] for g in down_genes if g in gene_map]
    if not up_uuids and not down_uuids:
        raise LincsUnavailable("No query genes resolved to LINCS entities.")

    lib = library_uuid or find_chem_library_uuid()
    results = rank_twosided(up_uuids, down_uuids, lib, limit=limit)
    if not results:
        return []

    sig_meta = resolve_signatures([r.get("uuid") or r.get("id") for r in results if r])

    hits: List[LincsHit] = []
    z_values = [float(r.get("zSum", r.get("z_sum", 0.0))) for r in results]
    z_min, z_max = (min(z_values), max(z_values)) if z_values else (0.0, 0.0)
    span = (z_max - z_min) or 1.0

    for r in results:
        sid = r.get("uuid") or r.get("id")
        z = float(r.get("zSum", r.get("z_sum", 0.0)))
        meta = sig_meta.get(sid, {})
        # Most negative z => strongest reverser => reversal_potential near 1.
        reversal = float((z_max - z) / span)
        hits.append(
            LincsHit(
                signature_uuid=sid,
                pert_name=meta.get("pert_name") or meta.get("pert_id") or sid,
                pert_id=meta.get("pert_id"),
                z_sum=round(z, 4),
                reversal_potential=round(reversal, 4),
                smiles=meta.get("SMILES") or meta.get("canonical_smiles"),
            )
        )
    hits.sort(key=lambda h: h.reversal_potential, reverse=True)
    return hits
