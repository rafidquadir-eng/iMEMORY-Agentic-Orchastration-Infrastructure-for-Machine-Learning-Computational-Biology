"""
discovery/closed_loop.py

The full closed-loop drug discovery pipeline, wired to REAL components.

    1. SIGNATURE   therapeutic Δ (diseased - healthy), inverse-project to the
                   gene/feature space, nominate targets.
    2. SCREEN      query the REAL LINCS L1000 library via the SigCom API
                   (discovery.lincs_api). Offline/unavailable -> synthetic screen.
    3. GENERATE    REAL de novo molecules via SELFIES+RDKit (discovery.denovo_
                   generation), seeded by the top LINCS-hit SMILES. Unavailable
                   -> simple enumeration fallback.
    4. VALIDATE    iMEMORY recovery scoring: each candidate's predicted effect
                   (read-across from parent hits' real signatures) is forward-
                   projected into the embedding space and applied to diseased
                   patients; recovery = angular shift toward healthy (cosine
                   angular displacement). REAL operation; only the embedding
                   space is synthetic in the public demo.
    5. RANK        order candidates by recovery fraction.
    6. CLOSE LOOP  refine generation and repeat if the best recovery is below
                   threshold and iterations remain.

Maps onto the repo's LangGraph state machine:
    plan_task -> 1 | execute_code -> 2,3 | audit_results -> 4 | replan -> 6 | interpret_bio -> 5
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from imemory.therapeutic_signature import (
    compute_therapeutic_delta,
    inverse_project,
    nominate_targets_from_delta,
)
from discovery.insilico_knockout import (
    RecoveryResult,
    apply_signature_perturbation,
    effect_from_signature,
    read_across_signature,
    recovery_score,
)

# Real components (imported lazily inside methods so the module loads even if
# optional deps are missing).


@dataclass
class ScoredCandidate:
    candidate_id: str
    smiles: str
    recovery: RecoveryResult
    qed: float = 0.0
    novelty: float = 1.0


@dataclass
class LoopIteration:
    iteration: int
    n_candidates: int
    best_recovery: float
    best_candidate_id: str


@dataclass
class PipelineResult:
    nominated_targets: List
    lincs_hits: List                 # list of hit objects (real LincsHit or synthetic CompoundHit)
    lincs_mode: str                  # "real" | "synthetic"
    generation_mode: str             # "selfies" | "fallback"
    ranked_candidates: List[ScoredCandidate]
    iterations: List[LoopIteration] = field(default_factory=list)
    converged: bool = False


class ClosedLoopDiscovery:
    def __init__(
        self,
        W_tau: np.ndarray,
        gene_labels: List[str],
        recovery_threshold: float = 0.5,
        max_iterations: int = 3,
        seed: int = 23,
    ):
        assert W_tau.shape == (128, 128)
        assert len(gene_labels) == 128
        self.W_tau = W_tau
        self.gene_labels = gene_labels
        self.recovery_threshold = recovery_threshold
        self.max_iterations = max_iterations
        self.seed = seed

    # -- Step 2: screening (real LINCS, synthetic fallback) -----------------
    def _screen(
        self,
        disease_feature_signature: np.ndarray,
        synthetic_library: Optional[Dict[str, np.ndarray]],
        top_hits: int,
    ) -> Tuple[List, str, Dict[str, np.ndarray], Dict[str, str]]:
        """
        Returns (hits, mode, hit_feature_signatures, hit_smiles).
        Tries the real SigCom LINCS API first; falls back to the synthetic screen.
        """
        try:
            from discovery.lincs_api import LincsUnavailable, query_lincs_reversers

            hits = query_lincs_reversers(
                disease_feature_signature, self.gene_labels, limit=top_hits
            )
            hit_sigs: Dict[str, np.ndarray] = {}
            hit_smiles: Dict[str, str] = {}
            for h in hits:
                # Real hit effect direction in feature space: reversal of disease,
                # scaled by the hit's reversal potential. (Fetching each compound's
                # full L1000 vector is optional; see docs/lincs_api.md /fetch/rank.)
                hit_sigs[h.pert_name] = (
                    -h.reversal_potential * disease_feature_signature
                ).astype("float32")
                if h.smiles:
                    hit_smiles[h.pert_name] = h.smiles
            return hits, "real", hit_sigs, hit_smiles
        except Exception:  # noqa: BLE001  (LincsUnavailable or any network error)
            from discovery.lincs_connectivity import screen_lincs_library

            assert synthetic_library is not None, "Synthetic library required for fallback."
            hits = screen_lincs_library(
                disease_feature_signature, synthetic_library, top_k=top_hits, weighted=True
            )
            hit_sigs = {h.compound_id: synthetic_library[h.compound_id] for h in hits}
            hit_smiles = {h.compound_id: h.smiles_seed for h in hits if h.smiles_seed}
            return hits, "synthetic", hit_sigs, hit_smiles

    # -- Step 3: generation (SELFIES real, fallback) ------------------------
    def _generate(self, seed_smiles: List[str], n: int, diversity: float):
        try:
            from discovery.denovo_generation import SelfiesGenerator

            gen = SelfiesGenerator(seed=self.seed)
            cands = gen.generate(seed_smiles=seed_smiles, n_candidates=n, diversity=diversity)
            return cands, "selfies"
        except Exception:  # noqa: BLE001  (GeneratorUnavailable etc.)
            from discovery.denovo_generation import Candidate

            seeds = seed_smiles or ["O=C(N)c1ccccc1"]
            cands = [
                Candidate(
                    candidate_id=f"IAB-GEN-{i:03d}",
                    smiles=seeds[i % len(seeds)] + "C" * (1 + i % 3),
                    parent_smiles=seeds[i % len(seeds)],
                )
                for i in range(n)
            ]
            return cands, "fallback"

    # -- read-across effect for a candidate ---------------------------------
    def _candidate_effect(self, candidate, hit_sigs: Dict[str, np.ndarray], hit_smiles: Dict[str, str]):
        """
        Estimate the candidate's embedding-space effect via read-across from the
        parent LINCS-hit signatures, then forward-project. Falls back to the mean
        hit effect if RDKit fingerprints are unavailable.
        """
        sigs = list(hit_sigs.values())
        if not sigs:
            return np.zeros(128, dtype="float32")

        cand_fp = getattr(candidate, "fingerprint", None)
        if cand_fp is not None:
            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem

                hit_fps, ordered_sigs = [], []
                for hid, sig in hit_sigs.items():
                    smi = hit_smiles.get(hid)
                    if not smi:
                        continue
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        continue
                    hit_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
                    ordered_sigs.append(sig)
                if hit_fps:
                    feat_sig = read_across_signature(cand_fp, hit_fps, ordered_sigs)
                    return effect_from_signature(feat_sig, self.W_tau)
            except Exception:  # noqa: BLE001
                pass

        # Fallback: mean hit feature signature forward-projected.
        feat_sig = np.mean(sigs, axis=0).astype("float32")
        return effect_from_signature(feat_sig, self.W_tau)

    # -- main loop ----------------------------------------------------------
    def run(
        self,
        diseased_embeddings: np.ndarray,
        healthy_embeddings: np.ndarray,
        synthetic_library: Optional[Dict[str, np.ndarray]] = None,
        top_targets: int = 5,
        top_hits: int = 10,
        candidates_per_round: int = 8,
    ) -> PipelineResult:
        # Step 1: SIGNATURE
        v_d = diseased_embeddings.mean(axis=0)
        v_h = healthy_embeddings.mean(axis=0)
        healthy_centroid = v_h
        delta = compute_therapeutic_delta(v_d, v_h)
        disease_feature_signature = inverse_project(delta, self.W_tau)
        targets = nominate_targets_from_delta(delta, self.W_tau, self.gene_labels, top_targets)

        # Step 2: SCREEN
        hits, lincs_mode, hit_sigs, hit_smiles = self._screen(
            disease_feature_signature, synthetic_library, top_hits
        )
        seed_smiles = [s for s in hit_smiles.values() if s]

        # Steps 3-6
        iterations: List[LoopIteration] = []
        all_scored: List[ScoredCandidate] = []
        diversity = 1.0
        gen_mode = "selfies"
        converged = False

        for it in range(1, self.max_iterations + 1):
            candidates, gen_mode = self._generate(seed_smiles, candidates_per_round, diversity)

            scored: List[ScoredCandidate] = []
            for cand in candidates:
                effect = self._candidate_effect(cand, hit_sigs, hit_smiles)
                perturbed = apply_signature_perturbation(diseased_embeddings, effect, step=1.0)
                rec = recovery_score(
                    diseased_embeddings, perturbed, healthy_centroid, label=cand.candidate_id
                )
                scored.append(
                    ScoredCandidate(
                        candidate_id=cand.candidate_id,
                        smiles=cand.smiles,
                        recovery=rec,
                        qed=getattr(cand, "qed", 0.0),
                        novelty=getattr(cand, "novelty", 1.0),
                    )
                )

            scored.sort(key=lambda s: s.recovery.recovery_fraction, reverse=True)
            all_scored.extend(scored)
            best = scored[0]
            iterations.append(
                LoopIteration(
                    iteration=it,
                    n_candidates=len(scored),
                    best_recovery=best.recovery.recovery_fraction,
                    best_candidate_id=best.candidate_id,
                )
            )

            if best.recovery.recovery_fraction >= self.recovery_threshold:
                converged = True
                break

            # Refine: increase diversity, re-seed from best survivors.
            diversity = min(2.0, diversity + 0.5)
            seed_smiles = [s.smiles for s in scored[: max(2, candidates_per_round // 4)]]

        # Deduplicate, keep best per candidate_id.
        all_scored.sort(key=lambda s: s.recovery.recovery_fraction, reverse=True)
        seen, ranked = set(), []
        for sc in all_scored:
            if sc.candidate_id not in seen:
                ranked.append(sc)
                seen.add(sc.candidate_id)

        return PipelineResult(
            nominated_targets=targets,
            lincs_hits=hits,
            lincs_mode=lincs_mode,
            generation_mode=gen_mode,
            ranked_candidates=ranked,
            iterations=iterations,
            converged=converged,
        )
