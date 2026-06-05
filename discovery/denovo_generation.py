"""
discovery/denovo_generation.py

Real de novo molecular generation using SELFIES + RDKit.

Generation strategy (STONED-SELFIES class; Nigam et al. 2021, Krenn et al. 2020):
SELFIES (SELF-referencing Embedded Strings) is a molecular string representation
in which EVERY string decodes to a syntactically valid molecule. By mutating a
seed molecule's SELFIES tokens (insert / replace / delete from the semantically
robust alphabet), we generate novel, guaranteed-valid molecules WITHOUT training
a model or using a GPU. RDKit then computes real physicochemical properties
(QED drug-likeness, molecular weight, logP, TPSA) and Morgan-fingerprint
similarity for novelty filtering.

This is a genuine generative-chemistry method, not a stub. It is NOT the full
production stack (structure-based diffusion + GPT SMILES refiner + RL reward on
docking/selectivity); that remains a separate, heavier system. The `generate`
method signature is the seam where a learned prior (REINVENT4 / MolGPT) drops in.

Dependencies: `selfies`, `rdkit`. If either is missing, generation raises
GeneratorUnavailable and the caller falls back to a simple enumeration so the
demo still runs; install both for the real path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


class GeneratorUnavailable(RuntimeError):
    """Raised when selfies/rdkit are unavailable for the real generation path."""


# Public, well-known drug-like seed scaffolds (used only when no LINCS-hit
# SMILES are supplied). Not therapeutic recommendations.
_DEFAULT_SEEDS = [
    "O=C(N)c1ccccc1",                 # benzamide
    "c1ccc2[nH]ccc2c1",               # indole
    "c1ccncc1",                       # pyridine
    "C1CCNCC1",                       # piperidine
    "O=S(=O)(N)c1ccccc1",             # benzenesulfonamide
]


@dataclass
class Candidate:
    candidate_id: str
    smiles: str
    parent_smiles: str
    qed: float = 0.0                  # RDKit drug-likeness, [0,1]
    mol_weight: float = 0.0
    logp: float = 0.0
    tpsa: float = 0.0
    novelty: float = 1.0              # 1 - max Tanimoto to any seed
    valid: bool = True
    fingerprint: Optional[object] = field(default=None, repr=False)


def _require_chem():
    try:
        import selfies as sf  # noqa: F401
        from rdkit import Chem  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise GeneratorUnavailable(
            "selfies and/or rdkit not installed (pip install selfies rdkit)."
        ) from exc


def _properties(smiles: str):
    """Return (mol, QED, MW, logP, TPSA) or None if RDKit cannot parse."""
    from rdkit import Chem
    from rdkit.Chem import QED, Descriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return (
        mol,
        float(QED.qed(mol)),
        float(Descriptors.MolWt(mol)),
        float(Descriptors.MolLogP(mol)),
        float(Descriptors.TPSA(mol)),
    )


def _morgan(mol):
    from rdkit.Chem import AllChem
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def _tanimoto(fp_a, fp_b) -> float:
    from rdkit import DataStructs
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


class SelfiesGenerator:
    """Valid-by-construction molecular generator via SELFIES token mutation."""

    def __init__(self, seed: int = 11, qed_floor: float = 0.4, novelty_floor: float = 0.2):
        self.rng = np.random.default_rng(seed)
        self.qed_floor = qed_floor          # drug-likeness gate
        self.novelty_floor = novelty_floor  # minimum novelty vs seeds

    def _mutate_selfies(self, selfies_str: str, alphabet: List[str], n_edits: int) -> str:
        import selfies as sf
        tokens = list(sf.split_selfies(selfies_str))
        for _ in range(n_edits):
            op = self.rng.choice(["insert", "replace", "delete"])
            if op == "delete" and len(tokens) > 2:
                del tokens[self.rng.integers(len(tokens))]
            elif op == "replace" and tokens:
                tokens[self.rng.integers(len(tokens))] = str(self.rng.choice(alphabet))
            else:  # insert
                pos = int(self.rng.integers(len(tokens) + 1))
                tokens.insert(pos, str(self.rng.choice(alphabet)))
        return "".join(tokens)

    def generate(
        self,
        seed_smiles: Optional[List[str]] = None,
        n_candidates: int = 8,
        max_attempts_per_candidate: int = 25,
        diversity: float = 1.0,
    ) -> List[Candidate]:
        """
        Generate novel, valid, drug-like candidate molecules from seed SMILES.

        PRODUCTION SEAM: to use a learned prior instead, replace the mutation
        loop with sampling from REINVENT4 / MolGPT conditioned on the target;
        keep the RDKit property + novelty filtering below unchanged. The return
        type (List[Candidate]) does not change.

        Args:
            seed_smiles:  Seed scaffolds (e.g., top LINCS-hit SMILES). Falls back
                          to public default scaffolds if empty/None.
            n_candidates: Number of accepted candidates to return.
            max_attempts_per_candidate: Mutation attempts before giving up on one.
            diversity:    Scales the number of token edits per mutation.

        Returns:
            List of accepted Candidate molecules (valid, QED >= floor, novel).

        Raises:
            GeneratorUnavailable: if selfies/rdkit are not installed.
        """
        _require_chem()
        import selfies as sf

        seeds = [s for s in (seed_smiles or []) if s] or _DEFAULT_SEEDS
        alphabet = list(sf.get_semantic_robust_alphabet())

        # Precompute seed fingerprints for novelty scoring.
        seed_fps = []
        for s in seeds:
            props = _properties(s)
            if props:
                seed_fps.append(_morgan(props[0]))

        accepted: List[Candidate] = []
        idx = 0
        while len(accepted) < n_candidates and idx < n_candidates * 4:
            parent = seeds[idx % len(seeds)]
            idx += 1
            try:
                parent_selfies = sf.encoder(parent)
            except Exception:  # noqa: BLE001
                continue
            if parent_selfies is None:
                continue

            for _ in range(max_attempts_per_candidate):
                n_edits = max(1, int(round(diversity * self.rng.integers(1, 4))))
                mutant_selfies = self._mutate_selfies(parent_selfies, alphabet, n_edits)
                try:
                    smiles = sf.decoder(mutant_selfies)
                except Exception:  # noqa: BLE001
                    continue
                props = _properties(smiles)
                if props is None:
                    continue
                mol, qed, mw, logp, tpsa = props
                if qed < self.qed_floor:
                    continue
                fp = _morgan(mol)
                novelty = 1.0 - (max((_tanimoto(fp, sfp) for sfp in seed_fps), default=0.0))
                if novelty < self.novelty_floor:
                    continue

                accepted.append(
                    Candidate(
                        candidate_id=f"IAB-GEN-{len(accepted):03d}",
                        smiles=smiles,
                        parent_smiles=parent,
                        qed=round(qed, 3),
                        mol_weight=round(mw, 1),
                        logp=round(logp, 2),
                        tpsa=round(tpsa, 1),
                        novelty=round(novelty, 3),
                        valid=True,
                        fingerprint=fp,
                    )
                )
                break

        return accepted
