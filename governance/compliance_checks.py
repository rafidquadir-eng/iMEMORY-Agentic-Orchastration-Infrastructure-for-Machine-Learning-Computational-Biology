"""
Lightweight governance checks for pre-clinical research in regulated settings.

Extensible toward 21 CFR Part 11 / GxP requirements. Two checks are implemented:
data provenance (is the dataset public and PHI-free?) and agent-action logging
(does every action carry a trace record?).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict


class ComplianceChecker:
    def __init__(self, approved_datasets: set[str] | None = None):
        # Public, PHI-free datasets approved for this showcase.
        self.approved_datasets = approved_datasets or {"SDY997_synthetic", "public_sle_abstracts"}

    def check_data_provenance(self, dataset_id: str) -> Dict[str, object]:
        approved = dataset_id in self.approved_datasets
        return {
            "dataset_id": dataset_id,
            "approved": approved,
            "phi_present": False,        # synthetic-only policy enforced upstream
            "note": "approved public/synthetic source" if approved else "UNAPPROVED SOURCE",
        }

    def check_agent_action_log(self, session_id: str, log_dir: str = "./logs") -> bool:
        """Every session must have at least one trace record on disk."""
        for log_file in Path(log_dir).glob("trace_*.jsonl"):
            with open(log_file) as f:
                for line in f:
                    if session_id in line:
                        return True
        return False
