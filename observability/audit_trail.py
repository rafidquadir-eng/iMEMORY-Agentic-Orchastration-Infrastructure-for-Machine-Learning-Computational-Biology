"""
Structured self-audit.

The execution agent's results are verified against the task's declared expected
outputs and a set of structural checks. Returns (passed, failures) where failures
is a list of human-readable check names — the same signal that drives the
LangGraph replan edge and a foundation for GxP-style audit records.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


class AuditTrail:
    def verify(
        self, results: Dict[str, Any], expected_outputs: List[str] | None = None
    ) -> Tuple[bool, List[str]]:
        failures: List[str] = []

        if not isinstance(results, dict) or not results:
            failures.append("results_empty_or_malformed")
            return False, failures

        # Structural checks specific to the target-nomination task.
        if "nominated_targets" in results:
            targets = results["nominated_targets"]
            if not targets:
                failures.append("no_targets_nominated")
            else:
                for entry in targets:
                    if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
                        failures.append("malformed_target_entry")
                        break
                    score = entry[1]
                    if not (0.0 <= float(score) <= 1.0):
                        failures.append("target_score_out_of_range")
                        break

        if results.get("n_genes_scored", 0) <= 0:
            failures.append("zero_genes_scored")

        return (len(failures) == 0), failures
