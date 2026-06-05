"""
Per-call observability.

Every agent model call records input/output tokens, estimated cost, and latency
to a daily JSONL file (one record per line, queryable with jq or pandas).
session_summary() aggregates cost and latency for a run — the data you need to
reason about the cost/reliability/latency trade-offs of a multi-agent system.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# Illustrative Claude Sonnet pricing (USD per million tokens). Override per model as needed.
PRICE_PER_M = {"input": 3.0, "output": 15.0}


class AgentTracer:
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = str(uuid.uuid4())[:8]

    def trace(
        self,
        agent: str,
        task: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        audit_passed: bool,
        model: str = "claude-sonnet-4-20250514",
    ) -> Dict:
        cost = (tokens_in * PRICE_PER_M["input"] + tokens_out * PRICE_PER_M["output"]) / 1_000_000
        record = {
            "session_id": self.session_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "agent": agent,
            "model": model,
            "task": task[:120],
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": round(latency_ms, 1),
            "cost_usd": round(cost, 6),
            "audit_passed": audit_passed,
        }
        log_file = self.log_dir / f"trace_{datetime.now(timezone.utc).date()}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def session_summary(self) -> Dict:
        """Aggregate cost and latency for the current session across all daily logs."""
        records: List[Dict] = []
        for log_file in self.log_dir.glob("trace_*.jsonl"):
            with open(log_file) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec["session_id"] == self.session_id:
                        records.append(rec)
        if not records:
            return {"session_id": self.session_id, "calls": 0}
        return {
            "session_id": self.session_id,
            "calls": len(records),
            "total_cost_usd": round(sum(r["cost_usd"] for r in records), 6),
            "total_tokens_in": sum(r["tokens_in"] for r in records),
            "total_tokens_out": sum(r["tokens_out"] for r in records),
            "mean_latency_ms": round(sum(r["latency_ms"] for r in records) / len(records), 1),
            "audit_pass_rate": round(sum(r["audit_passed"] for r in records) / len(records), 3),
        }
