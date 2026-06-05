"""Tracer writes well-formed records and session_summary aggregates them."""
import tempfile

from observability.tracer import AgentTracer


def test_trace_and_summary():
    with tempfile.TemporaryDirectory() as d:
        tracer = AgentTracer(log_dir=d)
        tracer.trace("planning_agent", "task A", 1000, 200, 850.0, True)
        tracer.trace("execution_agent", "task B", 500, 100, 400.0, True)
        summary = tracer.session_summary()
        assert summary["calls"] == 2
        assert summary["total_tokens_in"] == 1500
        assert summary["audit_pass_rate"] == 1.0
        assert summary["total_cost_usd"] > 0
