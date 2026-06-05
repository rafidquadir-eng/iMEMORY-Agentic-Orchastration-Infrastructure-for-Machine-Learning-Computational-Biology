"""
End-to-end demo (synthetic data only).

Runs one full plan -> execute -> audit -> interpret cycle through the LangGraph
state machine, nominates example targets by embedding distance, and prints an
observability summary. Requires ANTHROPIC_API_KEY; runs without Box or AWS.
"""
from __future__ import annotations

import numpy as np

from agents.orchestrator import PlanningAgent
from agents.executor import ExecutionAgent
from rag.document_loader import BioRAGPipeline
from langraph_wrapper.graph import create_imemory_graph
from observability.tracer import AgentTracer
from imemory.graph_builder import build_synthetic_patient_graph
from imemory.embedding_space import disease_centroid


def main() -> None:
    tracer = AgentTracer()

    # 1. Build a synthetic disease cohort and centroid.
    graphs = [build_synthetic_patient_graph(f"SYN-{i:03d}", seed=i) for i in range(20)]
    centroid = disease_centroid(graphs)

    # 2. Synthetic gene embeddings + labels (illustrative target names).
    rng = np.random.default_rng(7)
    gene_labels = ["Novel Marker A", "Novel Marker B", "GENE_C", "GENE_D", "GENE_E"]
    gene_embeddings = rng.normal(size=(len(gene_labels), centroid.shape[0])).astype("float32")
    # Nudge two genes toward the centroid so the nomination is deterministic-ish.
    gene_embeddings[0] += 0.8 * centroid
    gene_embeddings[1] += 0.6 * centroid

    exec_context = {
        "embeddings": gene_embeddings,
        "gene_labels": gene_labels,
        "disease_centroid": centroid,
    }

    # 3. RAG (ingest is optional; demo proceeds even with an empty index).
    rag = BioRAGPipeline()

    # 4. Wire the agents and compile the graph.
    planner = PlanningAgent(tracer=tracer)
    executor = ExecutionAgent(tracer=tracer)
    app = create_imemory_graph(planner, executor, rag, exec_context)

    # 5. Run.
    initial = {
        "task": "Identify candidate druggable targets in the ATNR SLE endotype "
                "from the trained-immunity embedding space.",
        "iteration_count": 0,
        "a2a_messages": [],
        "audit_failures": [],
    }
    final = app.invoke(initial)

    print("\n=== NOMINATED TARGETS ===")
    print(final["executed_results"].get("nominated_targets"))
    print("\n=== INTERPRETATION ===")
    print(final.get("interpretation"))
    print("\n=== A2A MESSAGE LOG ===")
    for rec in final["a2a_messages"]:
        print(f"  {rec['sender_id']} -> {rec['receiver_id']} [{rec['message_type']}]")
    print("\n=== OBSERVABILITY SUMMARY ===")
    print(tracer.session_summary())


if __name__ == "__main__":
    main()
