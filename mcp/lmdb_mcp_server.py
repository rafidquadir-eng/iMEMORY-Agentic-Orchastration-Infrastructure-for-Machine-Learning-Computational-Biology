"""
LMDB graph-tensor store, exposed to agents as an MCP-compliant tool.

IMPORTANT: LMDB here is key-value storage for serialized HetGAT graph *tensors*,
backed up to S3/GCP. It is NOT a property-graph database and does not run
Cypher-style queries. For relationship queries over biological graphs, a graph
database (Amazon Neptune / Neo4j) is the natural upstream layer.
"""
from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional

import lmdb


class LMDBGraphStore:
    """Fast, memory-mapped storage for HetGAT graph tensors."""

    def __init__(self, db_path: str = "./data/graphs.lmdb", map_size_gb: int = 10):
        self.db_path = db_path
        self.env = lmdb.open(db_path, map_size=map_size_gb * 1024**3)

    def store_graph(self, graph_id: str, graph_data: Dict[str, Any]) -> None:
        with self.env.begin(write=True) as txn:
            txn.put(graph_id.encode(), pickle.dumps(graph_data))

    def load_graph(self, graph_id: str) -> Optional[Dict[str, Any]]:
        with self.env.begin() as txn:
            raw = txn.get(graph_id.encode())
        return pickle.loads(raw) if raw else None

    def exists(self, graph_id: str) -> bool:
        with self.env.begin() as txn:
            return txn.get(graph_id.encode()) is not None

    def list_graph_ids(self, limit: int = 1000) -> List[str]:
        ids: List[str] = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                ids.append(key.decode())
                if len(ids) >= limit:
                    break
        return ids

    # -- MCP tool definitions ----------------------------------------------
    @staticmethod
    def load_tool_definition() -> Dict[str, Any]:
        return {
            "name": "load_graph",
            "description": "Load a HetGAT graph tensor from the LMDB store by graph_id.",
            "input_schema": {
                "type": "object",
                "properties": {"graph_id": {"type": "string"}},
                "required": ["graph_id"],
            },
        }

    @staticmethod
    def store_tool_definition() -> Dict[str, Any]:
        return {
            "name": "store_graph",
            "description": "Persist a HetGAT graph tensor to the LMDB store under graph_id.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "graph_id": {"type": "string"},
                    "graph_data": {"type": "object"},
                },
                "required": ["graph_id", "graph_data"],
            },
        }
