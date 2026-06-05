"""Thin ChromaDB wrapper so the vector store backend can be swapped centrally."""
from __future__ import annotations

import chromadb


def get_collection(path: str = "./data/chroma_db", name: str = "imemory_docs"):
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(name)
