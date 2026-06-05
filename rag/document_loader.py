"""
Biomedical RAG pipeline.

Ingests PDF and CSV sources, splits them with the tuned biomedical chunker, and
indexes them in ChromaDB. Retrieval returns the top-n chunks for a query, which
the retriever module then formats into the planning agent's system context.
"""
from __future__ import annotations

from typing import List

from langchain_community.document_loaders import CSVLoader, PyPDFLoader

from .chunker import biomedical_splitter
from .vector_store import get_collection


class BioRAGPipeline:
    def __init__(self, chroma_path: str = "./data/chroma_db"):
        self.splitter = biomedical_splitter()
        self.collection = get_collection(chroma_path)

    def ingest(self, file_path: str) -> int:
        """Load, chunk, and index one document. Returns the number of chunks added."""
        ext = file_path.rsplit(".", 1)[-1].lower()
        loader = PyPDFLoader(file_path) if ext == "pdf" else CSVLoader(file_path)
        docs = loader.load()
        chunks = self.splitter.split_documents(docs)
        if not chunks:
            return 0
        self.collection.add(
            documents=[c.page_content for c in chunks],
            metadatas=[c.metadata for c in chunks],
            ids=[f"{file_path}_{i}" for i in range(len(chunks))],
        )
        return len(chunks)

    def retrieve(self, query: str, n: int = 5) -> List[str]:
        results = self.collection.query(query_texts=[query], n_results=n)
        docs = results.get("documents", [[]])
        return docs[0] if docs else []
