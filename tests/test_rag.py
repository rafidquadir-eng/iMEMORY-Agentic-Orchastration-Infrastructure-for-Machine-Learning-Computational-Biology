"""Chunking + retrieval on a tiny synthetic document (no network)."""
from rag.chunker import biomedical_splitter


def test_chunker_overlap_and_size():
    splitter = biomedical_splitter()
    text = ("Trained immunity is an epigenetic state. " * 200)
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    # Overlap should make consecutive chunks share some tail/head content.
    assert any(chunks[i][-20:] in chunks[i + 1] or chunks[i + 1][:20] in chunks[i]
               for i in range(len(chunks) - 1))
