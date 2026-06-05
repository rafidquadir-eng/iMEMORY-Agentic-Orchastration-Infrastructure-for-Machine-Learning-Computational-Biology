"""
Chunking configuration for biomedical documents.

A 1000-token chunk with 200-token overlap is tuned to keep a paper's
methods-section context intact: experimental design, sample sizes, and the
statistical test that produced a result tend to span several sentences, and a
smaller chunk would sever a result from the method that produced it.
"""
from __future__ import annotations

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[no-redef]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def biomedical_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", ",", " "],
    )
