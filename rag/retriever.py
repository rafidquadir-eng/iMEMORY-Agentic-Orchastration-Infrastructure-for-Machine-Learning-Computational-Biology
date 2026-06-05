"""Format retrieved chunks into a context block for the planning agent's prompt."""
from __future__ import annotations

from typing import List


def format_context(chunks: List[str], max_chars: int = 4000) -> str:
    """Concatenate retrieved chunks into a bounded, labeled context block."""
    if not chunks:
        return ""
    blocks, total = [], 0
    for i, chunk in enumerate(chunks, 1):
        piece = f"[ctx {i}] {chunk.strip()}"
        if total + len(piece) > max_chars:
            break
        blocks.append(piece)
        total += len(piece)
    return "\n\n".join(blocks)
