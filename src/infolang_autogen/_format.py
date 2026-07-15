"""Rendering helpers shared by the tools and the memory provider."""

from __future__ import annotations

from infolang import RecallResult

__all__ = ["format_recall", "NO_RESULTS"]

NO_RESULTS = "No relevant memories found."


def format_recall(result: RecallResult, *, include_scores: bool = True) -> str:
    """Render a :class:`RecallResult` as a compact, LLM-friendly string.

    Chunks are numbered in rank order. When ``include_scores`` is set, each line
    is prefixed with the similarity score. A weak top match (below the SDK's
    0.85 confidence floor) is annotated so the model can down-weight it.
    """

    if not result.chunks:
        return NO_RESULTS

    lines: list[str] = []
    for i, chunk in enumerate(result.chunks, 1):
        prefix = f"{i}. "
        if include_scores and chunk.score is not None:
            prefix += f"[{chunk.score:.2f}] "
        lines.append(prefix + chunk.text)

    body = "\n".join(lines)
    if result.weak:
        body += "\n(weak match — consider narrowing the query)"
    return body
