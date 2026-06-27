"""
consistency.py - Source Consistency Scorer for DocSentinel
==========================================================

Scores the consistency of a retrieved chunk based on how many
independent retrieval methods found it.

Formula:
    - Found by both BM25 and Semantic  → 1.0
    - Found by only one method         → 0.6
    - Found by neither (shouldn't occur)→ 0.3
"""


def calculate_consistency(in_bm25: bool, in_semantic: bool) -> float:
    """Calculate a consistency score based on retrieval overlap.

    When multiple independent search strategies find the same document,
    it represents a higher confidence that the document is highly relevant.

    Args:
        in_bm25: True if the BM25 lexical search found this chunk.
        in_semantic: True if the Vector semantic search found this chunk.

    Returns:
        Float score: 1.0 (both), 0.6 (single), 0.3 (neither).
    """
    if in_bm25 and in_semantic:
        return 1.0
    elif in_bm25 or in_semantic:
        return 0.6
    else:
        # Fallback edge case: the chunk was somehow retrieved but
        # both flags are False (e.g. metadata error).
        return 0.3
