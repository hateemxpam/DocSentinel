"""
scorer.py - Composite Confidence Scorer for DocSentinel
=======================================================

Calculates the ultimate confidence score for a retrieved chunk by fusing:
    - Reranker Score (from cross-encoder)       [Weight: 35%]
    - Intrinsic Trust Score (from ingestion DB) [Weight: 25%]
    - Freshness Score (based on ingestion date) [Weight: 25%]
    - Consistency Score (retrieval overlap)     [Weight: 15%]
"""

from trust.freshness import calculate_freshness
from trust.consistency import calculate_consistency


def score_chunk(chunk: dict) -> dict:
    """
    Calculate the composite confidence score for a single chunk.

    Args:
        chunk: The dictionary representing the chunk (from retrieval).

    Returns:
        The chunk dictionary with three new keys added:
            - freshness_score: float
            - consistency_score: float
            - confidence_score: float (the composite)
    """
    # 1. Freshness
    date_ingested = chunk.get("date_ingested")
    freshness = calculate_freshness(date_ingested)

    # 2. Consistency
    in_bm25 = chunk.get("in_bm25", False)
    in_semantic = chunk.get("in_semantic", False)
    consistency = calculate_consistency(in_bm25, in_semantic)

    # 3. Trust and Reranker (fallback to 0.0 if missing)
    trust = float(chunk.get("trust_score", 0.0))
    reranker = float(chunk.get("reranker_score", 0.0))

    # Normalize reranker if it's not strictly 0-1 (Cross-encoders output logits).
    # We apply a standard sigmoid function to map real-valued logits to (0, 1).
    import math
    try:
        reranker_norm = 1.0 / (1.0 + math.exp(-reranker))
    except Exception:
        reranker_norm = max(0.0, min(1.0, reranker))
    
    trust = max(0.0, min(1.0, trust))

    # 4. Composite Formula
    confidence = (
        (0.35 * reranker_norm) +
        (0.25 * trust) +
        (0.25 * freshness) +
        (0.15 * consistency)
    )

    # 5. Add fields to chunk
    chunk["freshness_score"] = freshness
    chunk["consistency_score"] = consistency
    chunk["confidence_score"] = confidence

    return chunk


def score_all(chunks: list[dict]) -> list[dict]:
    """
    Score a list of chunks and sort them by confidence.

    Args:
        chunks: List of chunk dictionaries.

    Returns:
        List of chunk dictionaries sorted descending by confidence_score.
    """
    if not chunks:
        return []

    scored_chunks = [score_chunk(chunk.copy()) for chunk in chunks]
    scored_chunks.sort(key=lambda x: x["confidence_score"], reverse=True)

    return scored_chunks
