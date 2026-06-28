"""
gate.py - Quality Assurance Gate for DocSentinel
================================================

Examines the top retrieved chunks and decides whether they have enough
collective confidence to be sent to the LLM for answer generation.
If the confidence is too low, it blocks the query to prevent hallucination.
"""


def apply_gate(scored_chunks: list[dict], threshold: float = 0.50) -> dict:
    """
    Evaluate the top chunks against a confidence threshold.

    Calculates the average confidence score of the top 5 chunks.
    If there are fewer than 5 chunks, it averages the ones available.
    If the average is below the threshold, it blocks the pipeline.

    Args:
        scored_chunks: List of chunk dicts (must have 'confidence_score').
        threshold: The minimum acceptable average confidence (0 to 1).

    Returns:
        Dict with keys:
            - passed: bool (True if avg >= threshold)
            - reason: str (Explanation of the decision)
            - avg_confidence: float (The calculated average)
            - chunks: list[dict] (The chunks if passed, empty if blocked)
    """
    if not scored_chunks:
        return {
            "passed": False,
            "reason": "No chunks provided to the gate.",
            "avg_confidence": 0.0,
            "chunks": []
        }

    # Grab up to the top 5 chunks for evaluation
    top_n = scored_chunks[:5]
    
    # Calculate average confidence
    total_confidence = sum(chunk.get("confidence_score", 0.0) for chunk in top_n)
    avg_confidence = total_confidence / len(top_n)

    if avg_confidence >= threshold:
        return {
            "passed": True,
            "reason": "Top chunks passed the confidence threshold.",
            "avg_confidence": avg_confidence,
            "chunks": scored_chunks
        }
    else:
        return {
            "passed": False,
            "reason": "Insufficient source confidence. Aborting generation to prevent hallucination.",
            "avg_confidence": avg_confidence,
            "chunks": []
        }
