"""
pipeline.py - Trust Layer Orchestrator for DocSentinel
======================================================

Takes retrieved chunks, scores their trustworthiness, and passes them
through the quality gate.
"""

from trust.scorer import score_all
from trust.gate import apply_gate


def run_trust_layer(chunks: list[dict]) -> dict:
    """
    Run the full trust evaluation pipeline.

    1. Scores all chunks (freshness, consistency, composite confidence).
    2. Sorts chunks by composite confidence.
    3. Applies the quality gate (average of top 5 must beat 0.45).

    Args:
        chunks: List of retrieved candidate chunks.

    Returns:
        A gate result dictionary containing 'passed', 'reason',
        'avg_confidence', and the evaluated 'chunks'.
    """
    print("\n--- Trust & Verification Layer ---")
    
    if not chunks:
        print("[trust] No chunks provided.")
        return {
            "passed": False,
            "reason": "No chunks found.",
            "avg_confidence": 0.0,
            "chunks": []
        }

    # Step 1: Score and sort chunks
    scored_chunks = score_all(chunks)
    
    # Extract just the scores for logging
    confidences = [round(c.get("confidence_score", 0.0), 2) for c in scored_chunks]
    # Print the top 5 scores
    print(f"  Chunk confidences: {confidences[:5]}...")

    # Step 2: Apply quality gate
    gate_result = apply_gate(scored_chunks, threshold=0.45)
    
    avg_conf = gate_result["avg_confidence"]
    passed = gate_result["passed"]
    
    print(f"  Avg top-5 confidence: {avg_conf:.2f}")
    
    if passed:
        print("  Gate: PASSED [OK] -> Proceeding to Generation")
    else:
        print("  Gate: BLOCKED [FAIL] -> Source material insufficient")

    return gate_result
