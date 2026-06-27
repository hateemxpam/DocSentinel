"""
reranker.py - Cross-encoder reranker for hybrid retrieval candidates.

Cross-encoders vs. Bi-encoders
------------------------------
Bi-encoders (used in semantic retrieval) encode the query and each document
*independently* into fixed-size vectors, then compare them via cosine similarity.
This is fast (vectors are pre-computed) but loses fine-grained token-level
interactions between the query and the document.

Cross-encoders, on the other hand, feed the query AND the document *together*
through a single transformer pass.  This lets every query token attend to every
document token (and vice versa), producing much more accurate relevance scores.
The trade-off is speed: cross-encoders can't pre-compute anything, so they must
run inference for every (query, document) pair at query time.

Strategy: Use a bi-encoder to cheaply retrieve a broad candidate set, then use
a cross-encoder to *rerank* only those top candidates for maximum precision.
This is the standard "retrieve-then-rerank" pipeline.
"""

from sentence_transformers import CrossEncoder

# ---------------------------------------------------------------------------
# Module-level model loading – happens exactly once on first import.
# ---------------------------------------------------------------------------
model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

try:
    reranker_model: CrossEncoder | None = CrossEncoder(model_name)
    print(f"Cross-encoder reranker loaded: {model_name}")
except Exception as exc:
    print(
        f"[reranker] Failed to load cross-encoder '{model_name}': {exc}. "
        "Reranking will be skipped (candidates returned as-is)."
    )
    reranker_model = None


def rerank(query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
    """
    Rerank hybrid retrieval candidates using a cross-encoder model.

    The cross-encoder scores each (query, candidate_text) pair jointly,
    capturing deep token-level interactions that bi-encoders miss.

    Args:
        query: The original user query.
        candidates: List of candidate dicts from hybrid search.  Each dict is
                     expected to have at least 'chunk_id' and 'chunk_text' keys
                     (plus any extras like rrf_score, in_bm25, in_semantic).
        top_k: Number of top results to return after reranking.

    Returns:
        A list of the top_k candidate dicts, sorted by reranker_score
        descending.  Every original key is preserved; a new 'reranker_score'
        key is added.
    """
    # --- Guard: model unavailable ---
    if reranker_model is None:
        print(
            "  [reranker] WARNING: Cross-encoder model is not loaded. "
            "Returning candidates without reranking."
        )
        return candidates[:top_k]

    # --- Guard: empty input ---
    if not candidates:
        return []

    try:
        # Build (query, document) pairs for the cross-encoder.
        # The model processes all pairs in a single batch for efficiency.
        pairs = [(query, candidate["chunk_text"]) for candidate in candidates]

        # Run cross-encoder inference.
        # Returns an array of float scores, one per pair.
        scores = reranker_model.predict(pairs)

        # Attach the reranker score to each candidate dict.
        # We preserve ALL existing keys (chunk_id, chunk_text, rrf_score,
        # in_bm25, in_semantic, etc.) and simply add the new score.
        for candidate, score in zip(candidates, scores):
            candidate["reranker_score"] = float(score)

        # Sort by cross-encoder score (highest = most relevant) and trim.
        results = sorted(candidates, key=lambda x: x["reranker_score"], reverse=True)[:top_k]

        print(f"  After reranking: {len(results)}")

        return results

    except Exception as exc:
        # Graceful degradation: if anything goes wrong during reranking,
        # fall back to the original ordering (which is already RRF-sorted).
        print(f"  [reranker] Reranking failed: {exc}. Returning candidates as-is.")
        return candidates[:top_k]
