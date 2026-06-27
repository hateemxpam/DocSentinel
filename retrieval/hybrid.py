"""
hybrid.py - Hybrid retrieval combining BM25 and semantic search via Reciprocal Rank Fusion (RRF).

RRF is a simple yet effective method for merging ranked lists from different retrieval
systems. It doesn't require score normalization because it only relies on rank positions,
making it robust when combining scores from fundamentally different scoring functions
(e.g., BM25's term-frequency scores vs. cosine similarity from dense embeddings).

Reference: Cormack, Clarke & Buettcher (2009) - "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods"
"""

from retrieval.bm25_retriever import bm25_retriever
from retrieval.semantic_retriever import semantic_retriever


def search(query: str, top_k: int = 20) -> list[dict]:
    """
    Run hybrid search combining BM25 (sparse) and semantic (dense) retrieval,
    then merge results using Reciprocal Rank Fusion.

    Args:
        query: The user's search query string.
        top_k: Number of candidates to retrieve from each individual retriever.

    Returns:
        A list of dicts, each containing:
            - chunk_id: Unique identifier for the text chunk.
            - chunk_text: The raw text of the chunk.
            - rrf_score: The fused RRF score (higher is better).
            - in_bm25: Whether this chunk appeared in BM25 results.
            - in_semantic: Whether this chunk appeared in semantic results.
    """
    # --- Step 1: Retrieve candidates from both systems ---
    # Guard against a retriever being None (e.g., index not built yet).
    try:
        bm25_results = bm25_retriever.search(query, top_k=top_k) if bm25_retriever is not None else []
    except Exception as exc:
        print(f"  [hybrid] BM25 retrieval failed: {exc}")
        bm25_results = []

    try:
        semantic_results = semantic_retriever.search(query, top_k=top_k) if semantic_retriever is not None else []
    except Exception as exc:
        print(f"  [hybrid] Semantic retrieval failed: {exc}")
        semantic_results = []

    print(f"  BM25 candidates: {len(bm25_results)}")
    print(f"  Semantic candidates: {len(semantic_results)}")

    # Edge case: nothing from either retriever → nothing to fuse.
    if not bm25_results and not semantic_results:
        return []

    # --- Step 2: Apply Reciprocal Rank Fusion (RRF) ---
    #
    # The RRF formula for a document d across ranked lists R is:
    #
    #     RRF_score(d) = Σ  1 / (k + rank_i(d))
    #                    i
    #
    # where k is a smoothing constant (we use k = 60, the original paper's default).
    #
    # Why k = 60?
    #   • It dampens the influence of high-ranked items so that a single retriever
    #     can't dominate the fused score.  Lower k values would give disproportionate
    #     weight to the top-1 result; higher k values make all ranks almost equally
    #     weighted.  60 is the empirically validated sweet spot from the original paper.
    #
    # Note: rank positions below are 0-indexed from the result lists, so the
    # formula becomes  1 / (i + 1 + 60)  to convert to 1-indexed ranks.

    RRF_K = 60  # Smoothing constant (see explanation above).

    # Accumulator keyed by chunk_id.
    # Each value tracks the running rrf_score, source flags, and text.
    fused: dict[str, dict] = {}

    for i, result in enumerate(bm25_results):
        cid = result["chunk_id"]
        score_increment = 1.0 / (i + 1 + RRF_K)

        if cid not in fused:
            fused[cid] = {
                "chunk_id": cid,
                "chunk_text": result["chunk_text"],
                "rrf_score": 0.0,
                "in_bm25": False,
                "in_semantic": False,
            }

        fused[cid]["rrf_score"] += score_increment
        fused[cid]["in_bm25"] = True
        # Prefer the BM25 text if we see it first; it's the same chunk either way.

    for j, result in enumerate(semantic_results):
        cid = result["chunk_id"]
        score_increment = 1.0 / (j + 1 + RRF_K)

        if cid not in fused:
            fused[cid] = {
                "chunk_id": cid,
                "chunk_text": result["chunk_text"],
                "rrf_score": 0.0,
                "in_bm25": False,
                "in_semantic": False,
            }

        fused[cid]["rrf_score"] += score_increment
        fused[cid]["in_semantic"] = True

    # --- Step 3: Sort by fused score and take top 30 ---
    results = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)[:30]

    print(f"  After RRF merge: {len(results)}")

    return results
