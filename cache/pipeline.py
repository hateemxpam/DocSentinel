"""
pipeline.py - Semantic Cache Pipeline Wrapper for DocSentinel
=============================================================

Wraps compliance query execution with semantic caching checks,
saving successful runs and serving cached hits.
"""

from typing import Callable
from cache.semantic_cache import get_cached, store_cache


def run_with_cache(query: str, full_pipeline_fn: Callable[[str], dict]) -> dict:
    """
    Execute query against semantic cache first, falling back to full pipeline.

    Args:
        query: User compliance query.
        full_pipeline_fn: Callable representing the RAG E2E pipeline.

    Returns:
        dict: Response dictionary (from cache or fresh run).
    """
    print("\n--- Cache Layer Check ---")
    
    # 1. Look up query in the semantic cache
    cached_response = get_cached(query, threshold=0.92)
    
    if cached_response is not None:
        print("Cache HIT — returning cached response")
        return cached_response
        
    # 2. On cache miss, run the actual query pipeline
    print("Cache MISS — running full pipeline")
    response = full_pipeline_fn(query)
    
    # 3. Only cache positive SUCCESS responses
    if response.get("status") == "SUCCESS":
        store_cache(query, response)
        
    return response
