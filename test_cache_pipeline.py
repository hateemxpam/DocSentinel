"""
test_cache_pipeline.py - Semantic Cache Verification Test
=========================================================

Ties the Cache wrapper with the full Retrieval -> Trust -> Generation RAG pipeline.
Verifies Cache MISS (first run) and Cache HIT (subsequent runs with semantically similar wording).
"""

from retrieval.retriever import retrieve
from trust.pipeline import run_trust_layer
from generation.pipeline import generate_response
from cache.pipeline import run_with_cache
from cache.db import get_db_connection

def full_pipeline(query: str) -> dict:
    """End-to-End DocSentinel RAG Pipeline."""
    chunks = retrieve(query)
    trust_result = run_trust_layer(chunks)
    return generate_response(query, trust_result)

def clear_cache_table():
    """Clear query_cache to ensure a deterministic test."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("TRUNCATE TABLE query_cache RESTART IDENTITY;")
        conn.commit()
        cur.close()
        conn.close()
        print("\n[Test Setup] Cache table truncated.")
    except Exception as exc:
        print(f"[Test Setup] Error truncating table: {exc}")

def main():
    clear_cache_table()

    # 1. First Run (Cache MISS)
    query_1 = "What is the right to lodge a complaint under GDPR?"
    print(f"\n--- RUN 1 (Fresh query: '{query_1}') ---")
    res_1 = run_with_cache(query_1, full_pipeline)
    print(f"Run 1 Status: {res_1['status']}")

    # 2. Second Run (Exact query: Cache HIT)
    print(f"\n--- RUN 2 (Exact same query: '{query_1}') ---")
    res_2 = run_with_cache(query_1, full_pipeline)
    print(f"Run 2 Status: {res_2['status']}")

    # 3. Third Run (Semantically similar query: Cache HIT)
    query_3 = "What is the right to file a complaint under the GDPR?"
    print(f"\n--- RUN 3 (Semantically similar query: '{query_3}') ---")
    res_3 = run_with_cache(query_3, full_pipeline)
    print(f"Run 3 Status: {res_3['status']}")

    # Verification checks
    print("\n" + "="*50)
    print("VERIFICATION RESULTS")
    print("="*50)
    print(f"Run 1 successfully stored response? {res_1['status'] == 'SUCCESS'}")
    print(f"Run 2 response matched Run 1? {res_2['answer'] == res_1['answer']}")
    print(f"Run 3 response matched Run 1? {res_3['answer'] == res_1['answer']}")

if __name__ == "__main__":
    main()
