"""
test_generation_pipeline.py - End-to-End Integration Test for DocSentinel
==========================================================================

Ties together Retrieval -> Trust -> Generation layers.
Tests a valid query (GDPR fines) and an irrelevant query (Sharks).
"""

import json
from retrieval.retriever import retrieve
from trust.pipeline import run_trust_layer
from generation.pipeline import generate_response


def run_test(query: str):
    print(f"\n{'='*80}")
    print(f"RUNNING E2E TEST FOR QUERY: '{query}'")
    print(f"{'='*80}")

    # 1. Retrieve candidates
    print("[Pipeline] Step 1: Retrieval...")
    retrieved_chunks = retrieve(query)

    # 2. Run trust layer
    print("[Pipeline] Step 2: Trust Layer Verification...")
    trust_result = run_trust_layer(retrieved_chunks)

    # 3. Run generation layer
    print("[Pipeline] Step 3: LLM Generation...")
    response = generate_response(query, trust_result)

    # 4. Display results
    print(f"\n{'-'*80}")
    print("FINAL PIPELINE RESULT:")
    print(f"{'-'*80}")
    print(json.dumps(response, indent=2, default=str))


def main():
    # Test 1: Valid GDPR Compliance Query
    # Expected outcome: Gate passes, LLM answers, Cites gdpr_full.pdf, No hallucination.
    run_test("What are the administrative fines for GDPR violations?")

    # Test 2: Irrelevant query
    # Expected outcome: Gate blocks OR LLM returns INSUFFICIENT_CONTEXT.
    run_test("How many teeth does a shark have?")


if __name__ == "__main__":
    main()
