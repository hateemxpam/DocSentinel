"""
test_trust_pipeline.py - Integration Test for Retrieval and Trust Layers
"""

import json
from retrieval.retriever import retrieve
from trust.pipeline import run_trust_layer

def main():
    query = "What are the penalties under GDPR?"
    print(f"Testing Query: '{query}'")
    
    # 1. Retrieve top candidates
    retrieved_chunks = retrieve(query)
    
    # 2. Run trust layer
    gate_result = run_trust_layer(retrieved_chunks)
    
    # 3. Print gate results
    print("\n--- Gate Result ---")
    print(f"Passed: {gate_result['passed']}")
    print(f"Reason: {gate_result['reason']}")
    print(f"Avg Confidence: {gate_result['avg_confidence']:.4f}")
    
    if gate_result['passed']:
        print(f"\nTop 3 Scored Chunks:")
        for idx, chunk in enumerate(gate_result['chunks'][:3], 1):
            print(f"\n[{idx}] Source: {chunk['source']} (Page {chunk['page_number']})")
            print(f"    Confidence: {chunk['confidence_score']:.4f} | Reranker: {chunk['reranker_score']:.4f} | Freshness: {chunk['freshness_score']:.4f} | Consistency: {chunk['consistency_score']:.4f}")
            print(f"    Text snippet: {chunk['chunk_text'][:150]}...")
    else:
        print("\nPipeline blocked. No chunks sent to generation.")

if __name__ == "__main__":
    main()
