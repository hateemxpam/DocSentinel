# test_trust.py

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.retriever import retrieve
from trust.pipeline import run_trust_layer

# Test 1 — relevant query, should PASS gate
print("=== TEST 1: Relevant Query ===")
chunks = retrieve("What are GDPR data subject rights?")
result = run_trust_layer(chunks)
print(f"Gate passed: {result['passed']}")
print(f"Avg confidence: {result['avg_confidence']:.3f}")
if result['passed']:
    for c in result['chunks'][:3]:
        print(f"Confidence: {c['confidence_score']:.3f} | "
              f"Source: {c['source']} | "
              f"Page: {c['page_number']}")

print()

# Test 2 — irrelevant query, should BLOCK gate
print("=== TEST 2: Irrelevant Query ===")
chunks = retrieve("what is the price of bitcoin")
result = run_trust_layer(chunks)
print(f"Gate passed: {result['passed']}")
print(f"Avg confidence: {result['avg_confidence']:.3f}")
print(f"Reason: {result.get('reason', 'N/A')}")