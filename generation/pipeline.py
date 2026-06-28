"""
pipeline.py - Generation Pipeline Orchestrator for DocSentinel
==============================================================

Orchestrates the prompt construction, LLM generation, hallucination verification,
and citation mapping.
"""

from generation.prompt import build_prompt
from generation.llm import generate
from generation.hallucination_check import check as check_hallucination
from generation.citations import resolve_citations


def generate_response(query: str, trust_result: dict) -> dict:
    """
    Run the complete generation pipeline based on retrieved and verified chunks.

    Args:
        query: The user's query string.
        trust_result: The dictionary returned by trust.pipeline.run_trust_layer.

    Returns:
        dict: Final structured response containing status, answer, citations, etc.
    """
    print("\n--- Generation & Verification Pipeline ---")

    # Step 1: Check if Trust Layer passed
    if not trust_result.get("passed", False):
        print("[generation] Gate blocked: Insufficient source confidence.")
        return {
            "status": "BLOCKED",
            "reason": trust_result.get("reason", "Insufficient source confidence"),
            "answer": None,
            "citations": [],
            "avg_confidence": trust_result.get("avg_confidence", 0.0),
            "hallucination_flagged": False,
        }

    # Step 2: Build the prompt using chunks
    chunks = trust_result.get("chunks", [])
    print(f"[generation] Building prompt with {len(chunks)} trusted chunks...")
    system_prompt, chunk_mapping = build_prompt(query, chunks)

    # Compile a plain context string for the checker (used for hallucination check)
    context_text = "\n\n".join(
        [f"CHUNK_{idx}: {c.get('chunk_text', '')}" for idx, c in enumerate(chunks, 1)]
    )

    # Step 3: Call LLM
    answer = generate(system_prompt, query)

    # Step 4: Handle INSUFFICIENT_CONTEXT fallback
    if answer == "INSUFFICIENT_CONTEXT":
        print("[generation] LLM reported: INSUFFICIENT_CONTEXT.")
        return {
            "status": "INSUFFICIENT_CONTEXT",
            "reason": "The model determined the context does not contain enough information.",
            "answer": "No sufficient information found in documents.",
            "citations": [],
            "avg_confidence": trust_result.get("avg_confidence", 0.0),
            "hallucination_flagged": False,
        }

    # Step 5: Run hallucination check
    print("[generation] Running hallucination check...")
    check_result = check_hallucination(answer, context_text)
    hallucinated = check_result.get("hallucination", False)

    if hallucinated:
        print("[generation] WARNING: Hallucination detected!")
        status = "HALLUCINATION_RISK"
    else:
        print("[generation] Success: No hallucination detected.")
        status = "SUCCESS"

    # Step 6: Resolve citations
    print("[generation] Resolving chunk citations...")
    resolved = resolve_citations(answer, chunk_mapping)

    final_response = {
        "status": status,
        "answer": resolved["answer_text"],
        "citations": resolved["citations"],
        "avg_confidence": trust_result.get("avg_confidence", 0.0),
        "hallucination_flagged": hallucinated,
    }

    print(f"[generation] Pipeline complete. Status: {status}")
    return final_response
