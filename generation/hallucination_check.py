"""
hallucination_check.py - LLM-based Hallucination Verifier for DocSentinel
========================================================================

Verifies whether a generated answer is fully supported by the context documents,
returning a boolean flag.
"""

import os



def check(answer: str, context_text: str) -> dict:
    """
    Check if the generated answer contains information not supported by the context.

    Args:
        answer: The generated answer text.
        context_text: The concatenated text of all context chunks.

    Returns:
        dict: {"hallucination": bool}
    """
    # Skip check for insufficient context responses
    if answer == "INSUFFICIENT_CONTEXT":
        return {"hallucination": False}

    # Import shared generate wrapper
    from generation.llm import generate

    system_prompt = (
        "You are a strict compliance auditor. Your job is to verify if an answer "
        "is fully supported by the provided context. Answer ONLY with 'YES' or 'NO'."
        "Do not write anything else. If there are any claims in the answer that cannot "
        "be verified from the context, respond NO. Otherwise respond YES."
    )

    checker_prompt = f"""Given this context:
{context_text}

Is this answer fully supported by the context?
Answer only YES or NO, nothing else.

Answer to check:
{answer}
"""

    try:
        # Route specifically to llama-3.1-8b-instant for fast, low-token cost checks
        response_text = generate(
            system_prompt=system_prompt,
            query=checker_prompt,
            primary_model="llama-3.1-8b-instant"
        )
        response_text = str(response_text).strip().upper()
        print(f"Hallucination check response: {response_text}")

        # Evaluate response
        if "NO" in response_text:
            return {"hallucination": True}
        return {"hallucination": False}

    except Exception as exc:
        print(f"[hallucination_check] Check failed: {exc}")
        # Default to false (safe fallback, but log the failure)
        return {"hallucination": False}
