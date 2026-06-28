"""
citations.py - Citation Resolver and Text Formatter for DocSentinel
===================================================================

Extracts raw chunk references (e.g. [CHUNK_N]) from generated answers,
re-maps them to sequential numerical citations (e.g. [1]), and resolves
them to structured metadata dicts for UI/API consumption.
"""

import re
from datetime import datetime


def resolve_citations(answer: str, chunk_mapping: dict) -> dict:
    """
    Extract inline chunk placeholders, resolve them, and re-map to ordered citations.

    Example:
        - "According to [CHUNK_3] and [CHUNK_1]..."
        - Remapped to: "According to [1] and [2]..."
        - Returns a list of structured citations corresponding to [1] and [2].

    Args:
        answer: The raw LLM text with placeholders like [CHUNK_3].
        chunk_mapping: Map of "CHUNK_N" to their original chunk dictionaries.

    Returns:
        dict: {
            "answer_text": str (cleaned answer),
            "citations": list[dict] (metadata references)
        }
    """
    # Find all patterns of CHUNK_N (handles individual and grouped like [CHUNK_5, CHUNK_6])
    placeholders = re.findall(r'CHUNK_\d+', answer)
    
    citations = []
    seen = {}
    citation_index = 1
    
    resolved_answer = answer

    for placeholder in placeholders:
        if placeholder in chunk_mapping:
            if placeholder not in seen:
                chunk = chunk_mapping[placeholder]
                seen[placeholder] = citation_index
                
                citations.append({
                    "reference": f"[{citation_index}]",
                    "chunk_id": chunk.get("chunk_id"),
                    "source": chunk.get("source"),
                    "page_number": chunk.get("page_number"),
                    "confidence_score": chunk.get("confidence_score", 0.0),
                    "date_ingested": chunk.get("date_ingested"),
                    "retrieved_at": datetime.now()
                })
                citation_index += 1
            
            # Replace placeholder with its sequential index (e.g., CHUNK_5 -> 5)
            resolved_answer = re.sub(r'\b' + placeholder + r'\b', str(seen[placeholder]), resolved_answer)
            
    return {
        "answer_text": resolved_answer,
        "citations": citations
    }
