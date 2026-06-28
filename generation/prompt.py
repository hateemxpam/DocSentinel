"""
prompt.py - System Prompts and Context Builders for DocSentinel
===============================================================

Formats retrieved document chunks into numbered contexts and builds
the strict compliance-oriented system prompt.
"""

def build_prompt(query: str, chunks: list[dict]) -> tuple[str, dict]:
    """
    Format chunk texts as context and construct a strict system prompt.

    Args:
        query: The user's search query.
        chunks: List of retrieved policy chunk dictionaries.

    Returns:
        tuple containing:
            - system_prompt (str): The structured system prompt with context.
            - chunk_mapping (dict): Maps "CHUNK_N" keys to their original chunk dictionaries.
    """
    formatted_chunks = []
    chunk_mapping = {}

    for idx, chunk in enumerate(chunks, 1):
        chunk_key = f"CHUNK_{idx}"
        chunk_text = chunk.get("chunk_text", "").strip()
        
        # Save reference mapping
        chunk_mapping[chunk_key] = chunk
        
        # Format the block
        formatted_chunks.append(f"{chunk_key}: {chunk_text}")

    context_str = "\n\n".join(formatted_chunks)

    system_prompt = f"""You are DocSentinel, a compliance document assistant.
RULES:
1. Answer ONLY using the provided context below.
2. Do NOT use any external knowledge or assumptions.
3. Cite every claim using [CHUNK_N] inline.
4. If the context does not contain enough information, respond exactly: INSUFFICIENT_CONTEXT
5. Be precise and formal.

CONTEXT:
{context_str}
"""

    return system_prompt, chunk_mapping
