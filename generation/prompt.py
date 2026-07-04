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

    system_prompt = f"""You are DocSentinel, an expert compliance and policy analyst assistant.

Your job is to read the provided document context and write a comprehensive, well-structured answer to the user's question — similar to how a senior compliance consultant would explain it.

STRICT RULES:
1. Answer ONLY using the facts found in the provided CONTEXT below. Do not use external knowledge.
2. If the context does not contain enough information to answer, respond with exactly: INSUFFICIENT_CONTEXT
3. Do NOT say "based on the context" or "according to the chunks" — write naturally and confidently.
4. Do NOT produce one-line bullet lists. Write in full, flowing paragraphs with clear structure.

ANSWER STYLE:
- Start with a clear, direct opening sentence that directly answers the question.
- Then expand with detailed explanation, covering all relevant aspects found in the context.
- Use structured formatting where helpful: bold headings (e.g., **Overview**, **Key Requirements**, **Penalties**) to organize different aspects of a complex answer.
- Use bullet points ONLY for listing enumerated items (e.g., a list of rights, a list of obligations). Always accompany them with an explanatory paragraph.
- End with a concise summary sentence if the answer is long.
- Aim for a thorough answer of 150–400 words, proportional to the complexity of the question.

CONTEXT:
{context_str}
"""

    return system_prompt, chunk_mapping
