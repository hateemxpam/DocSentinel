"""
llm.py - Interface for Groq Llama-3.3-70B Chat model for DocSentinel
===================================================================

Handles sending system/user prompts to the Groq API via ChatGroq,
with robust rate-limit retry logic and deterministic (temp=0) settings.
"""

import os
import time
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


def generate(system_prompt: str, query: str) -> str:
    """
    Send system prompt and query to Groq LLM (Llama 3.3 70B) and return response.

    Features exponential backoff retries for rate limits or other transient errors.

    Args:
        system_prompt: Formatting guidelines and policy context.
        query: The user query.

    Returns:
        The raw LLM response text.
    """
    # Fetch API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[llm] ERROR: GROQ_API_KEY not found in environment.")
        return "INSUFFICIENT_CONTEXT"

    # Initialize client (Llama 3.3 70B, temperature 0.2 for highly deterministic factual answer)
    chat = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=api_key,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    max_attempts = 3
    delay = 2.0  # initial delay in seconds

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Calling LLM (attempt {attempt}/{max_attempts})...")
            
            # API call
            response = chat.invoke(messages)
            
            print("LLM response received.")
            return str(response.content).strip()

        except Exception as exc:
            print(f"[llm] Attempt {attempt} failed: {exc}")
            
            if attempt == max_attempts:
                print("[llm] All attempts failed. Returning fallback.")
                return "INSUFFICIENT_CONTEXT"
            
            # Exponential backoff
            sleep_time = delay * (2 ** (attempt - 1))
            print(f"[llm] Retrying in {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)

    return "INSUFFICIENT_CONTEXT"
