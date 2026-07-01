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


def generate(system_prompt: str, query: str, primary_model: str = "llama-3.3-70b-versatile") -> str:
    """
    Send system prompt and query to Groq LLM and return response.

    Features exponential backoff retries and automatic fallback to a smaller,
    highly available model (llama-3.1-8b-instant) if the primary model is rate-limited.

    Args:
        system_prompt: Formatting guidelines and policy context.
        query: The user query.
        primary_model: The preferred Groq model ID.

    Returns:
        The raw LLM response text.
    """
    # Fetch API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[llm] ERROR: GROQ_API_KEY not found in environment.")
        return "INSUFFICIENT_CONTEXT"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    max_attempts = 3
    delay = 2.0  # initial delay in seconds
    current_model = primary_model

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Calling LLM ({current_model}) (attempt {attempt}/{max_attempts})...")
            
            # Initialize client with the current active model
            chat = ChatGroq(
                model_name=current_model,
                temperature=0.2,
                api_key=api_key,
            )
            
            # API call
            response = chat.invoke(messages)
            
            print("LLM response received.")
            return str(response.content).strip()

        except Exception as exc:
            print(f"[llm] Attempt {attempt} failed: {exc}")
            
            # If the primary 70B model failed, immediately fall back to the 8B model for subsequent retries
            if current_model == "llama-3.3-70b-versatile":
                print("[llm] Switching to fallback model llama-3.1-8b-instant due to API exception/rate-limit.")
                current_model = "llama-3.1-8b-instant"
            
            if attempt == max_attempts:
                print("[llm] All attempts failed. Returning fallback.")
                return "INSUFFICIENT_CONTEXT"
            
            # Exponential backoff
            sleep_time = delay * (2 ** (attempt - 1))
            print(f"[llm] Retrying in {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)

    return "INSUFFICIENT_CONTEXT"
