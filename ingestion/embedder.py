"""
embedder.py - Embedding Generator for DocSentinel

Uses the sentence-transformers library to convert chunk text into
dense 384-dimensional vector embeddings.  These embeddings power
semantic search and similarity-based retrieval in the RAG pipeline.

Model: all-MiniLM-L6-v2  (free, runs locally, 384-dim output)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
# The model is loaded once at module level so it stays in memory across
# repeated calls to generate_embeddings().  This avoids the overhead of
# re-downloading / re-loading weights on every invocation.
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer

    # all-MiniLM-L6-v2 is a lightweight, high-quality model that produces
    # 384-dimensional embeddings.  It will be downloaded automatically on
    # first use and cached locally for subsequent runs.
    model = SentenceTransformer("all-MiniLM-L6-v2")
except ImportError as exc:
    raise ImportError(
        "The 'sentence-transformers' package is required but not installed. "
        "Install it with:  pip install sentence-transformers"
    ) from exc
except Exception as exc:
    raise RuntimeError(
        f"Failed to load the SentenceTransformer model 'all-MiniLM-L6-v2': {exc}"
    ) from exc


def generate_embeddings(chunks):
    """
    Generate dense vector embeddings for a list of document chunks.

    Args:
        chunks (list[dict]): List of chunk dicts. Each dict must contain
            at least a "chunk_text" key whose value is the string to embed.

    Returns:
        numpy.ndarray: A 2-D array of shape (num_chunks, 384) where each
            row is the embedding vector for the corresponding chunk.

    Raises:
        KeyError: If any chunk dict is missing the "chunk_text" key.
        RuntimeError: If embedding generation fails for any reason.
    """
    try:
        # --- Extract raw text from each chunk dict ---
        # We pull just the text strings into a flat list so sentence-transformers
        # can encode them in a single vectorised call.
        texts = [chunk["chunk_text"] for chunk in chunks]

        # --- Encode texts into embeddings ---
        # batch_size=32 keeps peak memory usage reasonable for large document
        # sets while still leveraging efficient batched computation.
        # show_progress_bar=True gives visibility during long-running jobs.
        # model.encode() returns a numpy ndarray by default.
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
        )

        # Ensure the return value is a proper numpy array (defensive cast
        # in case a future model version changes the default return type).
        embeddings = np.asarray(embeddings)

        return embeddings

    except KeyError as exc:
        raise KeyError(
            f"Chunk dict is missing the required key: {exc}. "
            "Each chunk must have a 'chunk_text' key."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to generate embeddings: {exc}"
        ) from exc
