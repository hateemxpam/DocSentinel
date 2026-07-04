"""
semantic_retriever.py - Vector Similarity Search for DocSentinel
================================================================

Embeds queries with the `all-MiniLM-L6-v2` SentenceTransformer model
(384-dim) and searches the Qdrant 'docsentinel' collection for the
most similar document chunks.

Exposes a singleton `semantic_retriever` ready for import by other
modules.
"""

# ---------------------------------------------------------------------------
# Environment & standard library imports
# ---------------------------------------------------------------------------
from dotenv import load_dotenv

load_dotenv()  # Load .env variables (QDRANT_HOST, QDRANT_PORT)

import os
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Module-level embedding model (loaded once, shared across all calls)
# ---------------------------------------------------------------------------
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("SemanticRetriever: Embedding model 'all-MiniLM-L6-v2' loaded successfully.")
except Exception as model_err:
    print(
        f"ERROR: Failed to load SentenceTransformer model 'all-MiniLM-L6-v2': {model_err}. "
        "Semantic search will be unavailable."
    )
    model = None


class SemanticRetriever:
    """Performs vector similarity search against Qdrant."""

    def __init__(self) -> None:
        """
        Connect to the Qdrant instance using environment variables.

        Reads QDRANT_HOST and QDRANT_PORT from the .env file and
        stores the client and collection name for later queries.
        """
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if qdrant_host.startswith("http://") or qdrant_host.startswith("https://"):
            self.client = QdrantClient(url=qdrant_host, api_key=qdrant_api_key)
        else:
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port, api_key=qdrant_api_key)
        self.collection_name: str = "docsentinel"

        print("Semantic retriever connected to Qdrant.")

    # -------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------
    def search(self, query: str, top_k: int = 20, session_id: str = "global") -> list[dict]:
        """
        Embed the query and find the most similar chunks in Qdrant.

        Args:
            query:  Free-text search string.
            top_k:  Maximum number of results to return.
            session_id: Scope filter — returns chunks from this session AND global chunks.

        Returns:
            List of dicts with keys: chunk_id, chunk_text, semantic_score.
            Returns an empty list on any failure.
        """
        try:
            if model is None:
                print("SemanticRetriever.search: Embedding model is not available.")
                return []

            # --- Step 1: Embed the query --------------------------------
            query_vector = model.encode(query).tolist()

            # --- Step 2: Search Qdrant with session scope filter ---
            # Returns chunks belonging to this session OR global (admin) chunks
            session_filter = Filter(
                should=[
                    FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                    FieldCondition(key="session_id", match=MatchValue(value="global")),
                ]
            )

            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
                query_filter=session_filter,
            )

            # --- Step 3: Format results ---------------------------------
            results: list[dict] = []
            for point in response.points:
                payload = point.payload or {}
                results.append(
                    {
                        "chunk_id": payload.get("chunk_id", ""),
                        "chunk_text": payload.get("chunk_text", ""),
                        "semantic_score": point.score,
                    }
                )

            return results

        except Exception as err:
            print(f"SemanticRetriever.search error: {err}")
            return []


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
# Create a single, reusable SemanticRetriever instance.  Other modules can
# simply:  from retrieval.semantic_retriever import semantic_retriever
try:
    semantic_retriever = SemanticRetriever()
except Exception as init_err:
    print(
        f"WARNING: Could not initialise SemanticRetriever singleton ({init_err}). "
        "semantic_retriever is set to None."
    )
    semantic_retriever = None
