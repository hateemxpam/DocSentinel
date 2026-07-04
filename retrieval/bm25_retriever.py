"""
bm25_retriever.py - BM25 Lexical Search for DocSentinel
========================================================

Builds a BM25Okapi keyword index over all document chunks at startup.
Data source priority:
    1. PostgreSQL  (chunks table)
    2. Qdrant      (docsentinel collection – fallback if PG is unavailable)

Exposes a singleton `bm25_retriever` ready for import by other modules.
"""

# ---------------------------------------------------------------------------
# Environment & standard library imports
# ---------------------------------------------------------------------------
from dotenv import load_dotenv

load_dotenv()  # Load .env variables (QDRANT_HOST, QDRANT_PORT, POSTGRES_URL)

import os
import numpy as np
import psycopg2
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient


class BM25Retriever:
    """Builds and queries a BM25Okapi index over document chunks."""

    def __init__(self, session_id: str = "global") -> None:
        """
        Load chunk texts for the given session from PostgreSQL or Qdrant.

        Loads chunks where session_id matches the user's session OR 'global',
        so users always have access to the shared admin documents.
        """
        self.session_id = session_id
        self.chunk_ids: list[str] = []
        self.chunk_texts: list[str] = []
        self.bm25: BM25Okapi | None = None

        # --- Step 1: Try loading chunks from PostgreSQL ----------------
        try:
            postgres_url = os.getenv("POSTGRES_URL")
            if not postgres_url:
                raise ValueError("POSTGRES_URL not set in environment.")

            conn = psycopg2.connect(postgres_url)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT chunk_id, chunk_text FROM chunks WHERE session_id = %s OR session_id = 'global'",
                (session_id,)
            )
            rows = cursor.fetchall()
            cursor.close()
            conn.close()

            # Unpack rows into parallel lists
            for chunk_id, chunk_text in rows:
                self.chunk_ids.append(str(chunk_id))
                self.chunk_texts.append(str(chunk_text))

            print(f"BM25Retriever: Loaded {len(self.chunk_ids)} chunks from PostgreSQL.")

        except Exception as pg_err:
            # --- Step 2: Fall back to Qdrant ---------------------------
            print(f"BM25Retriever: PostgreSQL unavailable ({pg_err}). Falling back to Qdrant.")
            try:
                qdrant_host = os.getenv("QDRANT_HOST", "localhost")
                qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
                qdrant_api_key = os.getenv("QDRANT_API_KEY")
                
                if qdrant_host.startswith("http://") or qdrant_host.startswith("https://"):
                    client = QdrantClient(url=qdrant_host, api_key=qdrant_api_key)
                else:
                    client = QdrantClient(host=qdrant_host, port=qdrant_port, api_key=qdrant_api_key)

                # Scroll through the entire 'docsentinel' collection
                records, _next_offset = client.scroll(
                    collection_name="docsentinel",
                    limit=10000,
                    with_payload=True,
                )

                for point in records:
                    payload = point.payload or {}
                    self.chunk_ids.append(str(payload.get("chunk_id", "")))
                    self.chunk_texts.append(str(payload.get("chunk_text", "")))

                print(f"BM25Retriever: Loaded {len(self.chunk_ids)} chunks from Qdrant.")

            except Exception as qd_err:
                print(f"BM25Retriever: Qdrant also unavailable ({qd_err}). No chunks loaded.")

        # --- Step 3: Build the BM25 index ------------------------------
        if not self.chunk_texts:
            # Gracefully handle an empty corpus
            print("BM25Retriever: Corpus is empty – BM25 index will NOT be built.")
            self.bm25 = None
            return

        # Simple whitespace tokenisation (lower-cased)
        tokenized_corpus = [text.lower().split() for text in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 index built with {len(self.chunk_ids)} chunks.")

    # -------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------
    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Search the BM25 index for the most relevant chunks.

        Args:
            query:  Free-text search string.
            top_k:  Maximum number of results to return.

        Returns:
            List of dicts with keys: chunk_id, chunk_text, bm25_score.
            Only results with a score > 0 are included.
        """
        try:
            if self.bm25 is None:
                return []

            # Tokenise the query the same way the corpus was tokenised
            tokenized_query = query.lower().split()

            # Score every document against the query
            scores = self.bm25.get_scores(tokenized_query)

            # Sort indices by descending score and pick the top_k
            top_indices = np.argsort(scores)[::-1][:top_k]

            results: list[dict] = []
            for idx in top_indices:
                score = float(scores[idx])
                if score > 0:
                    results.append(
                        {
                            "chunk_id": self.chunk_ids[idx],
                            "chunk_text": self.chunk_texts[idx],
                            "bm25_score": score,
                        }
                    )

            return results

        except Exception as err:
            print(f"BM25Retriever.search error: {err}")
            return []



# ---------------------------------------------------------------------------
# Factory function — creates a per-request BM25Retriever scoped to a session
# ---------------------------------------------------------------------------
def build_bm25_retriever(session_id: str = "global") -> BM25Retriever:
    """
    Build a BM25Retriever instance scoped to the given session_id.
    Called per-request so each user sees only their own + global chunks.
    """
    try:
        return BM25Retriever(session_id=session_id)
    except Exception as init_err:
        print(f"WARNING: Could not build BM25Retriever for session '{session_id}': {init_err}")
        return None


# Backward-compat singleton for health checks and startup (loads global chunks only)
try:
    bm25_retriever = BM25Retriever(session_id="global")
except Exception as init_err:
    print(f"WARNING: Could not initialise BM25Retriever singleton ({init_err}). "
          "bm25_retriever is set to None.")
    bm25_retriever = None

