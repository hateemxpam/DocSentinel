"""
semantic_cache.py - Semantic Query Caching for DocSentinel
==========================================================

Embeds queries and compares them using cosine similarity against previously
cached queries in PostgreSQL.
"""

import json
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from cache.db import get_db_connection

# Load the embedding model once at module level (same model used in ingestion/semantic_retriever)
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("SemanticCache: Embedding model 'all-MiniLM-L6-v2' loaded successfully.")
except Exception as model_err:
    print(f"ERROR: Failed to load SentenceTransformer model in SemanticCache: {model_err}")
    model = None


def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Calculate the cosine similarity between two vectors manually using numpy.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        float: Cosine similarity score between -1.0 and 1.0.
    """
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
        
    return float(np.dot(v1, v2) / (norm1 * norm2))


def get_cached(query: str, session_id: str = "global", threshold: float = 0.92) -> dict | None:
    """
    Check the cache for a semantically similar query within the same session.

    Args:
        query: The raw query text.
        session_id: Only match cache entries from this session or global.
        threshold: Cosine similarity threshold to define a match.

    Returns:
        dict: The cached response dict if a hit is found, else None.
    """
    if model is None:
        print("[cache] Warning: Embedding model not loaded. Skipping cache check.")
        return None

    try:
        # 1. Embed current query
        query_vector = model.encode(query).tolist()

        # 2. Fetch all entries from query_cache
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Ensure cache table has session_id column
        cur.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                id              SERIAL PRIMARY KEY,
                query_text      TEXT,
                query_embedding FLOAT[],
                response        JSONB,
                created_at      TIMESTAMP,
                hit_count       INTEGER DEFAULT 0,
                session_id      VARCHAR DEFAULT 'global'
            );
        """)
        cur.execute("""
            ALTER TABLE query_cache ADD COLUMN IF NOT EXISTS session_id VARCHAR DEFAULT 'global';
        """)
        conn.commit()

        cur.execute(
            "SELECT id, query_text, query_embedding, response FROM query_cache WHERE session_id = %s OR session_id = 'global'",
            (session_id,)
        )
        rows = cur.fetchall()
        
        best_match_id = None
        best_match_score = -1.0
        best_match_response = None
        
        # 3. Calculate similarity for each cached entry
        for cache_id, cached_text, cached_embedding, response_data in rows:
            if not cached_embedding:
                continue
            
            # psycopg2 automatically converts FLOAT[] to list of floats
            score = cosine_similarity(query_vector, cached_embedding)
            
            if score > best_match_score:
                best_match_score = score
                best_match_id = cache_id
                best_match_response = response_data

        # 4. Check if best match passes the similarity threshold
        if best_match_score >= threshold and best_match_id is not None:
            # Increment hit count in background
            cur.execute(
                "UPDATE query_cache SET hit_count = hit_count + 1 WHERE id = %s",
                (best_match_id,)
            )
            conn.commit()
            cur.close()
            conn.close()
            
            print(f"[cache] Semantic match found (similarity: {best_match_score:.4f}).")
            
            # Handle jsonb automatically returning as dict, fallback to load from string
            if isinstance(best_match_response, str):
                return json.loads(best_match_response)
            return best_match_response
        else:
            if best_match_id is not None:
                print(f"[cache] Best semantic match score: {best_match_score:.4f} (below threshold {threshold}).")

        cur.close()
        conn.close()
        return None

    except Exception as exc:
        print(f"[cache] Error checking query cache: {exc}")
        return None


def store_cache(query: str, response: dict, session_id: str = "global") -> None:
    """
    Store the query and its response in the session-scoped cache.

    Args:
        query: The raw query string.
        response: The final pipeline response dict.
        session_id: Session scope for this cache entry.
    """
    if model is None:
        print("[cache] Warning: Embedding model not loaded. Cannot store cache.")
        return

    try:
        # 1. Embed query
        query_vector = model.encode(query).tolist()

        # 2. Convert response dictionary to JSON string (handling datetime serialization)
        def json_serial(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")

        response_json = json.dumps(response, default=json_serial)

        # 3. Insert into DB
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute(
            """
            INSERT INTO query_cache (query_text, query_embedding, response, created_at, session_id)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (query, query_vector, response_json, datetime.now(), session_id)
        )
        
        conn.commit()
        cur.close()
        conn.close()
        print(f"Cache stored for query: '{query}'")

    except Exception as exc:
        print(f"[cache] Error writing to cache table: {exc}")
