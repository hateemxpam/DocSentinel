"""
db.py - PostgreSQL Database Operations for Semantic Caching in DocSentinel
========================================================================

Handles connecting to PostgreSQL and initializing the cache table.
"""

import os
import psycopg2

def get_db_connection():
    """
    Establish a connection to the PostgreSQL database.

    Returns:
        psycopg2 connection object.
    """
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        raise ValueError("POSTGRES_URL environment variable is not set.")
    return psycopg2.connect(postgres_url)


def init_cache_table() -> None:
    """
    Initialize the query_cache table if it does not already exist.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                id SERIAL PRIMARY KEY,
                query_text TEXT NOT NULL,
                query_embedding FLOAT[] NOT NULL,
                response JSONB NOT NULL,
                created_at TIMESTAMP NOT NULL,
                hit_count INTEGER DEFAULT 0
            );
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        print("  Query cache table initialized successfully (or already exists).")
    except Exception as exc:
        print(f"[cache/db] Error initializing cache table: {exc}")


# Initialize table on import
init_cache_table()
