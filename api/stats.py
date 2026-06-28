"""
stats.py - PostgreSQL Query Stats and Analytics Logging for DocSentinel API
==========================================================================
"""

import os
from datetime import datetime
import psycopg2


def get_db_connection():
    """
    Establish a connection to the PostgreSQL database.
    """
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        raise ValueError("POSTGRES_URL environment variable is not set.")
    return psycopg2.connect(postgres_url)


def init_stats_table() -> None:
    """
    Initialize the query_stats table if it does not exist.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS query_stats (
                id SERIAL PRIMARY KEY,
                query_text TEXT NOT NULL,
                latency_ms FLOAT NOT NULL,
                cached BOOLEAN NOT NULL,
                status VARCHAR NOT NULL,
                created_at TIMESTAMP NOT NULL
            );
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        print("  Query stats table initialized successfully (or already exists).")
    except Exception as exc:
        print(f"[api/stats] Error initializing stats table: {exc}")


def log_query(query: str, latency_ms: float, cached: bool, status: str) -> None:
    """
    Log query analytics to the query_stats table.

    Args:
        query: User compliance query text.
        latency_ms: Processing latency in milliseconds.
        cached: True if the request hit the semantic cache.
        status: SUCCESS, BLOCKED, INSUFFICIENT_CONTEXT, HALLUCINATION_RISK.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute(
            """
            INSERT INTO query_stats (query_text, latency_ms, cached, status, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (query, latency_ms, cached, status, datetime.now())
        )
        
        conn.commit()
        cur.close()
        conn.close()
    except Exception as exc:
        print(f"[api/stats] Error logging query stats: {exc}")


def get_stats() -> dict:
    """
    Fetch aggregated query statistics.

    Returns:
        dict: containing total_queries, cache_hits, cache_hit_rate, avg_latency_ms.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 1. Total queries
        cur.execute("SELECT COUNT(*) FROM query_stats")
        total_queries = cur.fetchone()[0]
        
        if total_queries == 0:
            cur.close()
            conn.close()
            return {
                "total_queries": 0,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "avg_latency_ms": 0.0
            }
            
        # 2. Cache hits
        cur.execute("SELECT COUNT(*) FROM query_stats WHERE cached = TRUE")
        cache_hits = cur.fetchone()[0]
        
        # 3. Average latency
        cur.execute("SELECT AVG(latency_ms) FROM query_stats")
        avg_latency = float(cur.fetchone()[0] or 0.0)
        
        cur.close()
        conn.close()
        
        cache_hit_rate = float(cache_hits) / float(total_queries)
        
        return {
            "total_queries": total_queries,
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hit_rate, 4),
            "avg_latency_ms": round(avg_latency, 2)
        }
        
    except Exception as exc:
        print(f"[api/stats] Error retrieving stats: {exc}")
        return {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_hit_rate": 0.0,
            "avg_latency_ms": 0.0
        }
