"""
retriever.py – Main retrieval orchestrator for DocSentinel.

Ties together hybrid search (BM25 + semantic), cross-encoder reranking,
and metadata enrichment from PostgreSQL (primary) or Qdrant (fallback).

Pipeline:
    1. Hybrid search  →  ~30 merged candidates
    2. Rerank          →  top-k refined results
    3. Metadata fetch  →  enrich each chunk with source info
    4. Return final enriched list
"""

# ---------------------------------------------------------------------------
# Environment & imports
# ---------------------------------------------------------------------------
from dotenv import load_dotenv

load_dotenv()  # Load .env before anything else reads env vars

import os
import json

import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from retrieval.hybrid import search as hybrid_search
from retrieval.reranker import rerank

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QDRANT_COLLECTION = "docsentinel"

# Metadata keys we expect in both Postgres and Qdrant payloads
_META_KEYS = [
    "source",
    "page_number",
    "doc_type",
    "version",
    "date_ingested",
    "trust_score",
    "content_type",
]

# Sensible defaults when metadata is missing for a chunk
_META_DEFAULTS = {
    "source": "unknown",
    "page_number": 0,
    "doc_type": "unknown",
    "version": "0.0",
    "date_ingested": "1970-01-01",
    "trust_score": 0.0,
    "content_type": "text",
}


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------
def _fetch_metadata_postgres(chunk_ids: list[str]) -> dict:
    """Fetch chunk metadata from the PostgreSQL *chunks* table.

    Args:
        chunk_ids: List of chunk identifiers to look up.

    Returns:
        Dict mapping each found chunk_id to its metadata fields:
        {chunk_id: {source, page_number, doc_type, version,
                    date_ingested, trust_score, content_type}}
        Returns an empty dict on any failure so the caller can
        fall back gracefully.
    """
    try:
        # Connect using the DATABASE_URL stored in .env
        postgres_url = os.getenv("POSTGRES_URL")
        if not postgres_url:
            print("[retriever] POSTGRES_URL not set – skipping Postgres lookup.")
            return {}

        conn = psycopg2.connect(postgres_url)
        cur = conn.cursor()

        # Build a parameterised IN clause:  WHERE chunk_id IN (%s, %s, …)
        placeholders = ", ".join(["%s"] * len(chunk_ids))
        query = (
            f"SELECT chunk_id, source, page_number, doc_type, version, "
            f"date_ingested, trust_score, content_type "
            f"FROM chunks WHERE chunk_id IN ({placeholders})"
        )
        cur.execute(query, chunk_ids)

        # Map each row to a dict keyed by chunk_id
        metadata: dict = {}
        for row in cur.fetchall():
            cid = row[0]
            metadata[cid] = {
                "source": row[1],
                "page_number": row[2],
                "doc_type": row[3],
                "version": row[4],
                "date_ingested": str(row[5]) if row[5] else _META_DEFAULTS["date_ingested"],
                "trust_score": float(row[6]) if row[6] is not None else _META_DEFAULTS["trust_score"],
                "content_type": row[7],
            }

        cur.close()
        conn.close()
        print(f"[retriever] Fetched metadata for {len(metadata)} chunks from Postgres.")
        return metadata

    except Exception as exc:
        # Don't crash – let the caller fall back to Qdrant
        print(f"[retriever] Postgres metadata fetch failed: {exc}")
        return {}


def _fetch_metadata_qdrant(chunk_ids: list[str]) -> dict:
    """Fallback: pull metadata from the Qdrant payload store.

    Since hybrid search candidates already originate from Qdrant,
    every chunk should have its full payload available.  We scroll
    through the collection and filter locally by chunk_id.

    Args:
        chunk_ids: List of chunk identifiers to look up.

    Returns:
        Dict mapping chunk_id -> metadata fields (same shape as
        _fetch_metadata_postgres).  Empty dict on failure.
    """
    try:
        # Qdrant connection settings from environment (defaults to localhost)
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_host.startswith("http://") or qdrant_host.startswith("https://"):
            client = QdrantClient(url=qdrant_host, api_key=qdrant_api_key)
        else:
            client = QdrantClient(host=qdrant_host, port=qdrant_port, api_key=qdrant_api_key)

        # Scroll all points with their payloads
        # NOTE: For very large collections a filtered scroll per chunk_id
        # would be more efficient, but this approach keeps the logic simple
        # and works well for moderate-sized collections.
        records, _next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=None,
            limit=10000,
            with_payload=True,
        )

        # Build a lookup set for fast membership checks
        target_ids = set(chunk_ids)
        metadata: dict = {}

        for point in records:
            payload = point.payload or {}
            cid = payload.get("chunk_id")
            if cid and cid in target_ids:
                metadata[cid] = {
                    key: payload.get(key, _META_DEFAULTS[key])
                    for key in _META_KEYS
                }

        print(f"[retriever] Fetched metadata for {len(metadata)} chunks from Qdrant.")
        return metadata

    except Exception as exc:
        print(f"[retriever] Qdrant metadata fetch failed: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------
def retrieve(query: str) -> list[dict]:
    """Run the full retrieval pipeline and return enriched chunks.

    Steps:
        1. Hybrid search  – BM25 ∪ semantic → ~30 merged candidates
        2. Rerank         – cross-encoder refinement → top 10
        3. Metadata fetch – Postgres first, Qdrant fallback
        4. Enrich         – merge metadata into each result dict

    Args:
        query: Natural-language user query.

    Returns:
        List of dicts, each containing:
            chunk_id, chunk_text, reranker_score, rrf_score,
            in_bm25, in_semantic, source, page_number, doc_type,
            version, date_ingested, trust_score, content_type
    """
    try:
        print("\n--- Retrieval Pipeline ---")

        # ------------------------------------------------------------------
        # Step 1: Hybrid search (BM25 + semantic, merged via RRF)
        # top_k=20 per sub-search → ~30 merged candidates after RRF union
        # ------------------------------------------------------------------
        hybrid_candidates = hybrid_search(query, top_k=20)
        print(f"[Step 1] Hybrid search returned {len(hybrid_candidates)} candidates.")

        # ------------------------------------------------------------------
        # Step 2: Rerank candidates with a cross-encoder model
        # Keep only the top 10 most relevant chunks
        # ------------------------------------------------------------------
        reranked = rerank(query, hybrid_candidates, top_k=10)
        print(f"[Step 2] Reranker narrowed to {len(reranked)} results.")

        # ------------------------------------------------------------------
        # Step 3: Fetch full metadata for all chunk_ids
        # Try PostgreSQL first; if it returns nothing, fall back to Qdrant
        # ------------------------------------------------------------------
        chunk_ids = [r["chunk_id"] for r in reranked if "chunk_id" in r]

        metadata = _fetch_metadata_postgres(chunk_ids)
        if not metadata:
            print("[Step 3] Postgres unavailable – falling back to Qdrant metadata.")
            metadata = _fetch_metadata_qdrant(chunk_ids)
        else:
            print("[Step 3] Metadata loaded from Postgres.")

        # ------------------------------------------------------------------
        # Step 4: Enrich each reranked result with its metadata
        # Use sensible defaults when a chunk has no metadata entry
        # ------------------------------------------------------------------
        results: list[dict] = []
        for item in reranked:
            cid = item.get("chunk_id", "")
            meta = metadata.get(cid, _META_DEFAULTS)

            enriched = {
                # Fields already present from hybrid + reranker stages
                "chunk_id": cid,
                "chunk_text": item.get("chunk_text", ""),
                "reranker_score": item.get("reranker_score", 0.0),
                "rrf_score": item.get("rrf_score", 0.0),
                "in_bm25": item.get("in_bm25", False),
                "in_semantic": item.get("in_semantic", False),
                # Metadata fields (enriched from Postgres / Qdrant)
                "source": meta.get("source", _META_DEFAULTS["source"]),
                "page_number": meta.get("page_number", _META_DEFAULTS["page_number"]),
                "doc_type": meta.get("doc_type", _META_DEFAULTS["doc_type"]),
                "version": meta.get("version", _META_DEFAULTS["version"]),
                "date_ingested": meta.get("date_ingested", _META_DEFAULTS["date_ingested"]),
                "trust_score": meta.get("trust_score", _META_DEFAULTS["trust_score"]),
                "content_type": meta.get("content_type", _META_DEFAULTS["content_type"]),
            }
            results.append(enriched)

        print(f"--- Retrieved {len(results)} chunks ---")
        return results

    except Exception as exc:
        print(f"[retriever] Pipeline error: {exc}")
        return []


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_query = "What are the penalties under GDPR?"
    print(f"Running test query: '{test_query}'\n")

    retrieved = retrieve(test_query)

    print(f"\n{'='*60}")
    print(f"Total results: {len(retrieved)}")
    print(f"{'='*60}")

    for idx, chunk in enumerate(retrieved, start=1):
        print(f"\n--- Result {idx} ---")
        print(json.dumps(chunk, indent=2, default=str))
