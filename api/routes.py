"""
routes.py - FastAPI Routes and Endpoint Implementations for DocSentinel API
==========================================================================
"""

import os
import time
import shutil
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from qdrant_client import QdrantClient

# Import pipeline components
from retrieval.retriever import retrieve
from trust.pipeline import run_trust_layer
from generation.pipeline import generate_response
from cache.semantic_cache import get_cached, store_cache
from api.stats import log_query, get_stats, get_db_connection
from ingestion.pipeline import run as run_ingestion

# Request/Response validation models
from api.models import QueryRequest, QueryResponse, UploadResponse, StatsResponse, Citation

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_pipeline(request: QueryRequest):
    """
    Execute a compliance query through DocSentinel's semantic cache, retrieval, trust, and generation pipelines.
    """
    start_time = time.time()
    query = request.query
    min_confidence = request.min_confidence
    session_id = request.session_id

    try:
        # 1. Check the semantic cache first (scoped to this session)
        cached_response = get_cached(query, session_id=session_id)
        
        if cached_response is not None:
            cached = True
            response_data = cached_response
            status = response_data.get("status", "SUCCESS")
        else:
            cached = False
            # 2. Run the full retrieval + trust + generation pipeline (scoped to this session)
            chunks = retrieve(query, session_id=session_id)
            trust_result = run_trust_layer(chunks, threshold=min_confidence)
            response_data = generate_response(query, trust_result)
            status = response_data.get("status", "SUCCESS")

            # 3. Store valid SUCCESS responses in the semantic cache (scoped to this session)
            if status == "SUCCESS":
                store_cache(query, response_data, session_id=session_id)

        # 4. Measure latency
        latency_ms = (time.time() - start_time) * 1000

        # 5. Log query execution stats to DB
        log_query(query, latency_ms, cached, status)

        # 6. Format response structure
        citations = [
            Citation(
                source=c["source"],
                page_number=c["page_number"],
                confidence_score=c["confidence_score"],
                retrieved_at=c["retrieved_at"]
            )
            for c in response_data.get("citations", [])
        ]

        return QueryResponse(
            status=status,
            answer=response_data.get("answer"),
            citations=citations,
            avg_confidence=response_data.get("avg_confidence"),
            hallucination_flagged=response_data.get("hallucination_flagged", False),
            cached=cached,
            latency_ms=round(latency_ms, 2)
        )

    except Exception as exc:
        print(f"[api/routes] Error processing query: {exc}")
        raise HTTPException(status_code=500, detail=f"Internal pipeline error: {exc}")


@router.post("/upload", response_model=UploadResponse)
async def upload_policy(file: UploadFile = File(...), session_id: str = Form("global")):
    """
    Upload a new policy document (PDF) and parse/ingest it into Qdrant & PostgreSQL
    under the given session_id so it is only visible to this user's session.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents (.pdf) are supported.")

    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    file_path = os.path.join(raw_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"[api/routes] Saved uploaded file to {file_path}. Session: {session_id}. Initiating ingestion...")

        # Run ingestion tagged with the user's session_id
        chunks_count = run_ingestion(specific_file=file_path, session_id=session_id)

        return UploadResponse(filename=file.filename, chunks_stored=chunks_count, status="SUCCESS")

    except Exception as exc:
        print(f"[api/routes] Ingestion failed for uploaded file: {exc}")
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline failed: {exc}")


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    Delete all documents and cached queries belonging to a user's session.
    Global (admin) chunks are never deleted by this endpoint.
    """
    if session_id == "global":
        raise HTTPException(status_code=403, detail="Cannot delete global admin documents.")

    deleted_chunks = 0
    deleted_cache = 0

    try:
        # 1. Delete from PostgreSQL chunks table
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM chunks WHERE session_id = %s", (session_id,))
        deleted_chunks = cur.rowcount
        cur.execute("DELETE FROM query_cache WHERE session_id = %s", (session_id,))
        deleted_cache = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
    except Exception as exc:
        print(f"[api/routes] PostgreSQL session cleanup failed: {exc}")

    try:
        # 2. Delete from Qdrant by session_id payload filter
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_host.startswith("http://") or qdrant_host.startswith("https://"):
            client = QdrantClient(url=qdrant_host, api_key=qdrant_api_key)
        else:
            client = QdrantClient(host=qdrant_host, port=qdrant_port, api_key=qdrant_api_key)

        client.delete(
            collection_name="docsentinel",
            points_selector=Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
            ),
        )
    except Exception as exc:
        print(f"[api/routes] Qdrant session cleanup failed: {exc}")

    return {
        "status": "cleared",
        "session_id": session_id,
        "chunks_deleted": deleted_chunks,
        "cache_entries_deleted": deleted_cache,
    }


@router.get("/health")
async def health_check():
    """
    Check the health of connection endpoints (Qdrant & PostgreSQL).
    """
    qdrant_ok = "ok"
    postgres_ok = "ok"

    # 1. Qdrant Health Check - supports both local and cloud URLs
    try:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_host.startswith("http://") or qdrant_host.startswith("https://"):
            client = QdrantClient(url=qdrant_host, api_key=qdrant_api_key)
        else:
            client = QdrantClient(host=qdrant_host, port=qdrant_port, api_key=qdrant_api_key)
        client.get_collections()
    except Exception as exc:
        print(f"[api/routes] Qdrant health check failed: {exc}")
        qdrant_ok = "failed"

    # 2. Postgres Health Check
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        cur.close()
        conn.close()
    except Exception as exc:
        print(f"[api/routes] Postgres health check failed: {exc}")
        postgres_ok = "failed"

    return {
        "status": "healthy" if qdrant_ok == "ok" and postgres_ok == "ok" else "unhealthy",
        "qdrant": qdrant_ok,
        "postgres": postgres_ok
    }



@router.get("/stats", response_model=StatsResponse)
async def stats_endpoint():
    """
    Retrieve aggregated usage stats, cache hits, hit rates, and average latencies.
    """
    stats_data = get_stats()
    return StatsResponse(
        total_queries=stats_data["total_queries"],
        cache_hits=stats_data["cache_hits"],
        cache_hit_rate=stats_data["cache_hit_rate"],
        avg_latency_ms=stats_data["avg_latency_ms"]
    )


@router.delete("/cache")
async def clear_cache():
    """
    Clear all semantically cached items inside PostgreSQL cache table.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("TRUNCATE TABLE query_cache RESTART IDENTITY;")
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "cache cleared"}
    except Exception as exc:
        print(f"[api/routes] Failed to clear query cache: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {exc}")
