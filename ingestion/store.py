"""
DocSentinel — Ingestion Store Module
Handles storing document chunks in Qdrant (vector DB) and PostgreSQL (metadata DB).
"""

from dotenv import load_dotenv

load_dotenv()

import os
import uuid

import qdrant_client
from qdrant_client.models import Distance, PointStruct, VectorParams
import psycopg2


def store_in_qdrant(chunks: list, metadata_list: list, embeddings, session_id: str = "global") -> None:
    """
    Store document chunks with their embeddings and metadata in Qdrant.

    Args:
        chunks: List of chunk dicts (each must contain 'chunk_text').
        metadata_list: List of metadata dicts (one per chunk). Each must contain 'chunk_id'.
        embeddings: List of embedding vectors (one per chunk), each of dimension 384.
        session_id: Session identifier to scope this upload. Use 'global' for shared admin docs.

    Raises:
        Exception: If connection to Qdrant fails or upsert encounters an error.
    """
    try:
        # Connect to Qdrant using environment variables
        host = os.getenv('QDRANT_HOST', 'localhost')
        port = int(os.getenv('QDRANT_PORT', 6333))
        api_key = os.getenv('QDRANT_API_KEY')

        if host.startswith("http://") or host.startswith("https://"):
            client = qdrant_client.QdrantClient(url=host, api_key=api_key)
        else:
            client = qdrant_client.QdrantClient(host=host, port=port, api_key=api_key)

        collection_name = "docsentinel"

        # Create collection if it doesn't already exist
        try:
            client.get_collection(collection_name)
        except Exception:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            print(f"  Created Qdrant collection '{collection_name}'.")

        # Build point structs for batch upsert
        points = []
        for chunk, metadata, embedding in zip(chunks, metadata_list, embeddings):
            chunk_id = metadata["chunk_id"]
            point_id = uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id).hex

            chunk_text = chunk["chunk_text"] if isinstance(chunk, dict) else chunk

            # Payload includes all metadata + chunk text + session_id
            payload = {**metadata, "chunk_text": chunk_text, "session_id": session_id}

            vector = embedding.tolist() if hasattr(embedding, 'tolist') else embedding

            points.append(
                PointStruct(id=point_id, vector=vector, payload=payload)
            )

        # Upsert in batches of 200 to avoid cloud write timeouts
        batch_size = 200
        total_points = len(points)
        print(f"  Uploading {total_points} chunks to Qdrant in batches of {batch_size}...")

        for i in range(0, total_points, batch_size):
            batch = points[i: i + batch_size]
            client.upsert(collection_name=collection_name, points=batch, timeout=60)
            print(f"    Uploaded chunks {i + 1} to {min(i + batch_size, total_points)}...")

        print(f"  Successfully stored all {total_points} chunks in Qdrant.")

    except Exception as e:
        print(f"[ERROR] Failed to store chunks in Qdrant: {e}")
        raise


def store_in_postgres(chunks: list, metadata_list: list, session_id: str = "global") -> None:
    """
    Store chunk metadata and text in PostgreSQL for relational querying.

    This function is **optional** — if PostgreSQL is unavailable the program
    prints a warning and continues without crashing.

    Args:
        chunks: List of chunk dicts (each must contain 'chunk_text').
        metadata_list: List of metadata dicts (one per chunk).
        session_id: Session identifier to scope this upload. Use 'global' for shared admin docs.
    """
    try:
        conn = psycopg2.connect(os.getenv('POSTGRES_URL'))
        cur = conn.cursor()

        # Create table with session_id column
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id      VARCHAR PRIMARY KEY,
                source        VARCHAR,
                page_number   INTEGER,
                doc_type      VARCHAR,
                version       INTEGER,
                date_ingested TIMESTAMP,
                trust_score   FLOAT,
                content_type  VARCHAR,
                chunk_text    TEXT,
                session_id    VARCHAR DEFAULT 'global'
            );
        """)

        # Migrate existing tables that don't have session_id yet
        cur.execute("""
            ALTER TABLE chunks ADD COLUMN IF NOT EXISTS session_id VARCHAR DEFAULT 'global';
        """)

        # Insert each chunk; upsert session_id if chunk already exists
        for chunk, metadata in zip(chunks, metadata_list):
            chunk_text = chunk["chunk_text"] if isinstance(chunk, dict) else chunk
            cur.execute(
                """
                INSERT INTO chunks (
                    chunk_id, source, page_number, doc_type,
                    version, date_ingested, trust_score, content_type, chunk_text, session_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET session_id = EXCLUDED.session_id;
                """,
                (
                    metadata.get("chunk_id"),
                    metadata.get("source"),
                    metadata.get("page_number"),
                    metadata.get("doc_type"),
                    metadata.get("version"),
                    metadata.get("date_ingested"),
                    metadata.get("trust_score"),
                    metadata.get("content_type"),
                    chunk_text,
                    session_id,
                ),
            )

        conn.commit()
        cur.close()
        conn.close()
        print(f"  Successfully stored {len(chunks)} chunks in PostgreSQL.")

    except Exception as e:
        print(f"[WARNING] PostgreSQL storage failed (non-fatal): {e}")
