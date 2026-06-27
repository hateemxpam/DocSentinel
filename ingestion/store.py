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


def store_in_qdrant(chunks: list, metadata_list: list, embeddings) -> None:
    """
    Store document chunks with their embeddings and metadata in Qdrant.

    Args:
        chunks: List of chunk dicts (each must contain 'chunk_text').
        metadata_list: List of metadata dicts (one per chunk). Each must contain 'chunk_id'.
        embeddings: List of embedding vectors (one per chunk), each of dimension 384.

    Raises:
        Exception: If connection to Qdrant fails or upsert encounters an error.
    """
    try:
        # Connect to Qdrant using environment variables
        host = os.getenv('QDRANT_HOST', 'localhost')
        port = int(os.getenv('QDRANT_PORT', 6333))
        client = qdrant_client.QdrantClient(host=host, port=port)

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
            # Generate a deterministic string ID from the chunk_id
            chunk_id = metadata["chunk_id"]
            point_id = uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id).hex

            # Extract text from chunk dict
            chunk_text = chunk["chunk_text"] if isinstance(chunk, dict) else chunk

            # Payload includes all metadata fields plus the chunk text
            payload = {**metadata, "chunk_text": chunk_text}

            # Convert numpy array to Python list for Qdrant compatibility
            vector = embedding.tolist() if hasattr(embedding, 'tolist') else embedding

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        # Upsert all points in a single batch
        client.upsert(collection_name=collection_name, points=points)
        print(f"  Successfully stored {len(points)} chunks in Qdrant.")

    except Exception as e:
        print(f"[ERROR] Failed to store chunks in Qdrant: {e}")
        raise


def store_in_postgres(chunks: list, metadata_list: list) -> None:
    """
    Store chunk metadata and text in PostgreSQL for relational querying.

    This function is **optional** — if PostgreSQL is unavailable the program
    prints a warning and continues without crashing.

    Args:
        chunks: List of chunk dicts (each must contain 'chunk_text').
        metadata_list: List of metadata dicts (one per chunk).
    """
    try:
        # Connect to PostgreSQL using the URL from environment
        conn = psycopg2.connect(os.getenv('POSTGRES_URL'))
        cur = conn.cursor()

        # Create the chunks table if it doesn't exist
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
                chunk_text    TEXT
            );
        """)

        # Insert each chunk; skip duplicates via ON CONFLICT
        for chunk, metadata in zip(chunks, metadata_list):
            # Extract text from chunk dict
            chunk_text = chunk["chunk_text"] if isinstance(chunk, dict) else chunk
            cur.execute(
                """
                INSERT INTO chunks (
                    chunk_id, source, page_number, doc_type,
                    version, date_ingested, trust_score, content_type, chunk_text
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO NOTHING;
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
                ),
            )

        conn.commit()
        cur.close()
        conn.close()
        print(f"  Successfully stored {len(chunks)} chunks in PostgreSQL.")

    except Exception as e:
        # PostgreSQL storage is optional — warn but do NOT crash
        print(f"[WARNING] PostgreSQL storage failed (non-fatal): {e}")
