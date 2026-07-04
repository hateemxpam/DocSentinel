"""
DocSentinel — Ingestion Pipeline
Orchestrates the full document ingestion workflow: parse → chunk → metadata → embed → store.
"""

from ingestion.parser import parse_pdfs
from ingestion.chunker import chunk_documents
from ingestion.metadata import generate_metadata
from ingestion.embedder import generate_embeddings
from ingestion.store import store_in_qdrant, store_in_postgres


def run(specific_file: str | None = None, session_id: str = "global") -> int:
    """
    Execute the complete ingestion pipeline.

    Steps:
        1. Parse PDFs (or specific_file).
        2. Chunk the extracted text.
        3. Generate metadata.
        4. Generate vector embeddings.
        5. Store chunks in Qdrant (tagged with session_id).
        6. Store metadata in PostgreSQL (tagged with session_id).

    Args:
        specific_file: Path to a single PDF to ingest. If None, all PDFs in data/raw/ are ingested.
        session_id: Scope identifier for this batch of chunks.
                    Use 'global' for shared admin documents (GDPR, EU AI Act).
                    Use a user UUID for user-uploaded documents.

    Returns:
        int: Number of chunks stored.
    """
    try:
        # Step 1 — Parse PDFs
        print('\n[1/6] Parsing PDFs...')
        pages = parse_pdfs(specific_file=specific_file)
        print(f'  Extracted {len(pages)} pages.')

        # Step 2 — Chunk text
        print('[2/6] Chunking text...')
        chunks = chunk_documents(pages)
        print(f'  Created {len(chunks)} chunks.')

        # Step 3 — Generate metadata
        print('[3/6] Generating metadata...')
        metadata_list = generate_metadata(chunks)

        # Step 4 — Generate embeddings
        print('[4/6] Generating embeddings...')
        embeddings = generate_embeddings(chunks)

        # Step 5 — Store in Qdrant (with session_id in payload)
        print(f'[5/6] Storing in Qdrant (session: {session_id})...')
        store_in_qdrant(chunks, metadata_list, embeddings, session_id=session_id)

        # Step 6 — Store metadata in PostgreSQL (with session_id column)
        print(f'[6/6] Storing metadata in PostgreSQL (session: {session_id})...')
        store_in_postgres(chunks, metadata_list, session_id=session_id)

        print(f'\nIngestion complete. {len(chunks)} chunks stored.')
        return len(chunks)

    except Exception as e:
        print(f'[ERROR] Ingestion pipeline failed: {e}')
        raise
