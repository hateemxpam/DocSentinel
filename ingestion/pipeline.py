"""
DocSentinel — Ingestion Pipeline
Orchestrates the full document ingestion workflow: parse → chunk → metadata → embed → store.
"""

from ingestion.parser import parse_pdfs
from ingestion.chunker import chunk_documents
from ingestion.metadata import generate_metadata
from ingestion.embedder import generate_embeddings
from ingestion.store import store_in_qdrant, store_in_postgres


def run() -> None:
    """
    Execute the complete ingestion pipeline.

    Steps:
        1. Parse PDFs from the configured input directory.
        2. Chunk the extracted text into smaller segments.
        3. Generate metadata for each chunk.
        4. Generate vector embeddings for each chunk.
        5. Store chunks + embeddings in Qdrant.
        6. Store chunk metadata in PostgreSQL (optional).
    """
    try:
        # Step 1 — Parse PDFs
        print('\n[1/6] Parsing PDFs...')
        pages = parse_pdfs()
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

        # Step 5 — Store in Qdrant
        print('[5/6] Storing in Qdrant...')
        store_in_qdrant(chunks, metadata_list, embeddings)

        # Step 6 — Store metadata in PostgreSQL
        print('[6/6] Storing metadata in PostgreSQL...')
        store_in_postgres(chunks, metadata_list)

        print(f'\nIngestion complete. {len(chunks)} chunks stored.')

    except Exception as e:
        print(f'[ERROR] Ingestion pipeline failed: {e}')
        raise
