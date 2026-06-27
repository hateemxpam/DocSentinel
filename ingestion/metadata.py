"""
metadata.py - Metadata Generator for DocSentinel

Generates structured metadata dictionaries for each document chunk.
Each metadata dict captures provenance (source, page), identity (chunk_id),
trust scoring, and ingestion timestamps for downstream retrieval and filtering.
"""

import os
from datetime import datetime, timezone


def generate_metadata(chunks):
    """
    Generate a metadata dictionary for every chunk in the input list.

    Args:
        chunks (list[dict]): List of chunk dicts. Each dict must contain:
            - chunk_text (str): The textual content of the chunk.
            - source_filename (str): Original document filename (e.g., "gdpr_full.pdf").
            - page_number (int): 1-based page number the chunk was extracted from.
            - content_type (str): Either "text" or "table".
            - chunk_index (int): 0-based positional index of this chunk.

    Returns:
        list[dict]: A list of metadata dicts (same order as input chunks).
                    Each dict contains the keys: chunk_id, source, page_number,
                    doc_type, version, date_ingested, trust_score, content_type.

    Raises:
        KeyError: If a required key is missing from any chunk dict.
        TypeError: If `chunks` is not a list.
    """
    try:
        if not isinstance(chunks, list):
            raise TypeError(f"Expected a list of chunk dicts, got {type(chunks).__name__}.")

        metadata_list = []

        # Capture the current UTC timestamp once so every chunk in the same
        # batch shares an identical ingestion time.
        date_ingested = datetime.now(timezone.utc).isoformat()

        for chunk in chunks:
            # --- Build a unique chunk ID ---
            # Format: "{filename_without_extension}_{page}_{index}"
            # e.g., "gdpr_full_3_12" for page 3, chunk index 12 of gdpr_full.pdf
            filename_no_ext = os.path.splitext(chunk["source_filename"])[0]
            chunk_id = f"{filename_no_ext}_{chunk['page_number']}_{chunk['chunk_index']}"

            # --- Assemble the metadata dict ---
            metadata = {
                # Unique identifier for this chunk across the entire corpus
                "chunk_id": chunk_id,

                # Original filename so we can trace the chunk back to its document
                "source": chunk["source_filename"],

                # Page number for citation and navigation purposes
                "page_number": chunk["page_number"],

                # Document type — all ingested docs are official regulation texts
                "doc_type": "regulation",

                # Schema / content version; increment when re-processing logic changes
                "version": 1,

                # ISO-8601 UTC timestamp recording when this chunk was ingested
                "date_ingested": date_ingested,

                # Trust score reflects source reliability.
                # 0.95 is assigned to all official regulation documents.
                "trust_score": 0.95,

                # Content type preserves whether the chunk is plain text or a table
                "content_type": chunk["content_type"],
            }

            metadata_list.append(metadata)

        return metadata_list

    except KeyError as exc:
        raise KeyError(
            f"Chunk dict is missing a required key: {exc}. "
            "Expected keys: chunk_text, source_filename, page_number, "
            "content_type, chunk_index."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to generate metadata: {exc}") from exc
