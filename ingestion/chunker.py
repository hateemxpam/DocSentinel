"""
chunker.py – Document chunking for the DocSentinel project.

Takes the list of parsed-page dictionaries produced by ``parser.py`` and
splits each page's content into smaller, overlapping text chunks using
LangChain's RecursiveCharacterTextSplitter.  Every chunk retains the
source metadata (filename, page number, content type) from its parent page.
"""

import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Logging setup – mirrors parser.py for consistent project-wide logging.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Splitter configuration constants – easy to tweak in one place.
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500       # Maximum characters per chunk
CHUNK_OVERLAP = 50     # Overlap between consecutive chunks for context continuity


def chunk_documents(parsed_pages: list[dict]) -> list[dict]:
    """Split parsed pages into smaller text chunks with preserved metadata.

    Each input dictionary (from ``parser.parse_pdfs``) is expected to have:
    ``content``, ``source_filename``, ``page_number``, ``content_type``.

    The function produces a flat list of chunk dictionaries, each carrying:

    * ``chunk_text``       – the chunk string.
    * ``source_filename``  – inherited from the parent page.
    * ``page_number``      – inherited from the parent page.
    * ``content_type``     – inherited from the parent page.
    * ``chunk_index``      – sequential index across *all* output chunks.

    Args:
        parsed_pages: List of dicts as returned by ``parser.parse_pdfs()``.

    Returns:
        A list of chunk dicts.  Returns an empty list when the input is
        ``None``, empty, or contains no usable content.
    """

    # ------------------------------------------------------------------
    # 1. Handle empty / None input gracefully.
    # ------------------------------------------------------------------
    if not parsed_pages:
        logger.warning("No parsed pages provided – returning empty chunk list.")
        return []

    # ------------------------------------------------------------------
    # 2. Initialise the LangChain text splitter.
    # ------------------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,           # Count characters (not tokens)
        is_separator_regex=False,      # Use plain-string separators
    )

    # ------------------------------------------------------------------
    # 3. Iterate over each parsed page and split its content.
    # ------------------------------------------------------------------
    chunks: list[dict] = []
    chunk_index = 0  # Global counter across all pages

    for page in parsed_pages:
        content = page.get("content", "")

        # Skip pages with no meaningful text
        if not content or not content.strip():
            logger.debug(
                "Skipping empty page %d from '%s'.",
                page.get("page_number", -1),
                page.get("source_filename", "unknown"),
            )
            continue

        # Split the page content into chunks
        try:
            page_chunks = splitter.split_text(content)
        except Exception as exc:
            logger.error(
                "Splitting failed for page %d of '%s': %s",
                page.get("page_number", -1),
                page.get("source_filename", "unknown"),
                exc,
            )
            continue

        # Build a dict for every chunk, carrying forward page metadata
        for text_piece in page_chunks:
            chunks.append({
                "chunk_text": text_piece,
                "source_filename": page.get("source_filename", "unknown"),
                "page_number": page.get("page_number", -1),
                "content_type": page.get("content_type", "text"),
                "chunk_index": chunk_index,
            })
            chunk_index += 1

    logger.info(
        "Chunking complete: %d page(s) → %d chunk(s).",
        len(parsed_pages),
        len(chunks),
    )
    return chunks


# ---------------------------------------------------------------------------
# Quick manual test – parse then chunk when run directly.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Import the parser from the same package
    from ingestion.parser import parse_pdfs

    parsed = parse_pdfs()
    chunks = chunk_documents(parsed)

    for chunk in chunks[:5]:  # Preview first 5 chunks
        print(f"[Chunk {chunk['chunk_index']}] "
              f"{chunk['source_filename']} p.{chunk['page_number']} "
              f"({chunk['content_type']})")
        print(chunk["chunk_text"][:150], "\n")
