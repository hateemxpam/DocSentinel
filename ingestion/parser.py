"""
parser.py – PDF text and table extraction for the DocSentinel project.

Uses PyMuPDF (fitz) to read every PDF found in the raw-data directory,
extracting plain text and structured tables on a per-page basis.
Each extracted element is returned as a dictionary carrying its source
filename, page number, content, and content type ("text" or "table").
"""

import os
import glob
import logging

import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Logging setup – gives callers visibility into parsing progress and errors.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def _table_to_text(table) -> str:
    """Convert a fitz table object to a clean pipe-separated text block.

    Each row becomes a line of pipe-separated values.  ``None`` cells are
    replaced with empty strings so the output always has consistent columns.

    Args:
        table: A table object returned by ``page.find_tables()``.

    Returns:
        A string with one row per line and columns separated by " | ".
    """
    rows = table.extract()  # list[list[str | None]]
    lines: list[str] = []
    for row in rows:
        # Replace None with empty string for clean formatting
        cleaned = [cell if cell is not None else "" for cell in row]
        lines.append(" | ".join(cleaned))
    return "\n".join(lines)


def parse_pdfs(data_dir: str | None = None) -> list[dict]:
    """Parse all PDFs in *data_dir*, extracting text and tables per page.

    For every page of every ``*.pdf`` file the function produces:

    * One ``"text"`` record with the full page text.
    * One ``"table"`` record **per table** detected on the page (if any).

    Args:
        data_dir: Path to the folder containing raw PDF files.
                  Defaults to ``<project_root>/data/raw/`` when *None*.

    Returns:
        A list of dicts, each with the keys:
        ``page_number``, ``content``, ``content_type``, ``source_filename``.
    """

    # ------------------------------------------------------------------
    # 1. Resolve the data directory – fall back to project-relative path.
    # ------------------------------------------------------------------
    if data_dir is None:
        # __file__ lives in <project>/ingestion/parser.py
        # Two dirname() calls reach the project root.
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, "data", "raw")

    # ------------------------------------------------------------------
    # 2. Discover all PDF files using glob.
    # ------------------------------------------------------------------
    pdf_pattern = os.path.join(data_dir, "*.pdf")
    pdf_files = sorted(glob.glob(pdf_pattern))

    if not pdf_files:
        logger.warning("No PDF files found in '%s'.", data_dir)
        return []

    logger.info("Found %d PDF(s) in '%s'.", len(pdf_files), data_dir)

    # ------------------------------------------------------------------
    # 3. Iterate over each PDF and extract content page by page.
    # ------------------------------------------------------------------
    results: list[dict] = []

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)

        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            logger.error("Failed to open '%s': %s", filename, exc)
            continue  # Skip unreadable files and move on

        try:
            num_pages = len(doc)  # Save page count before processing
            for page_index in range(num_pages):
                page = doc[page_index]
                page_number = page_index + 1  # 1-based page numbering

                # ----- 3a. Extract plain text from the page -----
                try:
                    text = page.get_text("text")
                except Exception as exc:
                    logger.error(
                        "Text extraction failed on page %d of '%s': %s",
                        page_number, filename, exc,
                    )
                    text = ""

                if text.strip():
                    results.append({
                        "page_number": page_number,
                        "content": text.strip(),
                        "content_type": "text",
                        "source_filename": filename,
                    })

                # ----- 3b. Extract tables from the page -----
                try:
                    tables = page.find_tables()
                    for table in tables:
                        table_text = _table_to_text(table)
                        if table_text.strip():
                            results.append({
                                "page_number": page_number,
                                "content": table_text.strip(),
                                "content_type": "table",
                                "source_filename": filename,
                            })
                except Exception as exc:
                    logger.error(
                        "Table extraction failed on page %d of '%s': %s",
                        page_number, filename, exc,
                    )

        finally:
            # Always close the document to free resources
            doc.close()

        logger.info("Finished parsing '%s' (%d pages).", filename, num_pages)

    logger.info("Total records extracted: %d", len(results))
    return results


# ---------------------------------------------------------------------------
# Quick manual test – run this file directly to inspect output.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parsed = parse_pdfs()
    for record in parsed[:5]:  # Preview first 5 records
        print(f"[{record['content_type'].upper()}] "
              f"{record['source_filename']} p.{record['page_number']}")
        print(record["content"][:200], "\n")
