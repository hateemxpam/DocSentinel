"""
freshness.py - Document Freshness Scorer for DocSentinel
========================================================

Calculates how "fresh" a document chunk is based on when it was ingested.
Newer documents get higher scores, older ones decay towards zero.

Formula:  freshness = 1 / (1 + days_old)

Examples:
    - Ingested today      → 1 / (1 + 0)   = 1.00
    - Ingested 1 day ago  → 1 / (1 + 1)   = 0.50
    - Ingested 9 days ago → 1 / (1 + 9)   = 0.10
    - Ingested 99 days ago→ 1 / (1 + 99)  = 0.01
"""

from datetime import datetime, date


def calculate_freshness(date_ingested) -> float:
    """Calculate a freshness score for a chunk based on its ingestion date.

    Args:
        date_ingested: When the chunk was stored.  Accepts:
            - datetime object
            - date object
            - ISO-format string ("2025-06-28" or "2025-06-28T12:00:00")
            - None or empty (returns 0.0 as a safe default)

    Returns:
        Float between 0 and 1. Higher = more recent.
    """
    # --- Handle missing or empty values gracefully ---
    if date_ingested is None:
        return 0.0

    # --- Convert string to datetime if needed ---
    if isinstance(date_ingested, str):
        date_ingested = date_ingested.strip()
        if not date_ingested or date_ingested == "1970-01-01":
            return 0.0
        try:
            # Try full datetime first, then date-only
            date_ingested = datetime.fromisoformat(date_ingested)
        except (ValueError, TypeError):
            return 0.0

    # --- Convert date to datetime for consistent math ---
    if isinstance(date_ingested, date) and not isinstance(date_ingested, datetime):
        date_ingested = datetime.combine(date_ingested, datetime.min.time())

    # --- Calculate days old ---
    try:
        days_old = (datetime.now() - date_ingested).days
        # Clamp to zero in case of future dates
        days_old = max(0, days_old)
    except Exception:
        return 0.0

    # --- Apply inverse decay formula ---
    freshness = 1.0 / (1.0 + days_old)

    return freshness
