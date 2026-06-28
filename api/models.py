"""
models.py - Pydantic Request/Response Models for DocSentinel API
===============================================================
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="The compliance query to execute.")
    min_confidence: float = Field(0.45, description="Minimum confidence threshold to pass the quality gate.")


class Citation(BaseModel):
    source: str
    page_number: int
    confidence_score: float
    retrieved_at: datetime


class QueryResponse(BaseModel):
    status: str = Field(..., description="Pipeline execution status: SUCCESS, BLOCKED, INSUFFICIENT_CONTEXT, HALLUCINATION_RISK")
    answer: Optional[str] = Field(None, description="The generated response answer.")
    citations: List[Citation] = Field(default_factory=list, description="List of source document citations.")
    avg_confidence: Optional[float] = Field(None, description="Average confidence of the top retrieved chunks.")
    hallucination_flagged: bool = Field(..., description="Whether hallucination check flagged a potential discrepancy.")
    cached: bool = Field(..., description="True if response was served from the semantic cache.")
    latency_ms: float = Field(..., description="Server-side processing latency in milliseconds.")


class UploadResponse(BaseModel):
    filename: str
    chunks_stored: int
    status: str


class StatsResponse(BaseModel):
    total_queries: int
    cache_hits: int
    cache_hit_rate: float
    avg_latency_ms: float
