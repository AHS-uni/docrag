"""
Pydantic models for RAG related types.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Query(BaseModel):  # SKELETON
    id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):  # SKELETON
    document_id: str
    page_number: int
    score: float
    path: Path
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeneratedAnswer(BaseModel):  # SKELETON
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
