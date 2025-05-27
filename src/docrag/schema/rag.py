"""
Pydantic models for RAG related types.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Query(BaseModel):  # SKELETON
    id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):  # SKELETON
    doc_id: str
    page_num: int
    score: float
    path: Path
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeneratedAnswer(BaseModel):  # SKELETON
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
