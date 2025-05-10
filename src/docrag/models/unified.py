"""
Unified schema for representing QA documents.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from .enums import *

__all__ = [
    "Question",
    "Page",
    "Document",
    "Evidence",
    "Answer",
    "Metadata",
    "Entry",
]


class Question(BaseModel):
    """A question posed in a QA example."""

    id: str = Field(..., description="Unique identifier for the question.")
    text: str = Field(..., description="The natural language text of the question.")
    type: List[QuestionType] = Field(
        default_factory=list,
        description="List of high-level question categories.",
    )


class Page(BaseModel):
    """A single page of a document."""

    id: str = Field(..., description="Unique page identifier.")
    number: int = Field(..., description="1-based page number.")
    path: Union[str, Path] = Field(
        ..., description="Filesystem path to the page image."
    )


class Document(BaseModel):
    """A multi-page document providing context."""

    id: str = Field(..., description="Unique identifier for the document.")
    pages: List[Page] = Field(..., description="Ordered list of Page objects.")
    type: List[DocumentType] = Field(
        default_factory=list,
        description="List of high-level document categories.",
    )
    source_path: Optional[Path] = Field(
        ..., description="Filesystem path to the original document file (PDF)."
    )


class Evidence(BaseModel):
    """Content from the document used to support the answer."""

    pages: List[Page] = Field(
        ..., description="Pages that form the evidence for this example."
    )
    source: Optional[EvidenceSource] = Field(
        None, description="Specific source type within the page(s)."
    )


class Answer(BaseModel):
    """The acceptable answer(s) to a question."""

    variants: List[str] = Field(..., description="List of valid answer variants.")
    answerable: bool = Field(
        ..., description="True if the question is answerable given the document."
    )
    rationale: Optional[str] = Field(
        None, description="Explanation or justification for the answer."
    )
    format: Optional[AnswerFormat] = Field(
        None, description="The expected data format of the answer."
    )


class Metadata(BaseModel):
    """Additional metadata for a single QA example."""

    origin: str = Field(..., description="Source dataset or file identifier.")
    reference_id: str = Field(..., description="Original entry ID in the raw dataset.")
    tags: List[str] = Field(
        default_factory=list, description="Free-form tags for filtering or grouping."
    )


class Entry(BaseModel):
    """Complete representation of a single entry/example in a QA dataset."""

    id: str = Field(..., description="Unique identifier for an entry.")
    question: Question = Field(..., description="Question object for an entry.")
    document: Document = Field(..., description="Document object for an entry.")
    evidence: Optional[Evidence] = Field(
        None, description="Evidence object for an entry."
    )
    answer: Optional[Answer] = Field(None, description="Answer object for an entry.")
    metadata: Metadata = Field(..., description="Metadata object for an entry.")
