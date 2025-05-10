"""
Package 'models':
    Pydantic schema definitions for:
      - Dataset-level metadata (`Dataset`, `DatasetMetadata`)
      - Enumerations for splits, question/document/answer types, evidence sources
      - Unified QA entry models (`Question`, `Document`, `Entry`, etc.)
"""

from .dataset import Metadata as DatasetMetadata, Dataset
from .enums import (
    DatasetSplit,
    QuestionType,
    DocumentType,
    AnswerFormat,
    EvidenceSource,
)
from .unified import (
    Question,
    Page,
    Document,
    Evidence,
    Answer,
    Metadata as EntryMetadata,
    Entry,
)

__all__ = [
    # dataset wrappers
    "DatasetMetadata",
    "Dataset",
    # enums
    "DatasetSplit",
    "QuestionType",
    "DocumentType",
    "AnswerFormat",
    "EvidenceSource",
    # unified models
    "Question",
    "Page",
    "Document",
    "Evidence",
    "Answer",
    "EntryMetadata",
    "Entry",
]
