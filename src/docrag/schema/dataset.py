"""
Pydantic model for a single dataset.
"""

from __future__ import annotations

from pydantic import BaseModel

__all__ = [
    "DatasetMetadata",
    "DatasetSplit",
]


class DatasetSplit(BaseModel):
    """
    A single split in a dataset.

    Attributes:
        name: Name of the dataset split.
        num_questions: Number of questions in this split.
    """

    name: str
    num_questions: int


class DatasetMetadata(BaseModel):
    """
    Metadata entry for an entire dataset.

    Attributes:
        name: Name of the dataset.
        num_documents: Number of documents in the dataset corpus.
        num_pages: Total number of pages across all documents.
        num_questions: Total number of questions across all splits.
        splits: List of dataset splits and their metadata.
    """

    name: str
    num_documents: int
    num_pages: int
    num_questions: int
    splits: list[DatasetSplit]
