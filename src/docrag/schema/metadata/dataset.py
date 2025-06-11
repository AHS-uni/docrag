"""
Pydantic model for a single dataset.
"""

from pydantic import BaseModel, Field

__all__ = [
    "DatasetMetadata",
    "SplitMetadata",
]


class SplitMetadata(BaseModel):
    """
    A single split in a dataset.

    Attributes:
        name (str): Name of the dataset split.
        count_questions (int): Number of questions in this split.
    """

    name: str
    count_questions: int


class DatasetMetadata(BaseModel):
    """
    Metadata entry for an entire dataset.

    Attributes:
        name (str): Name of the dataset.
        count_documents (int): Number of documents in the dataset corpus.
        count_pages (int): Total number of pages across all documents.
        count_questions (int): Total number of questions across all splits.
        splits (list[SplitMetadata]): List of dataset splits and their metadata.
        tag_summary (dict[str, dict[str, int]): Counts of tags emitted during unification
    """

    name: str
    count_documents: int
    count_pages: int
    count_questions: int
    splits: list[SplitMetadata]
    tag_summary: dict[str, dict[str, int]] = Field(default_factory=dict)
