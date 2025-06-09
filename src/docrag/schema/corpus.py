"""
Pydantic model for the retriever’s flat corpus of document pages.
"""

from pathlib import Path

from pydantic import BaseModel

__all__ = [
    "CorpusEntry",
]


class CorpusEntry(BaseModel):
    """
    Entry for a single page in the retriever corpus.

    Attributes:
        document_id (str): Matches Entry.document.id
        page_number (int): 0-based page index
        image_path (Path): File system path to the JPEG image
    """

    document_id: str
    page_number: int
    image_path: Path
