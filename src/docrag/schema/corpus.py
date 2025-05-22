"""
Pydantic model for the retriever’s flat corpus of document pages.
"""

from __future__ import annotations
from pathlib import Path

from pydantic import BaseModel

__all__ = [
    "CorpusPage",
]


class CorpusPage(BaseModel):
    """
    Entry for a single page in the retriever corpus.

    Attributes:
        doc_id: Matches Entry.document.id
        page_number: 0-based page index
        image_path: Filesystem path to the JPEG image
    """

    doc_id: str
    page_number: int
    image_path: Path
