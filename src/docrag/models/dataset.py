"""
Unified schema for representing QA datasets.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any


from pydantic import BaseModel, Field

__all__ = [
    "Metadata",
    "Dataset",
]


class Metadata(BaseModel):
    """Metadata for an entire dataset."""

    name: str = Field(..., description="Name of the dataset.")
    unified: bool = Field(
        ..., description="True if the dataset has parsed into the unified format."
    )
    unified_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the dataset was unified.",
    )
    version: Optional[str] = Field(None, description="Schema or dataset version.")
    extras: Dict[str, Any] = Field(
        default_factory=dict, description="Additional dataset-level metadata."
    )


class Dataset(BaseModel):
    """A single QA dataset."""

    metadata: Metadata = Field(..., description="Dataset-level metadata.")
    entries_file_path: Path = Field(
        ..., description="Path to file containing dataset entries (JSON or JSONL)."
    )
