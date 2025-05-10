"""
Abstract base classes for datasets and their entries.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Iterator

from docrag.config import dataset_settings
from docrag.models import Entry as UnifiedEntry

__all__ = []


class BaseEntry(ABC):
    """
    Abstract interface for one raw QA example before unification.
    """

    @abstractmethod
    def to_unified(self) -> UnifiedEntry:
        """
        Map this raw entry into the unified schema.
        """
        ...


class BaseLoader(ABC):
    """
    Base class for any QA dataset loader.

    Subclasses must specify how to discover, load, and parse raw data
    into a list of `BaseEntry` instances.
    """

    def __init__(
        self,
        name: str,
        raw_dir: Path | None = None,
        unified_dir: Path | None = None,
    ):
        """
        Args:
            name: short dataset name (used to name the output JSONL).
            raw_dir: where raw files live (defaults to DATASET_RAW_DIR).
            unified_dir: where to write unified JSONL
                         (defaults to DATASET_UNIFIED_DIR).
        """
        self.name = name
        self.raw_dir = raw_dir or dataset_settings.raw_data_dir
        self.unified_dir = unified_dir or dataset_settings.unified_data_dir

    @abstractmethod
    def discover_raw_files(self) -> List[Path]:
        """
        Return list of raw file paths needed by this dataset
        (e.g. JSONL, CSV, PDF directories, image directories).
        """
        ...

    @abstractmethod
    def load_raw(self, paths: List[Path]) -> Any:
        """
        Read those files into Python objects (e.g. JSON load, CSV read).
        """
        ...

    @abstractmethod
    def parse(self, raw_data: Any) -> List[BaseEntry]:
        """
        Convert loaded raw_data into a list of `BaseEntry` instances.
        """
        ...

    def unify(self) -> Path:
        """
        Orchestrate: discover → load → parse → map to unified → write JSONL.

        Returns:
            Path to the written `<name>.jsonl` file.
        """
        # 1) find files
        files = self.discover_raw_files()

        # 2) load them
        raw = self.load_raw(files)

        # 3) parse into raw entries
        entries = self.parse(raw)

        # 4) ensure output dir
        self.unified_dir.mkdir(parents=True, exist_ok=True)
        out_file = self.unified_dir / f"{self.name}.jsonl"

        # 5) write unified entries
        with out_file.open("w", encoding="utf-8") as f:
            for entry in entries:
                uni = entry.to_unified()
                f.write(uni.model_dump_json())
                f.write("\n")

        return out_file
