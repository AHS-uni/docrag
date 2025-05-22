"""
Abstract base classes for processing datasets and their entries.
"""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar
from collections.abc import Iterator

from docrag.schema import (
    BaseRawEntry,
    UnifiedEntry,
    DatasetMetadata,
    DatasetSplit,
    CorpusPage,
)

__all__ = ["BaseUnifier"]

RawT = TypeVar("RawT", bound=BaseRawEntry)
logger = logging.getLogger(__name__)


class BaseUnifier(ABC, Generic[RawT]):
    """
    Convert raw dataset entries into the unified schema and write them to disk.

    Attributes:
        name: Unique identifier for the dataset.
        data_dir: Root directory with `raw/` and `unified/` subdirectories.
        version: Dataset version string.
    """

    def __init__(self, name: str, data_dir: Path) -> None:
        """
        Args:
            name: Dataset name for file naming and metadata.
            data_dir: Base path containing raw data and where unified output will go.
        """
        self.name = name
        self.data_dir = data_dir

        # Internal metadata state
        self.num_documents: int = 0
        self.num_pages: int = 0
        self.total_questions: int = 0
        self.splits: list[DatasetSplit] = []

    @property
    def raw_qas_dir(self) -> Path:
        """Directory containing raw QA files."""
        return self.data_dir / "raw_qas"

    @property
    def documents_dir(self) -> Path:
        """Directory containing document files."""
        return self.data_dir / "documents"

    @property
    def unified_qas_dir(self) -> Path:
        """Directory containing unified QA files."""
        return self.data_dir / "unified_qas"

    @abstractmethod
    def discover_raw(self) -> list[Path]:
        """
        Discover all raw QA file paths (e.g. JSON, CSV) under `raw_dir`.

        Returns:
            A list of paths to raw QA files to process.
        """
        ...

    def discover_documents(self) -> Iterator[tuple[str, int, Path]]:
        """
        Walk `documents_dir` and yield (doc_id, page_number, image_path)
        for every JPEG whose filename stem is an integer.

        Yields:
            A tuple of (doc_id, page_number, image_path).
        """
        for doc_dir in sorted(self.documents_dir.iterdir()):
            if not doc_dir.is_dir():
                continue
            doc_id = doc_dir.name
            self.num_documents += 1
            for img_path in sorted(doc_dir.glob("*.jpg")):
                try:
                    page_num = int(img_path.stem)
                except ValueError:
                    continue
                self.num_pages += 1
                yield doc_id, page_num, img_path

    @abstractmethod
    def load_raw(self, path: Path) -> list[RawT]:
        """
        Load and parse a single raw QA file into typed entries.

        Args:
            path: Path to a raw QA file.

        Returns:
            A list of raw entries parsed from disk.
        """
        ...

    @abstractmethod
    def convert_entry(self, raw: RawT) -> UnifiedEntry:
        """
        Map a raw entry into the unified schema.

        Args:
            raw: A raw dataset entry.

        Returns:
            A UnifiedEntry instance.
        """
        ...

    def build_corpus(self) -> Path:
        """
        Build the document corpus from the documents directory and write it
        to 'corpus.jsonl'. Updates internal document and page counts.

        Returns:
            Path to the written corpus.jsonl file.
        """
        output_path = self.data_dir / "corpus.jsonl"
        with output_path.open("w", encoding="utf-8") as f:
            for doc_id, page_num, img_path in self.discover_documents():
                page = CorpusPage(
                    doc_id=doc_id, page_number=page_num, image_path=img_path
                )
                f.write(page.model_dump_json())
                f.write("\n")
        return output_path

    def unify(self) -> Path:
        """
        Run the full unification pipeline:

        1. Discover raw files.
        2. Load raw entries.
        3. Convert to unified entries.
        4. Write JSONL per split under `unified_qas/`.
        5. Update internal metadata state (splits and question count).

        Returns:
            Path to the `unified_qas/` directory.
        """
        unified_qas_dir = self.unified_qas_dir
        unified_qas_dir.mkdir(parents=True, exist_ok=True)

        raw_split_files = self.discover_raw()
        logger.info(
            "Discovered %d splits for dataset %s", len(raw_split_files), self.name
        )

        for split_file in raw_split_files:
            logger.info("Processing split file %s", split_file)
            split_name = split_file.stem
            raw_entries = self.load_raw(split_file)
            unified_entries = [self.convert_entry(r) for r in raw_entries]

            num_entries = len(unified_entries)
            self.total_questions += num_entries

            split = DatasetSplit(name=split_name, num_questions=num_entries)
            self.splits.append(split)

            output_file = unified_qas_dir / f"{split.name}.jsonl"
            with output_file.open("w", encoding="utf-8") as f:
                for entry in unified_entries:
                    f.write(entry.model_dump_json())
                    f.write("\n")

            logger.info("Wrote %d unified entries to %s", num_entries, output_file)

        return unified_qas_dir

    def write_dataset_metadata(self) -> Path:
        """
        Write dataset-level metadata to 'metadata.json' using accumulated state.

        Returns:
            Path to the written metadata.json file.
        """
        meta = DatasetMetadata(
            name=self.name,
            num_documents=self.num_documents,
            num_pages=self.num_pages,
            num_questions=self.total_questions,
            splits=self.splits,
        )
        meta_path = self.data_dir / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            f.write(meta.model_dump_json(indent=2))
        logger.info("Wrote metadata to %s", meta_path)
        return meta_path
