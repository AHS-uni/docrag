"""
Abstract base class for processing datasets and their entries into the
unified format.
"""

import json
import logging
import shutil
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Generic, TypeVar

from docrag.schema.entries import (
    Answer,
    CorpusEntry,
    Document,
    Evidence,
    Question,
    Tag,
    UnifiedEntry,
    RawEntry,
)
from docrag.schema.metadata import DatasetMetadata, SplitMetadata
from docrag.utils import get_logger

from .exceptions import UnificationError

__all__ = ["Unifier"]


RawT = TypeVar("RawT", bound=RawEntry)

class Unifier(ABC, Generic[RawT]):
    """
    Abstract base class for processing datasets and their entries into the
    unified format.

    Attributes:
        name (str): Unique identifier for the dataset.
        dataset_root (Path): Root directory for dataset.
        test_mode (bool): Whether to operate in test mode.
        skip-problematic (bool): Whether to skip problematic entries.
    """

    def __init__(
            self,
            *,
            name: str,
            dataset_root: Path,
            test_mode: bool =False,
            skip_problematic: bool = False
    ) -> None:
        """
        Args:
            name (str): Dataset name for file naming and metadata.
            dataset_root (Path): Root directory containing raw data and where unified output will go.
            test_mode (bool): Whether to operate in test mode.
            skip-problematic (bool): Whether to skip problematic entries.
        """
        self.name: str = name
        self.dataset_root: Path = dataset_root
        self.test_mode: bool = test_mode
        self.skip_problematic: bool = skip_problematic

        # Raw inputs
        self._raw_qa_files: list[Path] = []
        self._raw_document_files: list[Path] = []

        # Document corpus
        self._corpus_records: list[tuple[str, int, Path]] = []
        self._corpus_index: set[tuple[str, int]] = set()
        self._corpus_documents: dict[str, list] = defaultdict(list)

        # Internal metadata state
        self._count_documents: int = 0
        self._count_pages: int = 0
        self._total_questions: int = 0
        self._splits: list[SplitMetadata] = []
        self._tag_summary: dict[str, Counter[str]] = {
            "entry": Counter(),
            "question": Counter(),
            "document": Counter(),
            "evidence": Counter(),
            "answer": Counter(),
        }

        # Test mode
        self._problematic: dict[str, list[dict[str, Any]]] = {
            "questions": [],
            "documents": [],
        }

        # Logger
        self.logger = get_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG,
            log_file_path=Path("logs/unify.log"),
        )

    @property
    def raw_qas_directory(self) -> Path:
        """
        Directory containing raw QA files.
        """
        return self.dataset_root / "raw_qas"

    @property
    def raw_documents_directory(self) -> Path:
        """
        Directory containing raw document files.
        """
        return self.dataset_root / "raw_documents"

    @property
    def documents_directory(self) -> Path:
        """
        Directory containing (processed) document files.
        """
        return self.dataset_root / "documents"

    @property
    def unified_qas_directory(self) -> Path:
        """
        Directory containing unified/processed QA files.
        """
        return self.dataset_root / "unified_qas"

    def unify(self) -> Path:
        """
        Run the full dataset unification pipeline.

        1. Discover raw QA files and (if needed) raw document files.
        2. Convert any raw documents into the `documents/` layout.
        3. Load the processed corpus into memory.
        4. Convert raw QA entries into `UnifiedEntry` objects, grouped by split.
        5. Sanity-check that every evidence page exists in the corpus.
        6. Write out:
           - `corpus.jsonl`
           - per-split QA JSONL files under `qas/`
           - `metadata.json`

        Returns:
            Path: The path to the `qas/` directory containing unified QA files.

        Raises:
            FileNotFoundError: If required input directories are missing.
            ConversionError: If a raw document fails to convert.
            ValueError: If sanity checks fail (missing pages).
        """
        self.logger.info("Starting unification for dataset %s", self.name)
        # 1) Find raw inputs
        self._discover_raw()
        self.logger.info(
            "Discovered %d QA files and %d raw documents",
            len(self._raw_qa_files),
            len(self._raw_document_files),
        )

        # # 2) Build `documents/` if we have raw_documents to process
        # self._convert_raw_documents()
        # self.logger.info("Documents directory ready at %s", self.documents_directory)

        # 3) Load the corpus into memory
        self._load_corpus()
        self.logger.info(
            "Loaded corpus with %d documents and %d pages",
            self._count_documents,
            self._count_pages,
        )

        # 4) Convert QA entries
        split_mapping: dict[str, list[UnifiedEntry]] = self._convert_raw_qas()
        self.logger.info(
            "Converted QA into %d splits totaling %d questions",
            len(split_mapping),
            self._total_questions,
        )

        entries: list[UnifiedEntry] = [
            entry for entries in split_mapping.values() for entry in entries
        ]
        # self._sanity_check(entries)

        # Collect tag statistics
        for entry in entries:
            self._record_tags("entry", entry.tags)
            self._record_tags("question", entry.question.tags)
            self._record_tags("document", entry.document.tags)
            self._record_tags("evidence", entry.evidence.tags)
            self._record_tags("answer", entry.answer.tags)

        # 6) Write outputs
        corpus_path = self._write_corpus()
        self.logger.info("Wrote corpus to %s", corpus_path)
        qas_path = self._write_unified_qas(split_mapping)
        self.logger.info("Wrote unified qas to %s", qas_path)
        metadata_path = self._write_metadata()
        self.logger.info("Wrote dataset metadata to %s", metadata_path)
        if self.test_mode:
            problematic_path = self._write_problematic()
            self.logger.info("Wrote problematic items to %s", problematic_path)
        return self.unified_qas_directory

    def _discover_raw(self) -> None:
        """
        Discover all raw QA files and raw document files (if needed).

        Populates:
            _raw_qas_files (list[Path]):
                A list of paths for each raw QA split file.
            _raw_document_files (list[Path]):
                A list of paths for each raw document file.

        Raises:
            FileNotFoundError: if `raw_qas_directory` does not exist.
        """

        if not (self.raw_qas_directory).exists():
            raise FileNotFoundError(f"Raw QA directory not found: {self.raw_qas_directory}")

        self._raw_qa_files = self._discover_raw_qas()

        if (self.documents_directory).exists():
            self._raw_document_files = []
        else:
            self._raw_document_files = self._discover_raw_documents()

    @abstractmethod
    def _discover_raw_qas(self) -> list[Path]:
        """
        Discover all raw QA files ('.json' or '.jsonl') under the QA directory.

        Returns:
            list[Path]: A sorted list of paths to raw QA files.
        """
        ...

    def _discover_raw_documents(self):
        """
        Discover all raw document files (PDF) under the raw documents directory.

        Returns:
            list[Path]: A sorted list of paths to raw document files.
        """
        return sorted((self.raw_documents_directory).glob("*.pdf"))

    @abstractmethod
    def _load_raw_qas(self, path: Path) -> list[RawT]:
        """
        Load and parse a single raw QA file into typed entries.

        Args:
            path (Path): Path to a raw QA file.

        Returns:
            list[RawT]: A list of parsed raw entries.
        """
        ...

    def _load_corpus(self) -> None:
        """
        Load the processed document corpus into memory.

        Populates:
            _corpus_records (list[tuple[str, int, Path]]):
                A list of (document_id, page_number, image_path) for every page.
            _corpus_index (set[tuple[str, int]]):
                A set of (document_id, page_number) pairs for O(1) lookup.
            _count_documents (int):
                The number of distinct documents discovered.
            _count_pages (int):
                The total number of pages across all documents.

        Raises:
            FileNotFoundError: If `self.documents_directory` does not exist.
        """
        if not self.documents_directory.exists():
            raise FileNotFoundError(
                f"Processed documents directory not found: {self.documents_directory}"
            )

        # reset caches and counters
        self._corpus_records = []
        self._corpus_index = set()
        self._corpus_documents = defaultdict(list)
        self._count_documents = 0
        self._count_pages = 0

        for document_dir in sorted(self.documents_directory.iterdir()):
            if not document_dir.is_dir():
                self.logger.debug(
                    "Skipping non-directory in documents_dir: %s", document_dir
                )
                continue
            document_id = document_dir.name
            self._count_documents += 1

            for img_path in sorted(document_dir.glob("*.jpg")):
                try:
                    page_number = int(img_path.stem)
                except ValueError:
                    # skip files whose stem isn’t an integer
                    self.logger.debug(
                        "Skipping non-integer page file %s in document %s",
                        img_path.name,
                        document_id,
                    )
                    continue
                self._count_pages += 1
                self._corpus_records.append((document_id, page_number, img_path))
                self._corpus_index.add((document_id, page_number))
                self._corpus_documents[document_id].append(page_number)

            self._corpus_documents[document_id].sort()
            count = len(self._corpus_documents[document_id])
            self.logger.debug("Loaded document %s with %d pages", document_id, count)

    def _convert_raw_qas(self) -> dict[str, list[UnifiedEntry]]:
        """
        Convert all raw QA files into unified entries, grouped by split.

        Populates:
            _splits(list[DatasetSplit]): List of the splits in this dataset and their metadata
            _total_questions(int): Total number of questions across all splits.

        Returns:
            dict[str, list[UnifiedEntry]]: Mapping from split name to list of
            converted `UnifiedEntry` objects.
        """
        split_map: dict[str, list[UnifiedEntry]] = {}
        for qa_path in self._raw_qa_files:
            split_name = qa_path.stem
            raw_entries = self._load_raw_qas(qa_path)
            unified_list: list[UnifiedEntry] = []
            self.logger.info(
                "Converting split %s with %d raw entries", split_name, len(raw_entries)
            )
            skipped_entries = 0
            for raw_entry in raw_entries:
                try:
                    entry = self._convert_qa_entry(raw_entry)
                except UnificationError as e:
                    if self.skip_problematic:
                        skipped_entries += 1
                        self.logger.debug("Skipped problematic entry: %s", e)
                        continue
                    else:
                        raise
                if entry is not None:
                    unified_list.append(entry)

            if skipped_entries:
                self.logger.info(
                    "Split %s: skipped %d problematic rows", split_name, skipped_entries
                )

            self.logger.debug(
                "Split %s converted to %d unified entries",
                split_name,
                len(unified_list),
            )

            # update metadata
            count = len(unified_list)
            self._total_questions += count
            self._splits.append(SplitMetadata(name=split_name, count_questions=count))
            split_map[split_name] = unified_list
        return split_map


    def _convert_qa_entry(self, raw: RawT) -> UnifiedEntry | None:
        """
        Map a raw entry into the unified schema.

        Args:
            raw (RawT): A raw dataset entry.

        Returns:
            UnifiedEntry: converted unified entry.
        """

        try:
            document = self._build_document(raw)
        except ValueError:
            raise

        if document is None:
            return None

        question = self._build_question(raw)
        evidence = self._build_evidence(raw)
        answer = self._build_answer(raw)

        entry = self._build_entry(
            raw,
            document,
            question,
            evidence,
            answer,
        )

        return entry

    @abstractmethod
    def _build_entry(
        self,
        raw: RawT,
        document: Document,
        question: Question,
        evidence: Evidence,
        answer: Answer,
    ) -> UnifiedEntry:
        """
        Build the UnifiedEntry model.
        """
        ...

    @abstractmethod
    def _build_document(self, raw: RawT) -> Document | None:
        """
        Build the Document model from a raw entry.
        """
        ...

    @abstractmethod
    def _build_question(self, raw: RawT) -> Question:
        """
        Build the Question model from a raw entry.
        """
        ...

    @abstractmethod
    def _build_evidence(self, raw: RawT) -> Evidence:
        """
        Build the Evidence model from a raw entry.
        """
        ...

    @abstractmethod
    def _build_answer(self, raw: RawT) -> Answer:
        """
        Build the Answer model from a raw entry.
        """
        ...

    def _record_tags(self, level: str, tags: list[Tag]) -> None:
        """
        Increment counters for each Tag at the given object level.
        """
        for tag in tags:
            key = f"{tag.target}.{tag.name.value}"
            self._tag_summary[level][key] += 1


    def _write_corpus(self) -> Path:
        """
        Write the document corpus to a JSONL file.

        Returns:
            Path: The path to the written corpus.jsonl file.

        Raises:
            RuntimeError: If `_corpus_records` is empty (i.e. `_load_corpus`
                was not called before writing).
        """
        if not self._corpus_records:
            raise RuntimeError(
                "No corpus records to write; call `_load_corpus()` first."
            )

        # write to tmp file then swap
        tmp_path = self.dataset_root / "corpus.jsonl.tmp"
        with tmp_path.open("w", encoding="utf-8") as f:
            for document_id, page_number, image_path in self._corpus_records:
                page = CorpusEntry(
                    document_id=document_id, page_number=page_number, image_path=image_path
                )
                f.write(page.model_dump_json())
                f.write("\n")
        output_path = self.dataset_root / "corpus.jsonl"
        tmp_path.replace(output_path)
        return output_path

    def _write_unified_qas(self, split_map: dict[str, list[UnifiedEntry]]) -> Path:
        """
        Write unified QA entries to per-split JSONL files.

        Args:
            split_map (dict[str, list[UnifiedEntry]]): Mapping from split
                name to list of entries.

        Returns:
            Path: The path to the `qas/` directory containing all split files.
        """
        # Write to a temporary directory then swap
        tmp_dir = self.dataset_root / ".tmp_qas"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir()

        for split_name, entries in split_map.items():
            file_path = tmp_dir / f"{split_name}.jsonl"
            with file_path.open("w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(entry.model_dump_json())
                    f.write("\n")

        if (self.unified_qas_directory).exists():
            shutil.rmtree(self.unified_qas_directory)
        tmp_dir.rename(self.unified_qas_directory)

        return self.unified_qas_directory

    def _write_metadata(self) -> Path:
        """
        Write dataset metadata to a JSON file.

        Returns:
            Path: The path to the written metadata.json file.
        """
        meta = DatasetMetadata(
            name=self.name,
            count_documents=self._count_documents,
            count_pages=self._count_pages,
            count_questions=self._total_questions,
            splits=self._splits,
            tag_summary={
                level: dict(counter) for level, counter in self._tag_summary.items()
            },
        )

        meta_path = self.dataset_root / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            f.write(meta.model_dump_json(indent=2))
        return meta_path

    def _write_problematic(self) -> Path:
        """
        Write out the collected `_problematic` dictionary to a JSON file
        under the dataset root, so it can be inspected after unification.

        Returns:
            Path: the path to the written problematic.json file.
        """
        output_path = self.dataset_root / "problematic.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self._problematic, f, indent=2)

        self.logger.info("Wrote problematic entries to %s", output_path)
        return output_path
