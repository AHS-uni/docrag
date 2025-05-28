"""
Abstract base class for processing datasets and their entries into the
unified format.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar
import shutil
import tempfile
import logging

from docrag.schema import (
    BaseRawEntry,
    UnifiedEntry,
    DatasetMetadata,
    DatasetSplit,
    CorpusPage,
)
from docrag.utils import get_logger

__all__ = ["BaseUnifier"]

RawT = TypeVar("RawT", bound=BaseRawEntry)


class BaseUnifier(ABC, Generic[RawT]):
    """
    Abstract base class for processing datasets and their entries into the
    unified format.

    Attributes:
        name (str): Unique identifier for the dataset.
        data_dir (Path): Root directory with `raw/` and `unified/` subdirectories.
        remove_insane (bool): Whether to discard problematic entries.
    """

    def __init__(
        self, *, name: str, data_dir: Path, remove_insane: bool = False
    ) -> None:
        """
        Args:
            name: Dataset name for file naming and metadata.
            data_dir: Base path containing raw data and where unified output will go.
            remove_insane: Whether to discard problematic entries or not.
        """
        self.name = name
        self.data_dir = data_dir
        self.remove_insane = remove_insane

        # Raw inputs
        self._raw_qas_files: list[Path] = []
        self._raw_document_files: list[Path] = []

        # Document corpus
        self._corpus_records: list[tuple[str, int, Path]] = []
        self._corpus_index: set[tuple[str, int]] = set()

        # Internal metadata state
        self._num_documents: int = 0
        self._num_pages: int = 0
        self._total_questions: int = 0
        self._splits: list[DatasetSplit] = []

        # Logger
        self.logger = get_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG,
            log_file_path=Path("logs/unify.log"),
        )

    @property
    def raw_qas_dir(self) -> Path:
        """
        Directory containing raw QA files.
        """
        return self.data_dir / "raw_qas"

    @property
    def raw_documents_dir(self) -> Path:
        """
        Directory containing raw document files.
        """
        return self.data_dir / "raw_documents"

    @property
    def documents_dir(self) -> Path:
        """
        Directory containing (processed) document files.
        """
        return self.data_dir / "documents"

    @property
    def qas_dir(self) -> Path:
        """
        Directory containing unified/processed QA files.
        """
        return self.data_dir / "unified_qas"

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
            len(self._raw_qas_files),
            len(self._raw_document_files),
        )

        # 2) Build `documents/` if we have raw_documents to process
        self._convert_raw_documents()
        self.logger.info("Documents directory ready at %s", self.documents_dir)

        # 3) Load the corpus into memory
        self._load_corpus()
        self.logger.info(
            "Loaded corpus with %d documents and %d pages",
            self._num_documents,
            self._num_pages,
        )

        # 4) Convert QA entries
        split_map: dict[str, list[UnifiedEntry]] = self._convert_raw_qas()
        self.logger.info(
            "Converted QA into %d splits totaling %d questions",
            len(split_map),
            self._total_questions,
        )

        # 5) Sanity check all unified entries against the corpus
        all_entries: list[UnifiedEntry] = [
            ue for entries in split_map.values() for ue in entries
        ]
        self._sanity_check(all_entries)

        # 6) Write outputs
        corpus_path = self._write_corpus()
        self.logger.info("Wrote corpus to %s", corpus_path)
        qas_path = self._write_unified_qas(split_map)
        self.logger.info("Wrote unified qas to %s", qas_path)
        metadata_path = self._write_metadata()
        self.logger.info("Wrote dataset metadata to %s", metadata_path)
        return self.qas_dir

    def _discover_raw(self) -> None:
        """
        Discover all raw QA files and raw document files (if needed).

        Populates:
            _raw_qas_files (list[Path]):
                A list of paths for each raw QA split file.
            _raw_document_files (list[Path]):
                A list of paths for each raw document file.

        Raises:
            FileNotFoundError: if `raw_qas_dir` does not exist.
        """

        if not (self.raw_qas_dir).exists():
            raise FileNotFoundError(f"Raw QA directory not found: {self.raw_qas_dir}")

        self._raw_qas_files = self._discover_raw_qas()

        if (self.documents_dir).exists():
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
        return sorted((self.raw_documents_dir).glob("*.pdf"))

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
                A list of (doc_id, page_number, image_path) for every page.
            _corpus_index (set[tuple[str, int]]):
                A set of (doc_id, page_number) pairs for O(1) lookup.
            _num_documents (int):
                The number of distinct documents discovered.
            _num_pages (int):
                The total number of pages across all documents.

        Raises:
            FileNotFoundError: If `self.documents_dir` does not exist.
        """
        if not self.documents_dir.exists():
            raise FileNotFoundError(
                f"Processed documents directory not found: {self.documents_dir}"
            )

        # reset caches and counters
        self._corpus_records = []
        self._corpus_index = set()
        self._num_documents = 0
        self._num_pages = 0

        for doc_dir in sorted(self.documents_dir.iterdir()):
            if not doc_dir.is_dir():
                self.logger.debug(
                    "Skipping non-directory in documents_dir: %s", doc_dir
                )
                continue
            doc_id = doc_dir.name
            self._num_documents += 1

            for img_path in sorted(doc_dir.glob("*.jpg")):
                try:
                    page_number = int(img_path.stem)
                except ValueError:
                    # skip files whose stem isn’t an integer
                    self.logger.debug(
                        "Skipping non-integer page file %s in document %s",
                        img_path.name,
                        doc_id,
                    )
                    continue
                self._num_pages += 1
                self._corpus_records.append((doc_id, page_number, img_path))
                self._corpus_index.add((doc_id, page_number))

            count = len([p for (d, p, _) in self._corpus_records if d == doc_id])
            self.logger.debug("Loaded document %s with %d pages", doc_id, count)

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
        for qa_path in self._raw_qas_files:
            split_name = qa_path.stem
            raw_entries = self._load_raw_qas(qa_path)
            unified_list: list[UnifiedEntry] = []
            self.logger.info(
                "Converting split %s with %d raw entries", split_name, len(raw_entries)
            )
            for raw in raw_entries:
                ue = self._convert_qa_entry(raw)
                if ue is not None:
                    unified_list.append(ue)

            self.logger.debug(
                "Split %s converted to %d unified entries",
                split_name,
                len(unified_list),
            )

            # update metadata
            num = len(unified_list)
            self._total_questions += num
            self._splits.append(DatasetSplit(name=split_name, num_questions=num))
            split_map[split_name] = unified_list
        return split_map

    @abstractmethod
    def _convert_qa_entry(self, raw: RawT) -> UnifiedEntry | None:
        """
        Map a raw entry into the unified schema.

        Args:
            raw: A raw dataset entry.

        Returns:
            A UnifiedEntry instance.
        """
        ...

    def _convert_raw_documents(self) -> None:
        """
        Convert raw document files into the expected `documents/` layout.

        Raises:
            ConversionError: if any document fails to convert.
        """

        # Nothing to do if there are no raw inputs or the target already exists
        if not self._raw_document_files or (self.documents_dir).exists():
            self.logger.debug("Skipping document processing")
            return

        # Write to a temporary directory then swap
        tmp_dir = self.data_dir / ".tmp_documents"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir()

        for raw_path in self._raw_document_files:
            doc_id = raw_path.stem
            output_dir = self.documents_dir / doc_id
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(
                "Converting raw document %s → %s", raw_path.name, output_dir
            )
            # delegate to utility or subclass override
            self._convert_document(raw_path, output_dir)

        tmp_dir.rename(self.documents_dir)

        self.logger.info(
            "Processed %d raw documents into %s",
            len(self._raw_document_files),
            self.documents_dir,
        )

    def _convert_document(self, raw_path: Path, output_dir: Path) -> None:
        """
        Convert one raw document file into a directory of JPEG pages.

        By default, this assumes PDF input and delegates to a shared utility.
        Subclasses may override if they require a different conversion pipeline.

        Args:
            raw_path (Path): Path to the raw document (e.g., a PDF file).
            output_dir (Path): Directory under `documents_dir` where pages
                should be written as `0.jpg`, `1.jpg`, etc.

        Raises:
            ConversionError: If conversion fails for any page.
        """
        pass

    def _sanity_check(
        self, entries: list[UnifiedEntry]
    ) -> None:  # TODO: ADD PAGE LEVEL CHECKS
        """
        Verify that every evidence page in the unified entries exists in the corpus.

        This method requires that `_load_corpus()` has already been called to
        populate `self._corpus_index`.

        Args:
            entries (list[UnifiedEntry]): A list of `UnifiedEntry` objects whose `evidence.pages`
                should be validated against the known corpus.

        Raises:
            RuntimeError: If the corpus has not been loaded yet.
            ValueError: If any entry references a (doc_id, page) not found in the corpus.
        """
        if not getattr(self, "_corpus_index", None):
            raise RuntimeError(
                "Corpus index is empty – call `_load_corpus()` before sanity check."
            )

        invalid_entries = []
        for ue in entries:
            if ue.evidence:
                for page_num in ue.evidence.pages:
                    if (ue.document.id, page_num) not in self._corpus_index:
                        invalid_entries.append(ue)
                        break

        if invalid_entries:
            if self.remove_insane:
                for ue in invalid_entries:
                    self.logger.warning(
                        "Removing entry %s; referenced pages %s not in corpus",
                        ue.id,
                        ue.evidence.pages,
                    )
                    # drop them in place
                entries[:] = [ue for ue in entries if ue not in invalid_entries]
            else:
                msgs = [
                    f"{ue.id}: missing pages {ue.evidence.pages}"
                    for ue in invalid_entries
                ]
                raise ValueError(
                    "Corpus sanity check failed for the following entries:\n  "
                    + "\n  ".join(msgs)
                )

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
        tmp_path = self.data_dir / "corpus.jsonl.tmp"
        with tmp_path.open("w", encoding="utf-8") as f:
            for doc_id, page_num, img_path in self._corpus_records:
                page = CorpusPage(
                    doc_id=doc_id, page_number=page_num, image_path=img_path
                )
                f.write(page.model_dump_json())
                f.write("\n")
        output_path = self.data_dir / "corpus.jsonl"
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
        tmp_dir = self.data_dir / ".tmp_qas"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir()

        for split_name, entries in split_map.items():
            file_path = tmp_dir / f"{split_name}.jsonl"
            with file_path.open("w", encoding="utf-8") as f:
                for ue in entries:
                    f.write(ue.model_dump_json())
                    f.write("\n")

        if (self.qas_dir).exists():
            shutil.rmtree(self.qas_dir)
        tmp_dir.rename(self.qas_dir)

        return self.qas_dir

    def _write_metadata(self) -> Path:
        """
        Write dataset metadata to a JSON file.

        Returns:
            Path: The path to the written metadata.json file.
        """
        meta = DatasetMetadata(
            name=self.name,
            num_documents=self._num_documents,
            num_pages=self._num_pages,
            num_questions=self._total_questions,
            splits=self._splits,
        )
        meta_path = self.data_dir / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            f.write(meta.model_dump_json(indent=2))
        return meta_path
