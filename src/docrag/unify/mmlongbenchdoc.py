from pathlib import Path
import json

from docrag.schema.enums import AnswerFormat, AnswerType, EvidenceSource
from docrag.schema.raw_entry import MMLongBenchDocRaw
from docrag.unify.base import BaseUnifier
from docrag.schema import (
    UnifiedEntry,
    Question,
    Document,
    Evidence,
    Answer,
)


class MMLongBenchDocUnifier(BaseUnifier[MMLongBenchDocRaw]):
    """
    Unifier for the MMLongBench-Doc dataset.
    """

    def _discover_raw_qas(self) -> list[Path]:
        # All of the MMLongBench-Doc files are JSON under raw_qas/
        return sorted(self.raw_qas_dir.glob("*.json"))

    def _load_raw_qas(self, path: Path) -> list[MMLongBenchDocRaw]:
        # Load the JSON data
        data = json.loads(path.read_text(encoding="utf-8"))

        # Extract the dataset split from the filename (e.g., "train.json" -> "train")
        split = path.stem

        # For each entry, add the data_split field
        raw_entries = []
        for item in data:
            item["data_split"] = split  # UNNECESSARY: MMLONGBENCHDOC HAS ONLY ONE SPLIT
            raw_entries.append(MMLongBenchDocRaw.model_validate(item))

        return raw_entries

    def _convert_qa_entry(self, raw: MMLongBenchDocRaw) -> UnifiedEntry:
        """
        Map a raw MMLongBench-Doc entry into the unified schema.
        """
        # Build the Question model
        question = Question(
            id=f"{raw.doc_id}-{len(raw.question)}",  # Create an ID if none exists # UNNECESSARY: JUST USE QUESTION ID FROM RAW DATASET
            text=raw.question,
        )

        # Build the Document model
        document = Document(
            id=raw.doc_id,
            type=raw.doc_type,  # TODO: IMPLEMENT MAPPING FUNCTION FOR DOCUMENT TYPES "type 'str' cannot be assigned to parameter of type 'DocumentType'"
            # Since we don't have explicit page count, use the highest page number
            num_pages=(
                max(raw.evidence_pages) + 1 if raw.evidence_pages else 1
            ),  # NOT A VALID METHOD TO GET THE PAGE COUNT
            # TODO: USE THE SUPERCLASS ATTRIBUTE '_corpus_records' TO CALCULATE PAGE COUNTS
            # SEE LINES 103, 104 IN 'dude.py'
        )

        # Build the Evidence model
        evidence = Evidence(
            pages=raw.evidence_pages,
            sources=self._map_evidence_sources(raw.evidence_sources),
        )

        # Build the Answer model
        answer_format = self._map_answer_format(raw.answer_format)
        answer = Answer(
            type=(
                AnswerType.NOT_ANSWERABLE
                if raw.answer is None
                else AnswerType.ANSWERABLE
            ),
            variants=[str(raw.answer)] if raw.answer is not None else [],
            format=answer_format,
        )

        return UnifiedEntry(
            id=f"{raw.doc_id}-{hash(raw.question) % 10000}",  # Create a unique ID # UNNECESSARY:  '<doc-id>_<question_id>' IS SUFFICIENT
            question=question,
            document=document,
            evidence=evidence,
            answer=answer,
        )

    def _map_evidence_sources(self, sources: list[str]) -> list[EvidenceSource]:
        """
        Map the MMLongBench-Doc evidence sources to EvidenceSource enum values.
        """
        source_mapping = {
            "Pure-text (Plain-text)": EvidenceSource.SPAN,
            "Chart": EvidenceSource.CHART,
            "Table": EvidenceSource.TABLE,
            "Generalized-text (Layout)": EvidenceSource.SPAN,  # TODO: CHANGE TO 'LAYOUT'
        }

        return [source_mapping.get(source, EvidenceSource.OTHER) for source in sources]

    def _map_answer_format(self, format_str: str) -> AnswerFormat:
        """
        Map the MMLongBench-Doc answer format to AnswerFormat enum values.
        """
        format_mapping = {
            "Str": AnswerFormat.STRING,
            "Int": AnswerFormat.INTEGER,
            "Float": AnswerFormat.FLOAT,
            "List": AnswerFormat.LIST,
            "None": AnswerFormat.NONE,
        }

        return format_mapping.get(format_str, AnswerFormat.OTHER)

    def _map_document_type(self, doc_type):  # TODO
        pass
