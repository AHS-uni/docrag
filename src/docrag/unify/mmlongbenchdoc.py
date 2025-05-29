from pathlib import Path
import json

from docrag.schema.enums import AnswerFormat, AnswerType, EvidenceSource, DocumentType
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
            item["data_split"] = split
            raw_entries.append(MMLongBenchDocRaw.model_validate(item))
            
        return raw_entries

    def _convert_qa_entry(self, raw: MMLongBenchDocRaw) -> UnifiedEntry:
        """
        Map a raw MMLongBench-Doc entry into the unified schema.
        """
        # Build the Question model
        question = Question(
            id=f"q{hash(raw.question) % 10000}",  # Create a unique question ID
            text=raw.question,
        )

        # Build the Document model
        # Use corpus records to count pages
        page_numbers = [p for (doc, p, _) in self._corpus_records if doc == raw.doc_id]
        num_pages = len(page_numbers) if page_numbers else 1
        
        document = Document(
            id=raw.doc_id,
            type=self._map_document_type(raw.doc_type),
            num_pages=num_pages,
        )

        # Build the Evidence model
        evidence = Evidence(
            pages=raw.evidence_pages,
            sources=self._map_evidence_sources(raw.evidence_sources),
        )

        # Build the Answer model
        answer_format = self._map_answer_format(raw.answer_format)
        answer = Answer(
            type=AnswerType.NOT_ANSWERABLE if raw.answer is None else AnswerType.ANSWERABLE,
            variants=[str(raw.answer)] if raw.answer is not None else [],
            format=answer_format,
        )

        # If answerable but no evidence pages found, use all document pages
        if evidence.pages == [] and answer.type == AnswerType.ANSWERABLE:
            evidence = Evidence(pages=page_numbers)
            if evidence.pages == []:  # still no evidence
                self.logger.debug(
                    "Skipping entry with question '%s'. Unable to find evidence pages for document %s.",
                    raw.question[:30] + "..." if len(raw.question) > 30 else raw.question,
                    raw.doc_id,
                )
                return None  # Skip this entry due to missing evidence

        return UnifiedEntry(
            id=f"{raw.doc_id}_{question.id}",
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
            "Generalized-text (Layout)": EvidenceSource.OTHER,  # No LAYOUT in EvidenceSource enum
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
        
    def _map_document_type(self, doc_type: str) -> DocumentType:
        """
        Map the MMLongBench-Doc document types to DocumentType enum values.
        """
        # Map common document types to their corresponding enum values
        doc_type_lower = doc_type.lower()
        
        if "research" in doc_type_lower or "article" in doc_type_lower:
            return DocumentType.SCIENTIFIC
        elif "report" in doc_type_lower:
            return DocumentType.TECHNICAL
        elif "tutorial" in doc_type_lower or "workshop" in doc_type_lower:
            return DocumentType.TECHNICAL
        elif "policy" in doc_type_lower:
            return DocumentType.POLICY
        elif "legal" in doc_type_lower or "contract" in doc_type_lower:
            return DocumentType.LEGAL
        elif "news" in doc_type_lower:
            return DocumentType.NEWS
        elif "financial" in doc_type_lower:
            return DocumentType.FINANCIAL
        elif "correspondence" in doc_type_lower or "letter" in doc_type_lower:
            return DocumentType.CORRESPONDENCE
        else:
            return DocumentType.OTHER 
