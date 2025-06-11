import ast
import json
from pathlib import Path

from docrag.schema.entries import (
    Answer,
    AnswerFormat,
    AnswerType,
    Document,
    DocumentType,
    Evidence,
    EvidenceSource,
    Question,
    UnifiedEntry,
    MMLongBenchDocRawEntry,
)
from docrag.schema.entries.utils import tag_inferred, tag_missing

from ..unifier import Unifier
from ..registry import register
from ..exceptions import UnificationError

__all__ = ["MMLongBenchDocUnifier"]


@register("mmlongbenchdoc")
class MMLongBenchDocUnifier(Unifier[MMLongBenchDocRawEntry]):
    """
    Unifier for the MMLongBench-Doc dataset.
    """

    def _discover_raw_qas(self) -> list[Path]:
        # All JSON files under raw_qas/
        return sorted(self.raw_qas_directory.glob("*.json"))

    def _load_raw_qas(self, path: Path) -> list[MMLongBenchDocRawEntry]:
        # Entries are in a top level array
        data = json.loads(path.read_text(encoding="utf-8"))
        entries: list[MMLongBenchDocRawEntry] = []
        for idx, item in enumerate(data):
            item["question_id"] = str(idx)
            entries.append(MMLongBenchDocRawEntry.model_validate(item))
        return entries

    def _build_entry(
        self,
        raw: MMLongBenchDocRawEntry,
        document: Document,
        question: Question,
        evidence: Evidence,
        answer: Answer,
    ) -> UnifiedEntry:
        document_name = Path(raw.doc_id).stem
        entry = UnifiedEntry(
            id=f"{document_name}-{raw.question_id}",
            document=document,
            question=question,
            evidence=evidence,
            answer=answer,
        )

        return entry

    def _build_document(self, raw: MMLongBenchDocRawEntry) -> Document | None:
        count_pages = len(self._corpus_documents.get(Path(raw.doc_id).stem, []))

        if count_pages == 0:
            if self.test_mode:
                self.logger.warning(
                    "Document not found in corpus: doc_id=%s", raw.doc_id
                )
                self._problematic["documents"].append(
                    {
                        "document_id": raw.doc_id,
                        "reason": "Document missing in corpus records.",
                    }
                )
                return None
            else:
                raise UnificationError(
                    f"Document '{raw.doc_id}' not found in corpus. Cannot compute count_pages "
                )

        document = Document(
            id=Path(raw.doc_id).stem,
            type=self._map_document_type(raw.doc_type),
            count_pages=count_pages,
        )
        return document

    def _build_question(self, raw: MMLongBenchDocRawEntry) -> Question:
        question = Question(id=str(raw.question_id), text=raw.question)
        question.tags.append(
            tag_missing("type", "MMLongBenchDoc does not provide question types.")
        )
        return question

    def _build_evidence(self, raw: MMLongBenchDocRawEntry) -> Evidence:
        evidence = Evidence()

        if raw.answer.strip().lower() == "not answerable":
            evidence.sources = [EvidenceSource.NONE]
            return evidence

        pages: list[int] = []
        parsed = self._parse_list_int(raw.evidence_pages)
        if parsed:
            pages = [p - 1 for p in parsed]  # convert to 0-indexed
        else:
            evidence.tags.append(
                tag_missing("pages", "Answer pages not provided for this question.")
            )

        document_pages = self._corpus_documents.get(Path(raw.doc_id).stem, [])

        if pages:
            missing = [p for p in pages if p not in document_pages]

            if missing:
                if self.test_mode:
                    valid = [p for p in pages if p in document_pages]
                    evidence.pages = valid or [0]
                    self.logger.warning(
                        "Extracted pages %s not in corpus for doc_id=%s. Using %s instead.",
                        missing,
                        raw.doc_id,
                        evidence.pages,
                    )
                    self._problematic["questions"].append(
                        {
                            "question_id": raw.question_id,
                            "reason": f"Provided pages {pages} not found in corpus; used {evidence.pages}.",
                        }
                    )
                else:
                    raise UnificationError(
                        f"Provided pages {missing} do not exist in corpus for "
                        f"doc_id={raw.doc_id}, question_id={raw.question_id}."
                    )
            else:
                # All provided pages are valid
                evidence.pages = pages

        # 5) If no valid evidence.pages yet, fallback to all pages or [0]
        if not evidence.pages:
            if document_pages:
                evidence.pages = document_pages
                evidence.tags.append(
                    tag_inferred("pages", "Set evidence pages to all document pages.")
                )
            else:
                if self.test_mode:
                    evidence.pages = [0]
                    self.logger.warning(
                        "No document pages found in corpus for doc_id=%s.", raw.doc_id
                    )
                    self._problematic.setdefault("questions", []).append(
                        {
                            "question_id": raw.question_id,
                            "reason": "Document pages missing in corpus.",
                        }
                    )
                else:
                    raise UnificationError(
                        f"Document '{raw.doc_id}' has no associated pages in the corpus "
                        f"(question_id={raw.question_id})."
                    )

        # 6) Map and validate sources
        mapped_sources = self._map_evidence_sources(raw.evidence_sources)
        if mapped_sources and mapped_sources != [EvidenceSource.OTHER]:
            evidence.sources = mapped_sources
        else:
            evidence.tags.append(
                tag_missing(
                    "sources", "Evidence source not provided for this question."
                )
            )

        return evidence

    def _build_answer(self, raw: MMLongBenchDocRawEntry) -> Answer:
        answer = Answer()

        if not raw.answer or raw.answer.strip().lower() == "not answerable":
            answer.type = AnswerType.NOT_ANSWERABLE
            return answer

        variant_str = str(raw.answer).strip() or ""
        if answer.format == AnswerFormat.LIST:
            try:
                parsed = ast.literal_eval(variant_str)
                if isinstance(parsed, list):
                    variant_str = repr(parsed)
            except (ValueError, SyntaxError):
                # leave as-is on parse failure
                pass

        answer.variants = [variant_str]
        answer.format = self._map_answer_format(raw.answer_format)
        answer.type = AnswerType.ANSWERABLE
        answer.tags.append(
            tag_missing(
                "rationale", "MMLongBenchDoc does not provide rationale for answers."
            )
        )

        return answer

    def _map_evidence_sources(self, sources: str) -> list[EvidenceSource]:
        parsed = self._parse_list_string(sources)
        mapping = {
            "Pure-text (Plain-text)": EvidenceSource.SPAN,
            "Table": EvidenceSource.TABLE,
            "Chart": EvidenceSource.CHART,
            "Figure": EvidenceSource.IMAGE,
            "Generalized-text (Layout)": EvidenceSource.LAYOUT,
        }
        return [mapping.get(src, EvidenceSource.OTHER) for src in parsed]

    def _map_answer_format(self, format: str) -> AnswerFormat:
        mapping = {
            "int": AnswerFormat.INTEGER,
            "str": AnswerFormat.STRING,
            "none": AnswerFormat.NONE,
            "float": AnswerFormat.FLOAT,
            "list": AnswerFormat.LIST,
        }
        return mapping.get(format.lower(), AnswerFormat.OTHER)

    def _map_document_type(self, doc_type: str) -> DocumentType:
        mapping = {
            "research report / introduction": DocumentType.SCIENTIFIC,
            "academic paper": DocumentType.SCIENTIFIC,
            "guidebook": DocumentType.TECHNICAL,
            "tutorial/workshop": DocumentType.TECHNICAL,
            "financial report": DocumentType.FINANCIAL,
            "brochure": DocumentType.MARKETING,
            "administration/industry file": DocumentType.POLICY,  # internal/organizational docs
        }
        return mapping.get(doc_type.lower(), DocumentType.OTHER)

    def _parse_list_int(self, raw_list: str) -> list[int]:
        """
        Safely parse a string-encoded list of ints into an actual list of ints.
        E.g. "[1, 2, 3]" → [1, 2, 3]; "[]" or "" → [].
        """
        if not raw_list:
            return []
        try:
            parsed = ast.literal_eval(raw_list)
            if isinstance(parsed, list):
                return [
                    int(item)
                    for item in parsed
                    if isinstance(item, (int, str)) and str(item).isdigit()
                ]
        except (ValueError, SyntaxError):
            pass

        # Fallback: strip brackets and split on commas
        cleaned = raw_list.strip().lstrip("[").rstrip("]")
        ints: list[int] = []
        for part in cleaned.split(","):
            part = part.strip().strip("'\"")
            if part.isdigit():
                ints.append(int(part))
        return ints

    def _parse_list_string(self, list_string: str) -> list[str]:
        """
        Safely parse a string-encoded list of strings into an actual list of strings.
        E.g. "['Chart', 'Figure']" → ['Chart', 'Figure']; "[]" → [].
        """
        try:
            parsed = ast.literal_eval(list_string)
            if isinstance(parsed, list):
                return [s.strip() for s in parsed if isinstance(s, str)]
        except (ValueError, SyntaxError):
            pass
        return []
