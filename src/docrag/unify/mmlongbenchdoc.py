from pathlib import Path
import json
import ast

from docrag.schema.enums import AnswerFormat, AnswerType, EvidenceSource, DocumentType
from docrag.schema.raw_entry import MMLongBenchDocRaw
from docrag.unify.base import BaseUnifier
from docrag.schema import UnifiedEntry, Question, Document, Evidence, Answer
from docrag.schema.utils import tag_missing, tag_inferred

__all__ = ["MMLongBenchDocUnifier"]


class MMLongBenchDocUnifier(BaseUnifier[MMLongBenchDocRaw]):
    """
    Unifier for the MMLongBench-Doc dataset.
    """

    def _discover_raw_qas(self) -> list[Path]:
        # All JSON files under raw_qas/
        return sorted(self.raw_qas_dir.glob("*.json"))

    def _load_raw_qas(self, path: Path) -> list[MMLongBenchDocRaw]:
        # Entries are in a top level array
        data = json.loads(path.read_text(encoding="utf-8"))
        return [MMLongBenchDocRaw.model_validate(item) for item in data]

    def _convert_qa_entry(self, raw: MMLongBenchDocRaw) -> UnifiedEntry:
        stemmed_doc_id = Path(raw.doc_id).stem
        question = self._build_question(raw)
        document = self._build_document(raw)
        evidence = self._build_evidence(raw)
        answer = self._build_answer(raw)

        entry = UnifiedEntry(
            id=f"{stemmed_doc_id}_{question.id}",
            question=question,
            document=document,
            evidence=evidence,
            answer=answer,
        )

        return entry

    def _build_question(self, raw: MMLongBenchDocRaw) -> Question:
        """
        Construct the Question model.
        """
        q_id = f"{abs(hash(raw.question)) % 10000}"
        q = Question(id=q_id, text=raw.question)
        q.tags.append(tag_missing("type"))
        return q

    def _build_document(self, raw: MMLongBenchDocRaw) -> Document:
        """
        Construct the Document model.
        """
        num_pages = sum(
            1 for d, _, _ in self._corpus_records if d == Path(raw.doc_id).stem
        )
        doc = Document(
            id=Path(raw.doc_id).stem,
            type=self._map_document_type(raw.doc_type),
            num_pages=num_pages,
        )
        return doc

    def _build_evidence(self, raw: MMLongBenchDocRaw) -> Evidence:
        """
        Construct the Evidence model.
        """
        ev = Evidence()

        if str(raw.answer).strip().lower() == "not answerable":
            ev.sources = [EvidenceSource.NONE]
            return ev

        parsed_1idx = self._parse_list_int(raw.evidence_pages)
        if parsed_1idx:
            ev.pages = [p - 1 for p in parsed_1idx]
        else:
            ev.tags.append(tag_missing("pages"))

        mapped_sources = self._map_evidence_sources(raw.evidence_sources)
        if mapped_sources and mapped_sources != [EvidenceSource.OTHER]:
            ev.sources = mapped_sources
        else:
            ev.tags.append(tag_missing("sources"))

        if not ev.pages:
            all_pages = [
                p for d, p, _ in self._corpus_records if d == Path(raw.doc_id).stem
            ]
            if all_pages:
                ev.pages = all_pages
                ev.tags.append(tag_missing("pages"))
                ev.tags.append(
                    tag_inferred("pages", "Set evidence pages to all document pages.")
                )
            else:
                ev.pages = [0]
                ev.tags.append(
                    tag_missing("pages", "Could not find document pages in corpus.")
                )

        return ev

    def _build_answer(self, raw: MMLongBenchDocRaw) -> Answer:
        ans = Answer()

        if raw.answer is None or str(raw.answer).strip().lower() == "not answerable":
            ans.format = AnswerFormat.NONE
            ans.type = AnswerType.NOT_ANSWERABLE
            return ans

        variant_str = str(raw.answer).strip() or ""
        if ans.format == AnswerFormat.LIST:
            try:
                parsed = ast.literal_eval(variant_str)
                if isinstance(parsed, list):
                    variant_str = repr(parsed)
            except (ValueError, SyntaxError):
                # leave as-is on parse failure
                pass

        ans.variants = [variant_str]
        ans.format = self._map_answer_format(raw.answer_format)
        ans.type = AnswerType.ANSWERABLE

        return ans

    def _map_evidence_sources(self, raw_sources: str) -> list[EvidenceSource]:
        parsed = self._parse_list_string(raw_sources)
        mapping = {
            "Pure-text (Plain-text)": EvidenceSource.SPAN,
            "Table": EvidenceSource.TABLE,
            "Chart": EvidenceSource.CHART,
            "Figure": EvidenceSource.IMAGE,
            "Generalized-text (Layout)": EvidenceSource.LAYOUT,
        }
        return [mapping.get(src, EvidenceSource.OTHER) for src in parsed]

    def _map_answer_format(self, fmt: str) -> AnswerFormat:
        mapping = {
            "int": AnswerFormat.INTEGER,
            "str": AnswerFormat.STRING,
            "none": AnswerFormat.NONE,
            "float": AnswerFormat.FLOAT,
            "list": AnswerFormat.LIST,
        }
        return mapping.get(fmt.lower(), AnswerFormat.OTHER)

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
