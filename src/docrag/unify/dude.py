from pathlib import Path
import json

from docrag.schema.enums import EvidenceSource
from docrag.schema.raw_entry import DUDERaw
from docrag.unify.base import BaseUnifier
from docrag.schema import (
    UnifiedEntry,
    Question,
    Document,
    Evidence,
    Answer,
    QuestionType,
    AnswerFormat,
    AnswerType,
)
from docrag.schema.utils import tag_missing, tag_low_quality, tag_inferred

__all__ = ["DUDEUnifier"]


class DUDEUnifier(BaseUnifier[DUDERaw]):
    """
    Unifier for the DUDE competition dataset.
    """

    def _discover_raw_qas(self) -> list[Path]:
        # All of the DUDE files are plain JSON under raw_qas/
        return sorted(self.raw_qas_dir.glob("*.json"))

    def _load_raw_qas(self, path: Path) -> list[DUDERaw]:
        # Entries are in a top level 'data' array
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [DUDERaw.model_validate(item) for item in payload["data"]]

    def _convert_qa_entry(self, raw):
        question = self._build_question(raw)
        document = self._build_document(raw)
        evidence = self._build_evidence(raw)
        answer = self._build_answer(raw)

        entry = UnifiedEntry(
            id=f"{raw.question_id}",
            question=question,
            document=document,
            evidence=evidence,
            answer=answer,
        )

        if raw.data_split == "test":
            entry.tags.append(tag_missing("evidence"))
            entry.tags.append(tag_missing("answer"))

        return entry

    def _build_question(self, raw):
        """
        Construct the Question model.
        """
        q = Question(id=raw.question_id, text=raw.question)
        if raw.data_split != "test":
            q.type = self._map_question_type(raw.answer_type.lower())
            q.tags.append(tag_low_quality("type"))
        else:
            q.tags.append(tag_missing("type"))
        return q

    def _build_document(self, raw):
        """
        Construct the Document model.
        """
        num_pages = sum(1 for d, _, _ in self._corpus_records if d == raw.doc_id)
        doc = Document(id=raw.doc_id, num_pages=num_pages)
        doc.tags.append(tag_missing("type"))
        return doc

    def _build_evidence(self, raw):
        """
        Construct the Evidence model.
        """
        ev = Evidence()
        if raw.data_split == "test":
            return ev
        elif raw.data_split != "test" and raw.answer_type == "not-answerable":
            ev.sources = [EvidenceSource.NONE]
        elif (
            raw.data_split != "test"
            and raw.answer_type != "not-answerable"
            and raw.answers_page_bounding_boxes
        ):
            pages = {b.page for grp in raw.answers_page_bounding_boxes for b in grp}
            ev.pages = sorted(pages)
            ev.tags.append(tag_missing("sources"))
        else:
            ev.pages = [p for d, p, _ in self._corpus_records if d == raw.doc_id]
            if ev.pages:
                ev.tags.append(tag_missing("pages"))
                ev.tags.append(
                    tag_inferred("pages", "Set evidence pages to all pages.")
                )
            else:
                ev.pages = [0]
                ev.tags.append(
                    tag_missing("pages", "Could not find document pages in corpus.")
                )
        return ev

    def _build_answer(self, raw):
        """
        Construct the Answer Model
        """
        if raw.data_split == "test":
            return Answer()
        if raw.answer_type == "not-answerable":
            return Answer(type=AnswerType.NOT_ANSWERABLE)

        ans = Answer()
        if len(raw.answers) > 1:
            ans.variants = [self._handle_list_answers(raw.answers)]
            ans.format = AnswerFormat.LIST
            ans.type = AnswerType.ANSWERABLE
        else:
            single = str(raw.answers[0])
            ans.variants = [single] + (raw.answers_variants or [])
            ans.tags.append(tag_missing("format"))
            ans.type = AnswerType.ANSWERABLE
        return ans

    def _map_question_type(self, answer_type_raw: str) -> QuestionType:
        """
        Map the raw answer_type string to our QuestionType enum.
        """
        mapping: dict[str, QuestionType] = {
            "extractive": QuestionType.EXTRACTIVE,
            "list/extractive": QuestionType.EXTRACTIVE,
            "abstractive": QuestionType.ABSTRACTIVE,
            "list/abstractive": QuestionType.ABSTRACTIVE,
        }
        return mapping.get(answer_type_raw, QuestionType.OTHER)

    def _handle_list_answers(self, answers: list[str]) -> str:
        """
        Convert a list of answer items into a single string representation,
        e.g. ["foo", "bar"] → "['foo', 'bar']". Numeric items are left unquoted.
        """
        repr_items = [f"'{item}'" if not item.isdigit() else item for item in answers]
        return "[" + ", ".join(repr_items) + "]"
