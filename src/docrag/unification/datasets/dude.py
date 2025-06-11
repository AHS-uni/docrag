import json
from pathlib import Path

from docrag.schema.entries import (
    Answer,
    AnswerFormat,
    AnswerType,
    Document,
    Evidence,
    EvidenceSource,
    Question,
    QuestionType,
    UnifiedEntry,
    DUDERawEntry,
)
from docrag.schema.entries.utils import tag_inferred, tag_low_quality, tag_missing

from ..unifier import Unifier
from ..registry import register
from ..exceptions import UnificationError

__all__ = ["DUDEUnifier"]


@register("dude")
class DUDEUnifier(Unifier[DUDERawEntry]):
    """
    Unifier for the DUDE dataset.
    """

    def _discover_raw_qas(self) -> list[Path]:
        # All of the DUDE files are plain JSON under raw_qas/
        return sorted(self.raw_qas_directory.glob("*.json"))

    def _load_raw_qas(self, path: Path) -> list[DUDERawEntry]:
        # Entries are in a top level 'data' array
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [DUDERawEntry.model_validate(item) for item in payload["data"]]

    def _build_entry(
        self,
        raw: DUDERawEntry,
        document: Document,
        question: Question,
        evidence: Evidence,
        answer: Answer,
    ) -> UnifiedEntry:
        entry = UnifiedEntry(
            id=f"{raw.doc_id}-{raw.question_id}",
            document=document,
            question=question,
            evidence=evidence,
            answer=answer,
        )

        if raw.data_split.strip().lower() == "test":
            entry.tags.append(
                tag_missing(
                    "evidence",
                    "DUDE is a competition dataset. No evidence provided in test split.",
                )
            )
            entry.tags.append(
                tag_missing(
                    "answer",
                    "DUDE is a competition dataset. No answer provided in test split.",
                )
            )

        return entry

    def _build_document(self, raw):
        count_pages = len(self._corpus_documents.get(raw.doc_id, []))

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
                    f"Document '{raw.doc_id}' not found in corpus."
                    f"for question_id={raw.question_id}."
                )

        document = Document(id=raw.doc_id, count_pages=count_pages)
        document.tags.append(
            tag_missing("type", "DUDE does not provide document types.")
        )
        return document

    def _build_question(self, raw):
        question = Question(id=raw.question_id, text=raw.question)
        if raw.data_split == "test":
            question.tags.append(
                tag_missing(
                    "type",
                    "Question type is not available in the test split. DUDE considers question type to be a part of the answer.",
                )
            )
            return question

        assert raw.answer_type, "Expected non-empty answer type for non-test questions"

        question.type = self._map_question_type(raw.answer_type.strip().lower())
        question.tags.append(
            tag_low_quality(
                "type",
                "Question type classification in DUDE is limited. Only provides four broad categories.",
            )
        )

        return question

    def _build_evidence(self, raw):
        evidence = Evidence()

        if raw.data_split.strip().lower() == "test":
            return evidence

        assert raw.answer_type, "Expected non-empty answer type for non-test questions"

        if raw.answer_type.strip().lower() == "not-answerable":
            evidence.sources = [EvidenceSource.NONE]
            return evidence

        document_pages = self._corpus_documents.get(raw.doc_id, [])
        if not raw.answers_page_bounding_boxes:
            evidence.pages = document_pages
            evidence.tags.append(
                tag_inferred(
                    "pages",
                    "No evidence pages provided for this question. Using all document pages.",
                )
            )
            evidence.tags.append(
                tag_missing("sources", "DUDE does not provide evidence sources.")
            )
            return evidence

        pages = {b.page for grp in raw.answers_page_bounding_boxes for b in grp}
        missing = [p for p in pages if p not in document_pages]
        if missing:
            if self.test_mode:
                valid_pages = sorted(p for p in pages if p in document_pages)
                evidence.pages = valid_pages or [0]
                self.logger.warning(
                    "Extracted pages %s not in corpus for doc_id=%s. Using %s instead.",
                    missing,
                    raw.doc_id,
                    evidence.pages,
                )
                self._problematic["questions"].append(
                    {
                        "question_id": raw.question_id,
                        "reason": f"Extracted pages {missing} not found in corpus.",
                    }
                )
            else:
                raise UnificationError(
                    f"Pages {missing} extracted from bounding boxes do not exist in corpus for "
                    f"doc_id={raw.doc_id}, question_id={raw.question_id}."
                )

        evidence.pages = sorted(pages)
        evidence.tags.append(
            tag_missing("sources", "DUDE does not provide evidence sources.")
        )
        return evidence

    def _build_answer(self, raw):
        if raw.data_split.strip().lower() == "test":
            return Answer()
        if raw.answer_type == "not-answerable":
            return Answer(type=AnswerType.NOT_ANSWERABLE)

        assert raw.answers, (
            "Expected non-empty answers list for answerable non-test questions"
        )

        answer = Answer()
        if len(raw.answers) > 1:
            answer.variants = [self._handle_list_answers(raw.answers)]
            answer.format = AnswerFormat.LIST
        else:
            single = str(raw.answers[0])
            answer.variants = [single] + (raw.answers_variants or [])

        answer.type = AnswerType.ANSWERABLE
        answer.tags.append(
            tag_missing("format", "DUDE does not provide answer formats.")
        )
        answer.tags.append(
            tag_missing("rationale", "DUDE does not provide rationale for answers.")
        )
        return answer

    def _map_question_type(self, answer_type: str) -> QuestionType:
        """
        Map the raw answer_type string to our QuestionType enum.
        """
        mapping: dict[str, QuestionType] = {
            "extractive": QuestionType.EXTRACTIVE,
            "list/extractive": QuestionType.EXTRACTIVE,
            "abstractive": QuestionType.ABSTRACTIVE,
            "list/abstractive": QuestionType.ABSTRACTIVE,
        }
        return mapping.get(answer_type, QuestionType.OTHER)

    def _handle_list_answers(self, answers: list[str]) -> str:
        """
        Convert a list of answer items into a single string representation,
        e.g. ["foo", "bar"] → "['foo', 'bar']".
        """
        item_strings = [f"'{item}'" for item in answers]
        return "[" + ", ".join(item_strings) + "]"
