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
    QuestionType,
    UnifiedEntry,
    TATDQARawEntry,
)
from docrag.schema.entries.utils import tag_inferred, tag_low_quality, tag_missing

from ..unifier import Unifier
from ..registry import register
from ..exceptions import UnificationError

__all__ = ["TATDQAUnifier"]


@register("tatdqa")
class TATDQAUnifier(Unifier[TATDQARawEntry]):
    """
    Unifier for the TATDQA dataset.
    """

    def _discover_raw_qas(self) -> list[Path]:
        # All raw TAT-DQA splits are plain JSON files under raw_qas_directory/
        return sorted(self.raw_qas_directory.glob("*.json"))

    def _load_raw_qas(self, path: Path) -> list[TATDQARawEntry]:
        # Entries are in a top level array
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [TATDQARawEntry.model_validate(item) for item in payload]

    def _build_entry(
        self,
        raw: TATDQARawEntry,
        document: Document,
        question: Question,
        evidence: Evidence,
        answer: Answer,
    ) -> UnifiedEntry:
        entry = UnifiedEntry(
            id=f"{raw.doc_uid}-{raw.question_uid}",
            document=document,
            question=question,
            evidence=evidence,
            answer=answer,
        )
        return entry

    def _build_document(self, raw: TATDQARawEntry) -> Document | None:
        corpus_pages = self._corpus_documents.get(raw.doc_uid, [])

        if len(corpus_pages) == 0:
            # Document not found in corpus
            if self.test_mode:
                self.logger.warning(
                    "Document missing in corpus: doc_uid=%s", raw.doc_uid
                )
                self._problematic["documents"].append(
                    {"document_id": raw.doc_uid, "reason": "Missing in corpus."}
                )
                return None
            else:
                raise UnificationError(f"Document '{raw.doc_uid}' not found in corpus.")

        document = Document(
            id=raw.doc_uid,
            count_pages=len(corpus_pages),
        )
        document.type = DocumentType.FINANCIAL
        document.tags.append(
            tag_inferred("type", "All TAT-DQA documents are financial reports.")
        )
        return document

    def _build_question(self, raw: TATDQARawEntry) -> Question:
        question = Question(id=raw.question_uid, text=raw.question)
        question.type = self._map_question_type(raw.answer_type)
        question.tags.append(
            tag_low_quality(
                "type",
                "Question type  classification in TATDQA is limited. Only provides four broad categories.",
            )
        )
        return question

    def _build_evidence(self, raw: TATDQARawEntry) -> Evidence:
        evidence = Evidence()
        page_idx = raw.doc_page - 1

        if (raw.doc_uid, page_idx) not in self._corpus_index:
            if self.test_mode:
                evidence.pages = [0]
                self.logger.warning(
                    "doc_page %s for doc_uid=%s not found in corpus; defaulting to page 0",
                    page_idx,
                    raw.doc_uid,
                )
                self._problematic["questions"].append(
                    {
                        "question_uid": raw.question_uid,
                        "reason": "doc_page not found in corpus",
                    }
                )
            else:
                raise UnificationError(
                    f"doc_page {page_idx} not in corpus for doc_uid={raw.doc_uid}, "
                    f"question_uid={raw.question_uid}"
                )

        evidence.pages = [page_idx]
        evidence.tags.append(
            tag_missing("sources", "TATDQA does not provide evidence sources.")
        )

        return evidence

    def _build_answer(self, raw: TATDQARawEntry) -> Answer:
        answer = Answer()

        if not raw.answer:
            if self.test_mode:
                self.logger.warning(
                    "Missing answer for question_id=%s.", raw.question_uid
                )
                self._problematic["questions"].append(
                    {
                        "question_id": raw.question_uid,
                        "reason": "Missing answer.",
                    }
                )
                return answer
            else:
                raise UnificationError(
                    f"answer is empty or None for question_id={raw.question_uid}"
                )

        if isinstance(raw.answer, list):
            if len(raw.answer) > 1:
                combined = self._handle_list_answers(raw.answer)
                answer.variants = [combined]
                answer.format = AnswerFormat.LIST
            else:
                single = raw.answer[0]
                answer.variants = [str(single)]
                answer.format = self._infer_answer_format(single)
        else:
            answer.variants = [str(raw.answer)]
            answer.format = self._infer_answer_format(raw.answer)

        if raw.derivation:
            answer.rationale = raw.derivation
        else:
            answer.tags.append(
                tag_missing("rationale", "Rationale not provided for this answer.")
            )

        answer.type = AnswerType.ANSWERABLE
        answer.tags.append(
            tag_inferred(
                "format", "Answer format has been inferred based on it's data type."
            )
        )
        return answer

    def _infer_answer_format(self, answer: str | int | float) -> AnswerFormat:
        """
        Infer the answer format by checking the type of the answer field.
        """
        if isinstance(answer, int):
            return AnswerFormat.INTEGER
        if isinstance(answer, float):
            return AnswerFormat.FLOAT
        return AnswerFormat.STRING

    def _map_question_type(self, answer_type: str) -> QuestionType:
        """
        Map the raw answer_type string to our QuestionType enum.
        """
        mapping: dict[str, QuestionType] = {
            "span": QuestionType.EXTRACTIVE,
            "multi-span": QuestionType.EXTRACTIVE,
            "arithmetic": QuestionType.ARITHMETIC,
            "count": QuestionType.COUNTING,
        }
        return mapping.get(answer_type, QuestionType.OTHER)

    def _map_evidence_source(self, answer_type: str) -> EvidenceSource:
        """
        Map the raw answer_type string to our EvidenceSource enum.
        """
        mapping: dict[str, EvidenceSource] = {
            "span": EvidenceSource.SPAN,
            "multi-span": EvidenceSource.SPAN,
        }
        return mapping.get(answer_type, EvidenceSource.OTHER)

    def _handle_list_answers(self, answers: list[str]) -> str:
        """
        Convert a list of answer items into a single string representation,
        e.g. ["foo", "bar"] → "['foo', 'bar']".
        """
        item_strings = [f"'{item}'" for item in answers]
        return "[" + ", ".join(item_strings) + "]"
