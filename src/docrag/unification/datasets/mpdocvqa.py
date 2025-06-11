import json
from pathlib import Path

from docrag.schema.entries import (
    Answer,
    AnswerType,
    Document,
    Evidence,
    Question,
    UnifiedEntry,
    MPDocVQARawEntry,
)
from docrag.schema.entries.utils import tag_inferred, tag_missing

from ..unifier import Unifier
from ..registry import register
from ..exceptions import UnificationError

__all__ = ["MPDocVQAUnifier"]


@register("mpdocvqa")
class MPDocVQAUnifier(Unifier[MPDocVQARawEntry]):
    """
    Unifier for the MP-DocVQA competition dataset.
    """

    def _discover_raw_qas(self) -> list[Path]:
        # All of the MPDocVQA files are plain JSON under raw_qas/
        return sorted(self.raw_qas_directory.glob("*.json"))

    def _load_raw_qas(self, path: Path) -> list[MPDocVQARawEntry]:
        # Entries are in a top level 'data' array
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [MPDocVQARawEntry.model_validate(item) for item in payload["data"]]

    def _build_entry(
        self,
        raw: MPDocVQARawEntry,
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
                    "MPDocVQA is a competition dataset. No evidence provided in test split.",
                )
            )
            entry.tags.append(
                tag_missing(
                    "answer",
                    "MPDocVQA is a competition dataset. No answer provided in test split.",
                )
            )

        return entry

    def _build_document(self, raw: MPDocVQARawEntry) -> Document | None:
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
            tag_missing("type", "MPDocVQA does not provide document type.")
        )
        return document

    def _build_question(self, raw: MPDocVQARawEntry) -> Question:
        question = Question(id=str(raw.question_id), text=raw.question)
        question.tags.append(
            tag_missing("type", "MPDocVQA does not provide question type.")
        )
        return question

    def _build_evidence(self, raw: MPDocVQARawEntry) -> Evidence:
        evidence = Evidence()

        if raw.data_split.strip().lower() == "test":
            return evidence

        if raw.answer_page_idx is None:
            if self.test_mode:
                evidence.pages = [0]
                self.logger.warning(
                    "Missing answer_page_idx for doc_id=%s.", raw.doc_id
                )
                self._problematic["questions"].append(
                    {
                        "question_id": raw.question_id,
                        "reason": "Missing answer_page_idx.",
                    }
                )
                return evidence
            else:
                raise UnificationError(
                    f"answer_page_idx is None for doc_id={raw.doc_id}, question_id={raw.question_id}. "
                    f"Expected to index into: {raw.page_ids}"
                )

        try:
            page_id = raw.page_ids[raw.answer_page_idx]
            page_number = int(page_id.rsplit("_p", 1)[1])
        except Exception as e:
            if self.test_mode:
                evidence.pages = [0]
                self.logger.warning(
                    "Error parsing answer_page_idx for doc_id=%s.", raw.doc_id
                )
                self._problematic["questions"].append(
                    {
                        "question_id": raw.question_id,
                        "reason": "Invalid or unparseable answer_page_idx.",
                    }
                )
                return evidence
            else:
                raise UnificationError(
                    f"Failed to parse page number from page_ids[{raw.answer_page_idx}] "
                    f"for doc_id={raw.doc_id}, question_id={raw.question_id}: {raw.page_ids}"
                ) from e

        if (raw.doc_id, page_number) not in self._corpus_index:
            if self.test_mode:
                evidence.pages = [0]
                self.logger.warning(
                    "Extracted page %s not found in corpus for doc_id=%s.",
                    page_number,
                    raw.doc_id,
                )
                self._problematic["questions"].append(
                    {
                        "question_id": raw.question_id,
                        "reason": f"Extracted page {page_number} not found in corpus.",
                    }
                )
                return evidence
            else:
                raise UnificationError(
                    f"Page {page_number} extracted from answer_page_idx does not exist in corpus "
                    f"for doc_id={raw.doc_id}, question_id={raw.question_id}."
                )

        evidence.pages = [page_number]
        evidence.tags.append(
            tag_missing("sources", "MPDocVQA does not provide evidence sources.")
        )
        return evidence

    def _build_answer(self, raw: MPDocVQARawEntry) -> Answer:
        answer = Answer()
        if raw.data_split.strip().lower() == "test":
            return answer

        assert raw.answers, "Expected non-empty answers list for non-test questions"

        answer.variants = raw.answers
        answer.type = AnswerType.ANSWERABLE
        answer.tags.append(
            tag_inferred("type", "MPDocVQA only contains answerable questions.")
        )
        answer.tags.append(
            tag_missing("format", "MPDocVQA does not provide answer format.")
        )
        answer.tags.append(
            tag_missing("rationale", "MPDocVQA does not provide rationale for answers.")
        )
        return answer
