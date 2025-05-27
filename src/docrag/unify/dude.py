from pathlib import Path
import json
from collections import OrderedDict

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

    @staticmethod
    def _unify_variants(
        answers: list | None,
        answers_variants: list | None,
        *,
        sentinel: str = "<item>",
    ) -> tuple[list[str], AnswerFormat]:
        """
        Normalize answer candidates into a joined strings and assign an answer format.

        Args:
            answers (list | None): The primary list of answer candidates from the
                raw example.  Elements may be strings or lists of strings.
            answers_variants (list | None): Additional acceptable variants.  Same
                structure as ``answers``.
            sentinel (str): String/token used to join elements of list-type
                candidates.  Defaults to `<item>`.  Choose a token unlikely to
                occur inside a legitimate answer span.

        Returns:
            tuple[list[str], AnswerFormat]:
                - variants – list of unique answer strings
                - fmt – `AnswerFormat.LIST` if any candidate was itself a
                  list, otherwise `AnswerFormat.MISSING`.

        """
        candidates = []
        saw_list = False

        for group in (answers, answers_variants):
            if not group:
                continue

            if len(group) == 1:
                candidate = str(group[0])
            else:
                saw_list = True
                candidate = " ".join(f"{sentinel} {item}" for item in group)

            candidates.append(candidate)

        # de-duplicate exact duplicates, preserve order
        variants = list(OrderedDict.fromkeys(candidates))
        fmt = AnswerFormat.LIST if saw_list else AnswerFormat.MISSING
        return variants, fmt

    def _convert_qa_entry(self, raw: DUDERaw) -> UnifiedEntry | None:
        """
        Map a raw DUDE entry into the unified schema.

        For the test split (competition data), leaves `answer` and `evidence` as defaults.
        """
        split = raw.data_split.lower()
        answer_type_raw = (raw.answer_type or "").lower()

        # Build the Question model
        q_type_map = {
            "extractive": QuestionType.EXTRACTIVE,
            "list/extractive": QuestionType.EXTRACTIVE,
            "abstractive": QuestionType.ABSTRACTIVE,
            "list/abstractive": QuestionType.ABSTRACTIVE,
        }
        question = Question(
            id=raw.question_id,
            text=raw.question,
            type=q_type_map.get(answer_type_raw, QuestionType.MISSING),
        )

        # Build the Document model
        # Use corpus records to count pages
        page_numbers = [p for (doc, p, _) in self._corpus_records if doc == raw.doc_id]
        num_pages = len(page_numbers)
        document = Document(
            id=raw.doc_id,
            num_pages=num_pages,
        )

        # Build the Evidence model
        evidence = Evidence()
        if split != "test" and raw.answers_page_bounding_boxes:
            pages = {
                box.page for group in raw.answers_page_bounding_boxes for box in group
            }
            evidence = Evidence(pages=sorted(pages))

        # Build the Answer model
        if split == "test":
            answer = Answer()  # NONE
        elif answer_type_raw == "not-answerable":
            answer = Answer(type=AnswerType.NOT_ANSWERABLE)
            evidence = Evidence()  # clear pages/sources
        else:
            variants, fmt = self._unify_variants(
                raw.answers,
                raw.answers_variants,
            )

            answer = Answer(
                type=AnswerType.ANSWERABLE,
                variants=variants,
                format=fmt,
            )

        if evidence.pages == [] and answer.type == AnswerType.ANSWERABLE:
            evidence = Evidence(pages=page_numbers)
            if evidence.pages == []:  # still no evidence
                self.logger.debug(
                    "Skipping entry %s. Was unable to find evidence pages. Most likely a problem with the document %s.",
                    raw.question_id,
                    raw.doc_id,
                )
                return None  # something is broken internally with the document skip this entry

        return UnifiedEntry(
            id=raw.question_id,
            question=question,
            document=document,
            evidence=evidence,
            answer=answer,
        )
