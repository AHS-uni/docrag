import json
import re
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
    SlideVQARawEntry,
)
from docrag.schema.entries.utils import tag_inferred, tag_missing, tag_low_quality

from ..unifier import Unifier
from ..registry import register
from ..exceptions import UnificationError

__all__ = ["SlideVQAUnifier"]


_NUM_PATTERN = re.compile(r"^\d[\d,]*\.?\d*$")


@register("slidevqa")
class SlideVQAUnifier(Unifier[SlideVQARawEntry]):
    def _discover_raw_qas(self) -> list[Path]:
        # All raw splits are JSONL files under raw_qas/
        return sorted(self.raw_qas_directory.glob("*.jsonl"))

    def _load_raw_qas(self, path: Path) -> list[SlideVQARawEntry]:
        raws: list[SlideVQARawEntry] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raws.append(SlideVQARawEntry.model_validate(json.loads(line)))
        return raws

    def _build_entry(
        self,
        raw: SlideVQARawEntry,
        document: Document,
        question: Question,
        evidence: Evidence,
        answer: Answer,
    ) -> UnifiedEntry:
        entry = UnifiedEntry(
            id=f"{raw.deck_name}-{raw.qa_id}",
            document=document,
            question=question,
            evidence=evidence,
            answer=answer,
        )
        return entry

    def _build_document(self, raw: SlideVQARawEntry) -> Document | None:
        count_pages = len(self._corpus_documents.get(raw.deck_name, []))

        if count_pages == 0:
            if self.test_mode:
                self.logger.warning(
                    "Document not found in corpus: deck_name=%s", raw.deck_name
                )
                self._problematic["documents"].append(
                    {"document_id": raw.deck_name, "reason": "Missing in corpus."}
                )
                return None
            else:
                raise UnificationError(
                    f"Document '{raw.deck_name}' not found in corpus."
                )

        document = Document(id=raw.deck_name, count_pages=count_pages)
        document.tags.append(
            tag_missing("type", "SlideVQA does not provide document type.")
        )
        return document

    def _build_question(self, raw: SlideVQARawEntry) -> Question:
        question = Question(id=str(raw.qa_id), text=raw.question)

        if raw.answer_type or raw.resoning_type:
            question.type = self._determine_question_type(raw)
            question.tags.append(
                tag_inferred("type", "Inferred using answer_type and resoning_type.")
            )
            return question

        question.tags.append(
            tag_missing(
                "type",
                "SlideVQA does not provide question type information for non-test splits",
            )
        )
        return question

    def _build_evidence(self, raw: SlideVQARawEntry) -> Evidence:
        evidence = Evidence()

        pages = sorted({p - 1 for p in raw.evidence_pages or [] if isinstance(p, int)})
        document_pages = self._corpus_documents.get(raw.deck_name, [])
        missing = [p for p in pages if p not in document_pages]
        if missing:
            if self.test_mode:
                valid = [p for p in pages if p in document_pages]
                evidence.pages = valid or [0]
                self.logger.warning(
                    "Provided pages %s not in corpus for deck=%s. Using %s instead.",
                    missing,
                    raw.deck_name,
                    evidence.pages,
                )
                self._problematic["questions"].append(
                    {
                        "question_id": raw.qa_id,
                        "reason": f"Pages {missing} missing in corpus.",
                    }
                )
            else:
                raise UnificationError(
                    f"Pages {missing} not found in corpus for deck '{raw.deck_name}'."
                )

        evidence.pages = pages
        if raw.answer_type:
            sources = self._map_evidence_source(raw.answer_type)
            if sources:
                evidence.sources = sources
                evidence.tags.append(
                    tag_low_quality(
                        "sources",
                        "Evidence source information is limited in SlideVQA. Only provides distinction between span and non-span evidence source.",
                    )
                )
            else:
                evidence.tags.append(
                    tag_missing(
                        "sources",
                        "Evidence source information not provided for this question.",
                    )
                )
            return evidence

        evidence.tags.append(
            tag_missing(
                "sources",
                "SlideVQA does not provide evidence source information for non-test questions.",
            )
        )

        return evidence

    def _build_answer(self, raw: SlideVQARawEntry) -> Answer:
        answer = Answer()

        is_list, canonical = self._to_list_answer(raw.answer.strip())
        answer.variants = [canonical]
        answer.format = AnswerFormat.LIST if is_list else AnswerFormat.OTHER

        # rationale
        if raw.arithmetic_expression and raw.arithmetic_expression.lower() != "none":
            answer.rationale = raw.arithmetic_expression
        else:
            answer.tags.append(
                tag_missing("rationale", "No rationale provided for this answer.")
            )

        answer.type = AnswerType.ANSWERABLE
        return answer

    def _determine_question_type(self, raw: SlideVQARawEntry) -> QuestionType:
        answer_type = (raw.answer_type or "").lower()
        reasoning = (raw.resoning_type or "").lower()
        discrete = "discrete" in reasoning
        multi_hop = "multi-hop" in reasoning
        has_expr = bool(
            raw.arithmetic_expression and raw.arithmetic_expression.lower() != "none"
        )

        if answer_type.startswith(("single-span", "multi-span")) and not discrete:
            return QuestionType.EXTRACTIVE
        if discrete and has_expr:
            return QuestionType.ARITHMETIC
        if discrete:
            return QuestionType.COUNTING
        if multi_hop:
            return QuestionType.REASONING
        if answer_type == "non-span":
            return QuestionType.ABSTRACTIVE
        return QuestionType.OTHER

    def _has_discrete(self, resoning_type: str | None) -> bool:
        return bool(resoning_type and "discrete" in resoning_type.lower())

    def _is_multi_hop(self, resoning_type: str | None) -> bool:
        return bool(resoning_type and "multi-hop" in resoning_type.lower())

    def _map_evidence_source(self, answer_type: str | None) -> list[EvidenceSource]:
        if answer_type and answer_type.lower().startswith(
            ("single-span", "multi-span")
        ):
            return [EvidenceSource.SPAN]
        return []

    def _to_list_answer(self, text: str) -> tuple[bool, str]:
        """
        Detect whether `text` encodes a list of multiple items.

        Returns:
          (is_list, canonical)

        - If is_list=True, canonical is the Python-style list string "['a', 'b', ...]".
        - Otherwise canonical is the original text (with thousands-commas removed if numeric).
        """
        parts = self._split_answer_candidates(text)
        # not enough parts → not a list
        if len(parts) <= 1:
            return False, text

        # two numeric parts like ["2,105", "3,210"] are probably thousands-separators,
        # so treat as single number string "2105,3210"? Better to remove commas.
        if len(parts) == 2 and all(self._is_number_like(p) for p in parts):
            # remove commas for consistency
            return False, text.replace(",", "")

        # Real list: quote each item and emit Python-style list literal
        quoted = ", ".join(f"'{p}'" for p in parts)
        return True, f"[{quoted}]"

    def _is_number_like(self, token: str) -> bool:
        """
        Return True if `token` is purely numeric (possibly with commas or a single dot).
        """
        return bool(_NUM_PATTERN.fullmatch(token.strip()))

    def _split_answer_candidates(self, text: str) -> list[str]:
        """
        Split on commas or semicolons, trim whitespace, drop empty parts.
        E.g. "A, B; C" → ["A", "B", "C"].
        """
        parts = re.split(r"[;,]", text)
        return [p.strip() for p in parts if p.strip()]
