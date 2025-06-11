import json
import re
from pathlib import Path

from docrag.schema.entries import (
    Answer,
    AnswerType,
    Document,
    DocumentType,
    Evidence,
    EvidenceSource,
    Question,
    UnifiedEntry,
    ArxivQARawEntry,
)
from docrag.schema.entries.utils import tag_inferred, tag_missing

from ..unifier import Unifier
from ..registry import register
from ..exceptions import UnificationError

__all__ = ["ArxivQAUnifier"]


@register("arxivqa")
class ArxivQAUnifier(Unifier[ArxivQARawEntry]):
    """
    Unifier for the ArxivQA dataset.
    """

    def _discover_raw_qas(self) -> list[Path]:
        # All raw entries are in a single JSONL file under raw_qas_dir
        return sorted(self.raw_qas_directory.glob("*.jsonl"))

    def _load_raw_qas(self, path: Path) -> list[ArxivQARawEntry]:
        # Each line is a JSON object
        raws: list[ArxivQARawEntry] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raws.append(ArxivQARawEntry.model_validate(json.loads(line)))
        return raws

    def _build_entry(
        self,
        raw: ArxivQARawEntry,
        document: Document,
        question: Question,
        evidence: Evidence,
        answer: Answer,
    ) -> UnifiedEntry:
        entry = UnifiedEntry(
            id=raw.id,
            document=document,
            question=question,
            evidence=evidence,
            answer=answer,
        )

        return entry

    def _build_document(self, raw: ArxivQARawEntry) -> Document | None:
        """
        Construct the Document model.
        """
        paper_id, _ = self._extract_paper_page(raw.image)
        count_pages = len(self._corpus_documents.get(paper_id, []))

        if count_pages == 0:
            if self.test_mode:
                self.logger.warning(
                    "Document not found in corpus: paper_id=%s", paper_id
                )
                self._problematic["documents"].append(
                    {
                        "document_id": paper_id,
                        "reason": "Document missing in corpus records.",
                    }
                )
                return None
            else:
                raise UnificationError(
                    f"Document '{paper_id}' not found in corpus."
                    f"for question_id={raw.id}."
                )

        document = Document(id=paper_id, count_pages=count_pages)

        document.type = DocumentType.SCIENTIFIC
        document.tags.append(
            tag_inferred("type", "All ArxivQA figures are from scientific papers.")
        )

        return document

    def _build_question(self, raw: ArxivQARawEntry) -> Question:
        question = Question(id=raw.id, text=raw.question)
        question.tags.append(
            tag_missing("type", "ArxivQA does not provide question type")
        )
        return question

    def _build_evidence(self, raw: ArxivQARawEntry) -> Evidence:
        evidence = Evidence()
        paper_id, page_number = self._extract_paper_page(raw.image)

        if (paper_id, page_number) not in self._corpus_index:
            if self.test_mode:
                evidence.pages = [0]
                self.logger.warning(
                    "Extracted page %s from paper %s not found in corpus.",
                    page_number,
                    paper_id,
                )
                self._problematic["questions"].append(
                    {
                        "question_id": raw.id,
                        "reason": f"Extracted page {page_number} of paper {paper_id} not in corpus.",
                    }
                )
            else:
                raise UnificationError(
                    f"Page {page_number} of paper {paper_id} does not exist in corpus "
                    f"(raw.image={raw.image})."
                )

        evidence.pages = [page_number]
        evidence.sources = [EvidenceSource.IMAGE]
        evidence.tags.append(
            tag_inferred("sources", "All ArxivQA sources are figures.")
        )

        return evidence

    def _build_answer(self, raw: ArxivQARawEntry) -> Answer:
        answer = Answer()

        filtered_options = self._filter_options(raw.options)
        normalized = self._normalize_label(raw.label)
        chosen = self._select_variant_index(filtered_options, normalized)

        if chosen is None:
            if self.test_mode:
                self.logger.warning(
                    "Count not index answer label %s in answer options %s (id=%s)",
                    raw.label,
                    raw.options,
                    raw.id,
                )
                self._problematic["questions"].append(
                    {
                        "question_id": raw.id,
                        "reason": f"Could not index answer label {raw.label} in answer options {raw.options}.",
                    }
                )
                return answer
            else:
                raise UnificationError(
                    f"Could not index answer label {raw.label} in answer options {raw.options} "
                    f"(raw.label={raw.label})"
                    f"(raw.options={raw.options})"
                )

        answer.variants = [self._clean_option(filtered_options[chosen])]

        if not raw.rationale or raw.rationale.strip() == "":
            answer.rationale = ""
            answer.tags.append(
                tag_missing("rationale", "Rationale not provided for this answer.")
            )
        else:
            answer.rationale = raw.rationale

        answer.type = AnswerType.ANSWERABLE
        answer.tags.append(
            tag_missing("format", "ArxivQA does not provide answer format.")
        )

        return answer

    def _extract_paper_page(self, image_path: str) -> tuple[str, int]:
        """
        Given raw.image like "images/2302.14794_1.jpg", return ("2302.14794", 1).
        If parsing fails, return ("", 0).
        """
        try:
            filename = image_path.rsplit("/", 1)[-1]
            no_ext = filename.rsplit(".", 1)[0]
            paper_id, page_str = no_ext.rsplit("_", 1)
            return paper_id, int(page_str)
        except Exception:
            return "", 0

    def _filter_options(self, options: list[str]) -> list[str]:
        """
        Remove any entry that begins with "##" (figure/metadata) but preserve "-" or other valid choices.
        """
        return [opt for opt in (options or []) if not opt.strip().startswith("##")]

    def _normalize_label(self, raw_label: str | None) -> str:
        """
        Strip whitespace and surrounding brackets. E.g. "[A]" -> "A", "A) 0.05" -> "A) 0.05".
        """
        if not raw_label:
            return ""
        lab = raw_label.strip()
        if lab.startswith("[") and lab.endswith("]"):
            lab = lab[1:-1].strip()
        return lab

    def _select_variant_index(self, options: list[str], label: str) -> int | None:
        """
        Given a list of filtered_options and a normalized label, attempt:
         1) prefix-match "X." or "X)" where X = first letter of label
         2) prefix-match the entire label text
         3) fallback letter->index (A->0, B->1, etc.)
        Return the chosen index or None if no match.
        """
        lab = label or ""
        lab_lower = lab.lower().rstrip(")").rstrip(".").strip()

        # Try prefix matching against each option
        if lab_lower:
            first_char = lab_lower[0]
            for i, opt in enumerate(options):
                opt_clean = opt.strip().lower()
                if opt_clean.startswith(f"{first_char}.") or opt_clean.startswith(
                    f"{first_char})"
                ):
                    return i
                if lab_lower and opt_clean.startswith(lab_lower):
                    return i

        # Fallback letter->index
        if lab_lower and lab_lower[0].isalpha():
            idx0 = ord(lab_lower[0].upper()) - ord("A")
            if 0 <= idx0 < len(options):
                return idx0

        # No match found
        return None

    def _clean_option(self, text: str) -> str:
        text = text.strip()
        #   letter   optional space   "." or ")"   optional space   remainder
        m = re.match(r"^[A-Za-z]\s*[.)]\s*(.*)", text)
        if m:
            return m.group(1).strip()
        return text
