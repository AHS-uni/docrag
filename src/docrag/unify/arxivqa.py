from pathlib import Path
import json
from typing import Optional, Tuple

from docrag.schema.enums import (
    QuestionType,
    DocumentType,
    EvidenceSource,
    AnswerType,
    AnswerFormat,
)
from docrag.schema.raw_entry import ArxivQARaw
from docrag.unify.base import BaseUnifier
from docrag.schema import UnifiedEntry, Question, Document, Evidence, Answer
from docrag.schema.utils import tag_missing, tag_inferred, tag_low_quality


class ArxivQAUnifier(BaseUnifier[ArxivQARaw]):
    """
    Unifier for the ArxivQA dataset.
    """

    def _discover_raw_qas(self) -> list[Path]:
        # All raw entries are in a single JSONL file under raw_qas_dir
        return sorted(self.raw_qas_dir.glob("*.jsonl"))

    def _load_raw_qas(self, path: Path) -> list[ArxivQARaw]:
        # Each line is a JSON object
        raws: list[ArxivQARaw] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raws.append(ArxivQARaw.model_validate(json.loads(line)))
        return raws

    def _convert_qa_entry(self, raw: ArxivQARaw) -> UnifiedEntry:
        question = self._build_question(raw)
        document = self._build_document(raw)
        evidence = self._build_evidence(raw)
        answer = self._build_answer(raw)

        entry = UnifiedEntry(
            id=raw.id,
            question=question,
            document=document,
            evidence=evidence,
            answer=answer,
        )

        return entry

    def _build_question(self, raw: ArxivQARaw) -> Question:
        """
        Construct the Question model.
        """
        q = Question(id=raw.id, text=raw.question)
        q.type = QuestionType.OTHER
        q.tags.append(tag_missing("type"))
        return q

    def _build_document(self, raw: ArxivQARaw) -> Document:
        """
        Construct the Document model.
        """
        paper_id, _ = self._extract_paper_page(raw.image)
        num_pages = sum(1 for (d, _, _) in self._corpus_records if d == paper_id)
        doc = Document(id=paper_id, num_pages=num_pages)

        # Infer document type as scientific
        doc.type = DocumentType.SCIENTIFIC
        doc.tags.append(tag_inferred("type"))

        if num_pages == 0:
            doc.tags.append(tag_missing("num_pages"))

        return doc

    def _build_evidence(self, raw: ArxivQARaw) -> Evidence:
        """
        Construct the Evidence model.
        """
        ev = Evidence()
        paper_id, page_num = self._extract_paper_page(raw.image)

        if (paper_id, page_num) in self._corpus_index:
            ev.pages = [page_num]
        else:
            ev.pages = [0]
            ev.tags.append(tag_missing("pages"))

        ev.sources = [EvidenceSource.IMAGE]
        ev.tags.append(tag_inferred("sources"))

        return ev

    def _build_answer(self, raw: ArxivQARaw) -> Answer:
        """
        Construct the Answer model.
        """
        ans = Answer()

        # Filter out any "## ..." entries from options
        filtered_options = self._filter_options(raw.options)

        # Normalize the label
        lab = self._normalize_label(raw.label)

        # Attempt to match label → index in filtered_options
        chosen_idx = self._select_variant_index(filtered_options, lab)

        # If no match, tag missing variants
        if chosen_idx is None:
            ans.variants = []
            ans.tags.append(tag_missing("variants"))
        else:
            ans.variants = [filtered_options[chosen_idx]]

        # Rationale
        if not raw.rationale or raw.rationale.strip() == "":
            ans.rationale = ""
            ans.tags.append(tag_missing("rationale"))
        else:
            ans.rationale = raw.rationale

        ans.tags.append(tag_missing("format"))

        ans.type = AnswerType.ANSWERABLE

        return ans

    def _extract_paper_page(self, image_path: str) -> Tuple[str, int]:
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

    def _normalize_label(self, raw_label: Optional[str]) -> str:
        """
        Strip whitespace and surrounding brackets. E.g. "[A]" -> "A", "A) 0.05" -> "A) 0.05".
        """
        if not raw_label:
            return ""
        lab = raw_label.strip()
        if lab.startswith("[") and lab.endswith("]"):
            lab = lab[1:-1].strip()
        return lab

    def _select_variant_index(self, options: list[str], label: str) -> Optional[int]:
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
                if opt_clean.startswith(f"{first_char}.") or opt_clean.startswith(f"{first_char})"):
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
