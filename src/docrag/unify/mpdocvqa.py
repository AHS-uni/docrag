from pathlib import Path
import json

from pydantic import ValidationError

from docrag.schema.raw_entry import MPDocVQARaw
from docrag.unify.base import BaseUnifier
from docrag.schema import (
    UnifiedEntry,
    Question,
    Document,
    Evidence,
    Answer,
    AnswerType,
)
from docrag.schema.utils import tag_missing


class MPDocVQAUnifier(BaseUnifier[MPDocVQARaw]):
    """
    Unifier for the MP-DocVQA competition dataset.
    """

    def _discover_raw_qas(self) -> list[Path]:
        # All of the MPDocVQA files are plain JSON under raw_qas/
        return sorted(self.raw_qas_dir.glob("*.json"))

    def _load_raw_qas(self, path: Path) -> list[MPDocVQARaw]:
        # Entries are in a top level 'data' array
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [MPDocVQARaw.model_validate(item) for item in payload["data"]]

    def _convert_qa_entry(self, raw: MPDocVQARaw) -> UnifiedEntry:
        question = self._build_question(raw)
        document = self._build_document(raw)
        evidence = self._build_evidence(raw)
        answer = self._build_answer(raw)

        entry = UnifiedEntry(
            id=f"{raw.doc_id}-{raw.question_id}",
            question=question,
            document=document,
            evidence=evidence,
            answer=answer,
        )

        if raw.data_split.lower() == "test":
            entry.tags.append(tag_missing("evidence"))
            entry.tags.append(tag_missing("answer"))

        return entry

    def _build_question(self, raw: MPDocVQARaw) -> Question:
        q = Question(id=str(raw.question_id), text=raw.question)
        q.tags.append(tag_missing("type"))
        return q

    def _build_document(self, raw: MPDocVQARaw) -> Document:
        num_pages = sum(1 for d, _, _ in self._corpus_records if d == raw.doc_id)
        doc = Document(id=raw.doc_id, num_pages=num_pages)
        doc.tags.append(tag_missing("type"))
        return doc

    def _build_evidence(self, raw: MPDocVQARaw) -> Evidence:
        ev = Evidence()
        split = raw.data_split.lower()

        if split != "test" and raw.answer_page_idx is not None:
            try:
                page_id_str = raw.page_ids[raw.answer_page_idx]
                page_num = int(page_id_str.rsplit("_p", 1)[1])
            except Exception:
                ev.tags.append(tag_missing("pages"))
            else:
                if (raw.doc_id, page_num) in self._corpus_index:
                    ev.pages = [page_num]
                else:
                    ev.tags.append(tag_missing("pages"))

        ev.tags.append(tag_missing("sources"))
        return ev

    def _build_answer(self, raw: MPDocVQARaw) -> Answer:
        split = raw.data_split.lower()

        if split == "test":
            return Answer()

        ans = Answer(type=AnswerType.ANSWERABLE, variants=raw.answers or [])
        ans.tags.append(tag_missing("format"))
        return ans
