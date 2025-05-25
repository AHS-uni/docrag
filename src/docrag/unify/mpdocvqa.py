from pathlib import Path
import json

from docrag.schema.enums import AnswerFormat, AnswerType
from docrag.schema.raw_entry import MPDocVQARaw
from docrag.unify.base import BaseUnifier
from docrag.schema import (
    UnifiedEntry,
    Question,
    Document,
    Evidence,
    Answer,
)


from pathlib import Path
import json

from docrag.schema.raw_entry import MPDocVQARaw
from docrag.unify.base import BaseUnifier
from docrag.schema import (
    UnifiedEntry,
    Question,
    Document,
    Evidence,
    Answer,
    AnswerFormat,
)


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
        """
        Map a raw MP-DocVQA entry into the unified schema.

        For the test split (competition data), leaves `answer` and `evidence` contain empty values.
        """
        # Build the Question model
        question = Question(
            id=str(raw.question_id),
            text=raw.question,
        )

        # Build the Document model
        document = Document(
            id=raw.doc_id,
            num_pages=len(raw.page_ids),
        )

        # Build the Evidence model
        evidence = Evidence()
        if raw.data_split.lower() != "test" and raw.answer_page_idx is not None:
            evidence = Evidence(pages=[raw.answer_page_idx])

        # Build the Answer model
        answer = Answer()
        if raw.data_split.lower() != "test":
            answer = Answer(
                type=AnswerType.ANSWERABLE,
                variants=raw.answers,  # should always be non-empty in train/val
                format=AnswerFormat.MISSING,
            )

        return UnifiedEntry(
            id=f"{raw.doc_id}-{raw.question_id}",
            question=question,
            document=document,
            evidence=evidence,
            answer=answer,
        )
