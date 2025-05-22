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
)


from pathlib import Path
import json

from docrag.schema.raw_entry import MPDocVQARaw
from docrag.unify.base import BaseUnifier
from docrag.schema import UnifiedEntry, Question, Document, Evidence, Answer


class MPDocVQAUnifier(BaseUnifier[MPDocVQARaw]):
    """
    Unifier for the MP-DocVQA competition dataset.
    """

    def discover_raw(self) -> list[Path]:
        """
        Discover raw MP-DocVQA JSON files.

        Globs all `.json` files under the `raw_qas_dir`.

        Returns:
            List[Path]: Sorted list of paths to raw QA JSON files.
        """
        return sorted(self.raw_qas_dir.glob("*.json"))

    def load_raw(self, path: Path) -> list[MPDocVQARaw]:
        """
        Load and validate raw entries from a JSON file.

        The file must contain a top-level `"data"` array of QA objects.

        Args:
            path (Path): Path to the raw JSON file.

        Returns:
            List[MPDocVQARaw]: Parsed and validated raw entries.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [MPDocVQARaw.model_validate(item) for item in payload["data"]]

    def convert_entry(self, raw: MPDocVQARaw) -> UnifiedEntry:
        """
        Map a raw MP-DocVQA entry into the unified schema.

        For train/val splits, wraps each raw answer list into an `Answer` object.
        For the test split (competition data), leaves `answer` and `evidence` as None.

        Args:
            raw (MPDocVQARaw): A single raw MP-DocVQA example.

        Returns:
            UnifiedEntry: The normalized entry ready for downstream use.
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
        evidence = None
        if raw.data_split.lower() != "test" and raw.answer_page_idx is not None:
            evidence = Evidence(pages=[raw.answer_page_idx])

        # Build the Answer model
        answer = None
        if raw.data_split.lower() != "test":
            answer = Answer(
                answerable=True,
                variants=raw.answers,  # should always be non-empty in train/val
            )

        return UnifiedEntry(
            id=f"{raw.doc_id}-{raw.question_id}",
            question=question,
            document=document,
            evidence=evidence,
            answer=answer,
        )
