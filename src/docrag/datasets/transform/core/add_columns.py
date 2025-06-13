from typing import Any

from datasets import Dataset
from PIL import Image

from ...index import CorpusIndex
from .utils import get_by_key, get_by_key_from_batch

__all__ = [
    "add_images_column",
]


def add_images_column(
    dataset: Dataset,
    corpus_index: CorpusIndex,
    *,
    mode: str,
    document_id_path: str = "document.id",
    evidence_pages_path: str | None = None,
    batched: bool = False,
    batch_size: int = 1000,
    **kwargs,
) -> Dataset:
    """
    Add an 'images' column to each row in `dataset` by looking up pages via `corpus_index`.

    Two modes are supported:
      - "evidence_pages": For each example, fetch only the pages listed under `evidence_pages_path`.
      - "document_pages":  For each example, fetch all pages for the document given by `document_id_path`.

    Args:
        dataset (Dataset):
            The original QA Dataset to augment.
        corpus_index (CorpusIndex):
            An index built over the corpus allowing fast lookup by (doc_id, page_number)
            or by doc_id for all pages.
        mode (str):
            "evidence_pages" to look up only the pages in `evidence_pages_path` for each example.
            "document_pages" to look up all pages for the document. Required.
        document_id_path (str, optional):
            Dotted path in each example where the document ID can be found (e.g. "document.id").
            Defaults to "document.id".
        evidence_pages_path (str | None, optional):
            Dotted path in each example where the list of evidence pages lives (e.g. "evidence.pages").
            Required if mode="evidence_pages", ignored for mode="document_pages".
        batched (bool, optional):
            If True, apply the mapping in batches. Defaults to False.
        batch_size (int, optional):
            Number of examples per batch when batched=True. Defaults to 1000.

    Returns:
        Dataset: A new Dataset identical to `dataset` but with an extra field "images",
                 where each entry is a list of PIL.Image objects.

    Raises:
        ValueError:
            - If mode is not one of {"evidence_pages", "document_pages"}.
            - If evidence_pages_path is None when mode="evidence_pages".
            - If any example is missing the required fields.
            - If 'evidence_pages_path' does not refer to a list of ints in an example.
    """
    if mode not in ("evidence_pages", "document_pages"):
        raise ValueError(
            f"Unsupported mode '{mode}'. Expected 'evidence_pages' or 'document_pages'."
        )
    if mode == "evidence_pages" and evidence_pages_path is None:
        raise ValueError(
            "When mode='evidence_pages', you must provide evidence_pages_path."
        )

    # Convert dotted‐paths into tuple‐keys for nested lookup
    document_id_key: tuple[str, ...] = tuple(document_id_path.split("."))
    evidence_pages_key: tuple[str, ...] | None = (
        tuple(evidence_pages_path.split(".")) if evidence_pages_path else None
    )

    def add_images_one(example: dict[str, Any]) -> dict[str, list[Image.Image]]:
        # 1) Extract document ID
        try:
            doc_id = get_by_key(example, document_id_key)
        except ValueError:
            raise ValueError(
                f"Missing '{document_id_path}' in example when adding images."
            )

        # 2) Fetch images according to mode
        if mode == "evidence_pages":
            assert evidence_pages_key is not None
            try:
                pages = get_by_key(example, evidence_pages_key)
            except ValueError:
                raise ValueError(
                    f"Missing '{evidence_pages_path}' in example when mode='evidence_pages'."
                )
            if not isinstance(pages, list) or not all(
                isinstance(p, int) for p in pages
            ):
                raise ValueError(f"'{evidence_pages_path}' must be a list of ints.")
            images = corpus_index.get_pages(doc_id, pages)
        else:  # mode == "document_pages"
            images = corpus_index.get_document_pages(doc_id)
            if images is None:
                images = []

        return {"images": images}

    def add_images_batch(
        batch: dict[str, list[Any]],
    ) -> dict[str, list[list[Image.Image]]]:
        # Ensure each value in the batch is a list
        first_col = next(iter(batch.values()))
        if not isinstance(first_col, list):
            raise ValueError("Batched mapping expects each batch value to be a list.")
        n = len(first_col)

        images_list: list[list[Image.Image]] = []

        if mode == "document_pages":
            doc_ids: list[str] = []
            for i in range(n):
                try:
                    doc_id = get_by_key_from_batch(batch, document_id_key, i)
                except ValueError:
                    raise ValueError(
                        f"Missing '{document_id_path}' in batch element {i}."
                    )
                doc_ids.append(doc_id)

            batch_docs_to_images = corpus_index.get_batch_document_pages(doc_ids)
            for doc_id in doc_ids:
                images = batch_docs_to_images.get(doc_id, [])
                images_list.append(images)

        else:  # mode == "evidence_pages"
            assert evidence_pages_key is not None
            for i in range(n):
                try:
                    doc_id = get_by_key_from_batch(batch, document_id_key, i)
                except ValueError:
                    raise ValueError(
                        f"Missing '{document_id_path}' in batch element {i}."
                    )

                try:
                    pages = get_by_key_from_batch(batch, evidence_pages_key, i)
                except ValueError:
                    raise ValueError(
                        f"Missing '{evidence_pages_path}' in batch element {i}."
                    )

                if not isinstance(pages, list) or not all(
                    isinstance(p, int) for p in pages
                ):
                    raise ValueError(
                        f"'{evidence_pages_path}' must be a list of ints in batch element {i}."
                    )

                images = corpus_index.get_pages(doc_id, pages)
                images_list.append(images)

        return {"images": images_list}

    if batched:
        return dataset.map(
            add_images_batch,
            batched=True,
            batch_size=batch_size,
            remove_columns=None,  # keep all original columns
            **kwargs,
        )
    else:
        return dataset.map(
            add_images_one,
            remove_columns=None,  # keep all original columns
            **kwargs,
        )
