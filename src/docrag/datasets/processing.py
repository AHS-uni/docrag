from collections import defaultdict
import logging
from pathlib import Path
from typing import Any

from PIL import Image
from datasets import Dataset, Features, Sequence, ClassLabel

from docrag.utils import get_logger

__all__ = [
    "CorpusIndex",
    "project_fields",
    "filter_dataset",
]


class CorpusIndex:
    """
    Index for fast lookup of images in a corpus Dataset by document ID and page number.

    All methods now return Image objects (never full rows), or an empty list/None
    if no images are found.
    """

    def __init__(self, dataset: Dataset) -> None:
        """
        Initialize the index and build lookup tables.

        Args:
            dataset (Dataset): A Hugging Face Dataset containing at least these columns:
                - "doc_id" (str): Document identifier
                - "page_number" (int): Page number
                - "image" (PIL.Image.Image): The image on that page
        """
        self.dataset = dataset
        # Map (document_id, page_number) → dataset index
        self.document_page_index: dict[tuple[str, int], int] = {}
        # Map document_id → list of dataset indices for that document
        self.document_to_indices: dict[str, list[int]] = defaultdict(list)

        self.logger = get_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG,
            log_file_path=Path("logs/corpus_index.log"),
        )

        for idx, example in enumerate(dataset):
            doc_id = example["doc_id"]
            page_number = example["page_number"]
            self.document_page_index[(doc_id, page_number)] = idx
            self.document_to_indices[doc_id].append(idx)

    def get_page(self, document_id: str, page_number: int) -> Image.Image | None:
        """
        Retrieve a single image by (document_id, page_number).

        Args:
            document_id (str): Document identifier.
            page_number (int): Page number within the document.

        Returns:
            PIL.Image.Image | None: The image if found; otherwise None.
        """
        idx = self.document_page_index.get((document_id, page_number))
        if idx is None:
            self.logger.warning(
                f"get_page: Missing page {page_number} for document '{document_id}'."
            )
            return None

        row = self.dataset[idx]
        return row["image"]

    def get_pages(
        self, document_id: str, page_numbers: list[int]
    ) -> list[Image.Image | None]:
        """
        Retrieve a list of images for a document by looping over get_page().

        Args:
            document_id (str): Document identifier.
            page_numbers (list[int]): Page numbers to fetch.

        Returns:
            list[PIL.Image.Image | None]: One entry per page_number in order.
                If a page is missing, its entry is None.
        """
        results: list[Image.Image | None] = []
        for page_number in page_numbers:
            image = self.get_page(document_id, page_number)
            results.append(image)
        return results

    def get_document_pages(self, document_id: str) -> list[Image.Image]:
        """
        Retrieve all images for a document, sorted by page_number.

        Args:
            document_id (str): Document identifier.

        Returns:
            list[PIL.Image.Image]: List of images sorted by page_number.
                Returns an empty list if no pages are found.
        """
        indices = self.document_to_indices.get(document_id)
        if not indices:
            self.logger.warning(
                f"get_document_pages: No pages found for document '{document_id}'."
            )
            return []

        # Use HF Dataset sorting to ensure proper order
        subset = self.dataset.select(indices)
        sorted_subset = subset.sort("page_number")

        images: list[Image.Image] = [ex["image"] for ex in sorted_subset]
        if not images:
            self.logger.warning(
                f"get_document_pages: Document '{document_id}' had no images after sorting."
            )
        return images

    def get_batch_document_pages(
        self, document_ids: list[str]
    ) -> dict[str, list[Image.Image]]:
        """
        Retrieve images for multiple documents in one call by looping over get_document_pages().

        Args:
            document_ids (list[str]): List of document identifiers.

        Returns:
            dict[str, list[PIL.Image.Image]]: Mapping from each document_id to its list of images.
                If a document_id has no pages, its value is an empty list.
        """
        batch_result: dict[str, list[Image.Image]] = {}
        for document_id in document_ids:
            images = self.get_document_pages(document_id)
            batch_result[document_id] = images
        return batch_result


def _get_by_key(obj: dict, key: tuple) -> Any:
    """
    Retrieve a value from a nested dictionary using a sequence of keys.

    Args:
        obj (dict): The dictionary (or nested dictionaries) to traverse.
        key (tuple): A tuple of strings representing the nested path of keys.
            Each element in the tuple is a key to look up at that level.

    Returns:
        Any: The value found at the nested path.

    Raises:
        ValueError: If any key in the path is missing or if a non-dict type is encountered
            before reaching the final key.
    """
    for k in key:
        try:
            obj = obj[k]
        except (KeyError, TypeError):
            raise ValueError(f"Key path {'.'.join(key)} not found.")
    return obj


def _get_by_key_from_batch(batch: dict, key: tuple, idx: int) -> Any:
    """
    Retrieve a nested value from a batched dictionary at a specific row index.

    Args:
        batch (dict): A mapping from column names to lists of values. Each value
            in the list corresponds to a row in a batch of examples.
        key (tuple): A tuple of strings representing the nested path of keys.
            The first element refers to a top-level column in `batch`, and each
            subsequent element refers to a nested key within that row.
        idx (int): The row index (0-based) within each list in `batch`.

    Returns:
        Any: The value found at the nested path for the specified row index.

    Raises:
        ValueError: If the top-level key is missing in `batch` or if `idx` is out of range,
            or if any nested key in the path is missing or if a non-dict type is encountered
            before reaching the final key.
    """
    top_key = key[0]
    try:
        row_value = batch[top_key][idx]
    except (KeyError, IndexError):
        raise ValueError(f"Missing '{top_key}' in batch at index {idx}.")

    if len(key) == 1:
        return row_value

    current = row_value
    for subkey in key[1:]:
        try:
            current = current[subkey]
        except (KeyError, TypeError):
            raise ValueError(f"Key path '{'.'.join(key)}' not found at index {idx}.")
    return current


def _build_features(
    original_features: Features,
    select_map: dict[str, tuple[str, ...]],
) -> Features:
    """
    Build a new Features spec from a mapping of column_name → field_key tuple.

    Args:
        original_features (Features):
            The Features object from the original Dataset.
        select_map (dict[str, tuple[str, ...]]):
            A mapping where each key is the desired output column name (no dots allowed),
            and each value is the field_key tuple indicating which original feature to use.

    Returns:
        Features: A Features object whose keys are the dict’s keys (column names),
                  and whose values are the corresponding feature types.

    Raises:
        ValueError:
            - If any output column name contains a dot.
            - If any field_key tuple is not found in `original_features`.
    """
    features_spec: dict[str, Any] = {}

    for column_name, field_key in select_map.items():
        if "." in column_name:
            raise ValueError(
                f"Invalid output column name '{column_name}': may not contain '.'"
            )
        # Look up the nested feature type via get_feature_type (which uses _get_by_key)
        feature_type = _get_by_key(original_features, field_key)
        features_spec[column_name] = feature_type

    return Features(features_spec)


def project_fields(
    dataset: Dataset,
    select_fields: list[str] | dict[str, str],
    *,
    batched: bool = False,
    batch_size: int = 1000,
    **kwargs,
) -> Dataset:
    """
    Project nested fields into top‐level columns.
    Accepts either a dict mapping new names → dotted paths, or a list of dotted paths.

    NOTE: Any deeper nested lists beyond the first “list of dicts” level
    (e.g., trying to drill into a tag’s elements like "question.tags.name")
    are not supported.

    Args:
        dataset (Dataset):
            The original Hugging Face Dataset to project fields from.
        select_fields (dict[str, str] | list[str]):
            - If a dict: keys are the NEW column-names (must NOT contain dots),
              values are the dotted-paths of fields to extract (e.g. "question.text").
            - If a list: each item is a dotted-path (e.g. "question.text").
              In that case, output column-names will be the dotted path with dots replaced by underscores
              (e.g. "question.text" → "question_text").
        batched (bool, optional):
            If True, batch_size controls how many examples are passed at once.
            Defaults to False.
        batch_size (int, optional):
            Number of examples per batch when batched=True. Defaults to 1000.

    Returns:
        Dataset:
            A new Dataset containing only the projected columns (no original columns remain).
            Its `.features` reflect the original feature types (ClassLabel, Sequence, etc.) but keyed
            by the new (non-dotted) column-names.

    Raises:
        ValueError:
            - If any dotted-path does not exist in `dataset.features`.
            - If, in the dict case, a new column-name contains a dot.
            - If batched=True but a batch comes in with non-list columns.
    """
    select_map: dict[str, tuple[str, ...]] = {}
    if isinstance(select_fields, dict):
        for new_column_name, field_path in select_fields.items():
            if "." in new_column_name:
                raise ValueError(
                    f"Invalid new column name '{new_column_name}': column names may not contain '.'"
                )
            field_key = tuple(field_path.split("."))
            select_map[new_column_name] = field_key
    else:
        for field_path in select_fields:
            new_column_name = field_path.replace(".", "_")
            field_key = tuple(field_path.split("."))
            select_map[new_column_name] = field_key

    projected_features = _build_features(dataset.features, select_map)

    # Define projection functions for unbatched and batched modes
    def project_one(example: dict) -> dict[str, Any]:
        output: dict[str, object] = {}
        for new_column_name, field_key in select_map.items():
            output[new_column_name] = _get_by_key(example, field_key)
        return output

    def project_batch(batch: dict) -> dict[str, Any]:
        first_col = next(iter(batch.values()))
        if not isinstance(first_col, list):
            raise ValueError("Batched mapping expects each batch value to be a list.")
        n = len(first_col)

        output = {new_column_name: [] for new_column_name in select_map.keys()}
        for i in range(n):
            for new_column_name, field_key in select_map.items():
                try:
                    val = _get_by_key_from_batch(batch, field_key, i)
                except ValueError as e:
                    raise ValueError(
                        f"Error extracting '{'.'.join(field_key)}' at batch index {i}: {e}"
                    )
                output[new_column_name].append(val)

        return output

    return dataset.map(
        project_batch if batched else project_one,
        batched=batched,
        batch_size=batch_size if batched else None,
        remove_columns=dataset.column_names,
        features=projected_features,
        **kwargs,
    )


def filter_dataset(
    dataset: Dataset,
    *,
    field_filters: dict[str, Any] | None = None,
    tag_filters: list[dict[str, str]] | None = None,
    batched: bool = False,
    batch_size: int = 1000,
    **kwargs,
) -> Dataset:
    """
    Filter a Dataset by simple field equality and/or tag list membership.
    Supports both non‐batched and batched modes.

    Args:
        dataset (Dataset):
           Original Hugging Face Dataset.
        field_filters (dict[str, Any], optional):
           Mapping from dotted field paths to desired values. Example:
           {"answer.type": "not_answerable", "question.type": 3}.
        tag_filters (list[dict[str, Any]], optional):
           Each dict must have keys:
             - "tags_list_path": dotted path to the tags list (e.g. "answer.tags")
             - "name": desired tag name (string or integer index)
             - "target": desired tag target
        batched (bool, optional):
            If True, apply the filter in batches. Defaults to False.
        batch_size (int, optional):
            Number of examples per batch when batched=True. Defaults to 1000.

    Returns:
        Dataset: a new Dataset containing only rows that satisfy all
                 field_filters and all tag_filters. If either is None or empty,
                 that criterion is skipped.

    Raises:
        ValueError: if any dotted path does not exist in dataset.features, or
                    if the tag name provided in a tag filter is invalid.
    """

    compiled_field_filters: dict[tuple[str, ...], Any] = {}
    if field_filters:
        for field_path, raw_value in field_filters.items():
            field_key = tuple(field_path.split("."))

            try:
                field_feature = _get_by_key(dataset.features, field_key)
            except ValueError as e:
                raise ValueError(f"Field filter path '{field_path}' not found.") from e

            if isinstance(field_feature, ClassLabel):
                try:
                    expected_value = field_feature.encode_example(raw_value)
                except Exception as e:
                    raise ValueError(
                        f"Invalid class label '{raw_value}' for feature '{field_path}': {e}"
                    )
            else:
                expected_value = raw_value

            compiled_field_filters[field_key] = expected_value

    compiled_tag_filters: list[dict[str, Any]] = []
    if tag_filters:
        for tag_filter in tag_filters:
            tags_list_path = tag_filter["tags_list_path"]
            raw_name = tag_filter["name"]
            expected_target = tag_filter["target"]

            tags_list_key = tuple(tags_list_path.split("."))

            try:
                tags_list_feature = _get_by_key(dataset.features, tags_list_key)
            except ValueError as e:
                raise ValueError(f"Tags list path '{tags_list_path}' not found.") from e

            # Determine the inner feature dict:
            if isinstance(tags_list_feature, Sequence):
                tag_feature = tags_list_feature.feature
            else:
                tag_feature = tags_list_feature[0]

            name_feature = tag_feature["name"]
            if isinstance(name_feature, ClassLabel):
                try:
                    expected_name = name_feature.encode_example(raw_name)
                except Exception as e:
                    raise ValueError(
                        f"Invalid tag name '{raw_name}' for '{tags_list_path}.name': {e}"
                    )
            else:
                expected_name = raw_name

            compiled_tag_filters.append(
                {
                    "tags_list_key": tags_list_key,
                    "name": expected_name,
                    "target": expected_target,
                }
            )

    def filter_one(example: dict) -> bool:
        # Field-based checks
        for field_key, expected_value in compiled_field_filters.items():
            try:
                val = _get_by_key(example, field_key)
            except ValueError:
                return False
            if val != expected_value:
                return False

        # Tag-based checks
        for tag_filter in compiled_tag_filters:
            try:
                tags_list = _get_by_key(example, tag_filter["tags_list_key"])
            except ValueError:
                return False
            if not isinstance(tags_list, list):
                return False

            found = False
            for tag in tags_list:
                if (
                    tag.get("name") == tag_filter["name"]
                    and tag.get("target") == tag_filter["target"]
                ):
                    found = True
                    break
            if not found:
                return False

        return True

    def filter_batch(batch: dict) -> list[bool]:
        first_col = next(iter(batch.values()))
        if not isinstance(first_col, list):
            raise ValueError("Batched filter expects each batch value to be a list.")
        n = len(first_col)

        mask: list[bool] = []
        for i in range(n):
            # 1) Field‐based checks
            field_pass = True
            for field_key, expected_value in compiled_field_filters.items():
                try:
                    val = _get_by_key_from_batch(batch, field_key, i)
                except ValueError:
                    field_pass = False
                    break
                if val != expected_value:
                    field_pass = False
                    break

            if not field_pass:
                mask.append(False)
                continue

            # 2) Tag‐based checks
            tag_pass = True
            for filt in compiled_tag_filters:
                try:
                    tags_list = _get_by_key_from_batch(batch, filt["tags_list_key"], i)
                except ValueError:
                    tag_pass = False
                    break

                if not isinstance(tags_list, list):
                    tag_pass = False
                    break

                found = False
                for tag in tags_list:
                    if (
                        tag.get("name") == filt["name"]
                        and tag.get("target") == filt["target"]
                    ):
                        found = True
                        break
                if not found:
                    tag_pass = False
                    break

            mask.append(tag_pass)

        return mask

    if batched:
        return dataset.filter(
            filter_batch, batched=True, batch_size=batch_size, **kwargs
        )
    return dataset.filter(filter_one, **kwargs)


def add_images(
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
            doc_id = _get_by_key(example, document_id_key)
        except ValueError:
            raise ValueError(
                f"Missing '{document_id_path}' in example when adding images."
            )

        # 2) Fetch images according to mode
        if mode == "evidence_pages":
            assert evidence_pages_key is not None
            try:
                pages = _get_by_key(example, evidence_pages_key)
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
                    doc_id = _get_by_key_from_batch(batch, document_id_key, i)
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
                    doc_id = _get_by_key_from_batch(batch, document_id_key, i)
                except ValueError:
                    raise ValueError(
                        f"Missing '{document_id_path}' in batch element {i}."
                    )

                try:
                    pages = _get_by_key_from_batch(batch, evidence_pages_key, i)
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
