from collections import defaultdict

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
    Index for fast lookup of rows or images in the corpus dataset
    using (doc_id, page_number) or just doc_id.

    """

    def __init__(self, dataset: Dataset):
        """
        Initializes the index and builds internal lookup tables.

        Args:
            dataset (Dataset): A Hugging Face dataset with fields 'doc_id', 'page_number', and 'image'.
        """
        self.dataset = dataset
        self.doc_page_to_idx = {}
        self.doc_to_indices = defaultdict(list)

        self.logger = get_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG,
            log_file_path=Path("logs/corpus_index.log"),
        )

        for i, ex in enumerate(dataset):
            doc_id = ex["doc_id"]
            page_number = ex["page_number"]

            self.doc_page_to_idx[(doc_id, page_number)] = i
            self.doc_to_indices[doc_id].append(i)

    def get_page(
        self, doc_id: str, page_number: int, return_images: bool = True
    ) -> dict | Image.Image | None:
        """
        Retrieve a single page row or image by (doc_id, page_number).

        Args:
            doc_id (str): Document identifier.
            page_number (int): Page number within the document.
            return_images (bool, optional): If True, return the image object only.
                                            If False, return the full row. Defaults to True.

        Returns:
            Image.Image | dict | None: The image, the full row, or None if not found.
        """
        idx = self.doc_page_to_idx.get((doc_id, page_number))
        if idx is None:
            self.logger.warning(
                f"get_page: Missing page {page_number} for document '{doc_id}'."
            )
            return None

        row = self.dataset[idx]
        return row["image"] if return_images else row

    def get_pages(
        self, doc_id: str, page_numbers: list[int], return_images: bool = True
    ) -> list[Image.Image | dict | None]:
        """
        Retrieve a list of pages/images for a document in the same order
        as `page_numbers`. If a specific (doc_id, page_number) pair is not found,
        the corresponding position in the returned list will be None.

        Args:
            doc_id (str): Document identifier.
            page_numbers (list[int]): List of page numbers to fetch.
            return_images (bool, optional): If True, return image objects only.
                                            If False, return full rows. Defaults to True.

        Returns:
            list[Image.Image | dict | None]: One entry per page_number in order.
        """
        results: list[Union[Image.Image, dict, None]] = []
        for page_num in page_numbers:
            index = self.doc_page_to_index.get((doc_id, page_num))
            if index is None:
                self.logger.warning(
                    f"get_pages: Missing page {page_num} for document '{doc_id}'."
                )
                results.append(None)
            else:
                row = self.dataset[index]
                results.append(row["image"] if return_images else row)
        return results

    def get_document_pages(
        self, doc_id: str, return_images: bool = True
    ) -> list[Image.Image] | Dataset | None:
        """
        Retrieve all pages/images for a document, sorted by page_number.

        Args:
            doc_id (str): Document identifier.
            return_images (bool, optional): If True, return a list of image objects.
                                            If False, return a HF Dataset of rows. Defaults to True.

        Returns:
            list[Image.Image] | Dataset | None: List of images, subset Dataset,
                                               or None if not found.
        """
        indices = self.doc_to_indices.get(doc_id)
        if not indices:
            self.logger.warning(
                f"get_document_pages: No pages found for document '{doc_id}'."
            )
            return None

        # Load subset and sort by page_number
        subset = self.dataset.select(indices)
        sorted_subset = sorted(subset, key=lambda ex: ex["page_number"])

        if return_images:
            images = [row["image"] for row in sorted_subset]
            if not images:
                self.logger.warning(
                    f"get_document_pages: Document '{doc_id}' had no images after sorting."
                )
            return images
        else:
            rows = Dataset.from_list(sorted_subset)
            if len(rows) == 0:
                self.logger.warning(
                    f"get_document_pages: Document '{doc_id}' had no rows after sorting."
                )
            return rows

    def get_batch_document_pages(
        self, doc_ids: list[str], return_images: bool = True
    ) -> dict[str, list]:
        """
        Retrieve pages/images for multiple documents in one call.

        Args:
            doc_ids (list[str]): List of document identifiers.
            return_images (bool, optional): If True, return lists of image objects.
                                            If False, return lists of full rows. Defaults to True.

        Returns:
            dict[str, list]: A mapping from each doc_id to a list of its pages (images or rows),
                             sorted by page_number. If a doc_id is not found, its value is an empty list.
        """
        batch_result: dict[str, list] = {}

        for doc_id in doc_ids:
            indices = self.doc_to_indices.get(doc_id)
            if not indices:
                self.logger.warning(
                    f"get_batch_document_pages: No pages found for document '{doc_id}'."
                )
                batch_result[doc_id] = []
                continue

            # Select all rows for this document, then sort by page_number
            subset = self.dataset.select(indices)
            sorted_subset = sorted(subset, key=lambda ex: ex["page_number"])

            if return_images:
                images = [row["image"] for row in sorted_subset]
                if not images:
                    self.logger.warning(
                        f"get_batch_document_pages: Document '{doc_id}' had no images after sorting."
                    )
                    batch_result[doc_id] = images
            else:
                rows = [row for row in sorted_subset]
                if not rows:
                    self.logger.warning(
                        f"get_batch_document_pages: Document '{doc_id}' had no rows after sorting."
                    )
                    batch_result[doc_id] = rows

        return batch_result


def _get_by_key(obj: dict, field_key: tuple) -> any:
    for k in field_key:
        try:
            obj = obj[k]
        except (KeyError, TypeError):
            raise ValueError(f"Key path {'.'.join(field_key)} not found.")
    return obj


def _get_features(
    original_features: Features, select_fields: list[tuple] | list[str]
) -> Features:
    """
    Construct a new Features spec for exactly the given `select_fields`.

    Each entry in `select_fields` may be either:
      - A dotted‐string like "question.text"
      - A tuple of keys like ("question", "text")

    Args:
        original_features (Features): The features of the original dataset.
        select_fields: (list[tuple] | list[str]): List of field keys or paths to include

    Returns:
        Features: Feature spec matching paths provided in `select_fields` (keys are dotted-string).

    Raises:
        ValueError: If any requested path does not exist in `original_features`.
    """
    select_features: dict[str, object] = {}

    for f in select_fields:
        if isinstance(f, str):
            field_key = tuple(f.split("."))
            field_path = f
        else:
            field_key = f
            field_path = ".".join(f)

        feature = _get_by_key(original_features, field_key)
        select_features[field_path] = feature

    return Features(select_features)


def project_fields(
    dataset: Dataset,
    select_fields: list[str],
    *,
    batched: bool = False,
    batch_size: int = 1000,
    **kwargs,
) -> Dataset:
    """
    Project specified nested fields into top-level columns of a Dataset.

    This function creates a new Dataset containing only the fields listed in
    `select_fields`. Each entry in `select_fields` is a dotted path (e.g.,
    "question.text") that specifies a nested field within the original Dataset.

    NOTE: Any deeper nested lists beyond the first “list of dicts” level
    (e.g., trying to drill into a tag’s elements like "question.tags.name")
    are not supported.

    Args:
        dataset (Dataset):
            The original Hugging Face Dataset to project fields from.
        select_fields (list[str]):
            List of dotted paths denoting which nested fields to extract.
            Example: ["question.text", "question.tags", "evidence.sources"].
        batched (bool, optional):
            If True, batch size controls how many examples are passed at once
            into the projection function. Batched mode can be faster for large
            datasets. Defaults to False.
        batch_size (int, optional):
            Number of examples per batch when `batched=True`. Ignored if
            `batched=False`. Defaults to 1000.

    Returns:
        Dataset:
            A new Dataset containing only the projected columns. Its `.features`
            attribute reflects the original feature types (ClassLabel, Sequence,
            etc.) for each projected field.

    Raises:
        ValueError:
            If any dotted path in `select_fields` does not exist in
            `dataset.features`, or if a batched invocation sees an unexpected
            structure (e.g., a non-list column when batched mode is used).
    """
    select_field_keys: list[tuple] = [tuple(f.split(".")) for f in select_fields]
    select_features = _get_features(dataset.features, select_field_keys)

    # Define projection functions for unbatched and batched modes
    def project_one(example: dict) -> dict:
        output: dict[str, object] = {}
        for path, key in zip(select_fields, select_field_keys):
            output[path] = _get_by_key(example, key)
        return output

    def project_batch(batch: dict) -> dict:
        output = {path: [] for path in select_fields}
        first_col = next(iter(batch.values()))
        if not isinstance(first_col, list):
            raise ValueError("Batched mapping expects each batch value to be a list.")
        n = len(first_col)

        for i in range(n):
            for path, key in zip(select_fields, select_field_keys):
                top_key = key[0]
                try:
                    subtree = batch[top_key][i]
                except (KeyError, IndexError):
                    raise ValueError(
                        f"Field '{top_key}' missing or index out of range in batch."
                    )
                # If there are deeper keys, descend into subtree; otherwise use subtree directly
                if len(key) > 1:
                    val = _get_by_key(subtree, key[1:])
                else:
                    val = subtree
                output[path].append(val)

        return output

    return dataset.map(
        project_batch if batched else project_one,
        batched=batched,
        batch_size=batch_size if batched else None,
        remove_columns=dataset.column_names,
        features=select_features,
        **kwargs,
    )


def filter_dataset(
    dataset: Dataset,
    *,
    field_filters: dict[str, any] | None = None,
    tag_filters: list[dict[str, any]] | None = None,
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
        field_filters (dict[str, any], optional):
           Mapping from dotted field paths to desired values. Example:
           {"answer.type": "not_answerable", "question.type": 3}.
        tag_filters (list[dict[str, any]], optional):
           Each dict must have keys:
             - "tags_list_path": dotted path to the tags list (e.g. "answer.tags")
             - "name": desired tag name (string or integer index)
             - "target": desired tag target (string or integer if ClassLabel)
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
                    if conversion of a string to ClassLabel index fails.
    """

    compiled_field_filters: dict[tuple, any] = {}
    if field_filters:
        # Resolve all field paths at once
        field_paths = list(field_filters.keys())
        field_features = _get_features(dataset.features, field_paths)

        for field_path in field_paths:
            field_key = tuple(field_path.split("."))
            field_feature = field_features[field_path]  # ClassLabel, Value, etc.
            raw_value = field_filters[field_path]

            if isinstance(field_feature, ClassLabel) and isinstance(raw_value, str):
                try:
                    expected_value = field_feature.str2int(raw_value)
                except KeyError:
                    raise ValueError(
                        f"Invalid class label '{raw_value}' for feature '{field_path}'."
                    )
            else:
                expected_value = raw_value

            compiled_field_filters[field_key] = expected_value

    compiled_tag_filters: list[dict[str, any]] = []
    if tag_filters:
        # Resolve all tag paths at once
        tags_list_paths = [filter["tags_list_path"] for filter in tag_filters]
        tags_list_features = _get_features(dataset.features, tags_list_paths)

        for filter in tag_filters:
            tags_list_path = filter["tags_list_path"]
            tags_list_key = tuple(tags_list_path.split("."))
            raw_name = filter["name"]
            expected_target = filter["target"]

            tags_list_feature = tags_list_features[tags_list_path]

            # Determine the inner feature dict:
            if isinstance(tags_list_feature, Sequence):
                tag_features = tags_list_feature.feature
            else:
                tag_features = tags_list_feature[0]

            # Convert tag "name" to int if needed
            name_feature = tag_features["name"]
            if isinstance(name_feature, ClassLabel) and isinstance(raw_name, str):
                try:
                    expected_name = name_feature.str2int(raw_name)
                except KeyError:
                    raise ValueError(
                        f"Invalid tag name '{raw_name}' for '{tags_list_path}.name'."
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
        for filter in compiled_tag_filters:
            try:
                tags_list = _get_by_key(example, filter["tags_list_key"])
            except ValueError:
                return False
            if not isinstance(tags_list, list):
                return False

            found = False
            for tag in tags_list:
                if (
                    tag.get("name") == filter["name"]
                    and tag.get("target") == filter["target"]
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
            # 1) Field-based checks
            field_pass = True
            for field_key, expected_value in compiled_field_filters.items():
                # top-level value or nested
                top_key = field_key[0]
                try:
                    subtree = batch[top_key][i]
                except (KeyError, IndexError):
                    field_pass = False
                    break
                if len(field_key) > 1:
                    try:
                        val = _get_by_key(subtree, field_key[1:])
                    except ValueError:
                        field_pass = False
                        break
                else:
                    val = subtree
                if val != expected_value:
                    field_pass = False
                    break
            if not field_pass:
                mask.append(False)
                continue

            # 2) Tag-based checks
            tag_pass = True
            for filter in compiled_tag_filters:
                top_key = filter["tags_list_key"][0]
                try:
                    subtree = batch[top_key][i]
                except (KeyError, IndexError):
                    tag_pass = False
                    break
                if len(filter["tags_list_key"]) > 1:
                    try:
                        tags_list = _get_by_key(subtree, filter["tags_list_key"][1:])
                    except ValueError:
                        tag_pass = False
                        break
                else:
                    tags_list = subtree
                if not isinstance(tags_list, list):
                    tag_pass = False
                    break
                found = False
                for tag in tags_list:
                    if (
                        tag.get("name") == filter["name"]
                        and tag.get("target") == filter["target"]
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
    batched: bool = False,
    batch_size: int = 1000,
    **kwargs,
) -> Dataset:
    """
    Add an 'images' column to each row in `dataset` by looking up pages via `corpus_index`.
    Two modes are supported:
      - "evidence_pages": For each example, fetch only the pages listed under `evidence.pages`.
      - "document_pages":  For each example, fetch all pages for the document given by `document.id`.

    Args:
        dataset (Dataset):
            The original Hugging Face Dataset to augment.
        corpus_index (CorpusIndex):
            An index built over the corpus allowing fast lookup by (doc_id, page_number)
            or by doc_id for all pages.
        mode (str):
            "evidence_pages" to look up only the pages in `evidence.pages` for each example.
            "document_pages" to look up all pages for the document. Required.
        batched (bool, optional):
            If True, apply the mapping in batches. Defaults to False.
        batch_size (int, optional):
            Number of examples per batch when `batched=True`. Defaults to 1000.

    Returns:
        Dataset:  A new Dataset identical to `dataset` but with an extra field "images".

    Raises:
        ValueError:
            If `mode` is not one of {"evidence_pages", "document_pages"}, or if required fields
            are missing in an example.
    """

    evidence_pages_key = ("evidence", "pages")
    document_id_key = ("document", "id")

    if mode not in ("evidence_pages", "document_pages"):
        raise ValueError(
            f"Unsupported mode '{mode}'. Expected 'evidence_pages' or 'document_pages'."
        )

    def add_images_one(example: dict) -> dict:
        try:
            doc_id = _get_by_key(example, document_id_key)
        except ValueError:
            raise ValueError("Missing 'document.id' for example when adding images.")

        if mode == "evidence_pages":
            try:
                pages = _get_by_key(example, evidence_pages_key)
            except ValueError:
                raise ValueError(
                    "Missing 'evidence.pages' for example when mode='evidence_pages'."
                )

            if not isinstance(pages, list):
                raise ValueError("'evidence.pages' must be a list of integers.")

            images = corpus_index.get_pages(doc_id, pages, return_images=True)
        else:  # mode == "document_pages"
            images = corpus_index.get_document_pages(doc_id, return_images=True)
            if images is None:
                images = []

        return {"images": images}

    def add_images_batch(batch: dict) -> dict:
        first_col = next(iter(batch.values()))
        if not isinstance(first_col, list):
            raise ValueError("Batched mapping expects each batch value to be a list.")
        n = len(first_col)

        images_list: list[list[Any]] = []

        if mode == "document_pages":
            doc_ids = []
            for i in range(n):
                top_doc = batch[document_id_key[0]][i]
                try:
                    doc_id = _get_by_key(top_doc, document_id_key[1:])
                except ValueError:
                    raise ValueError(f"Missing 'document.id' in batch element {i}.")
                doc_ids.append(doc_id)

            batch_images = corpus_index.get_batch_document_pages(
                doc_ids, return_images=True
            )
            for doc_id in doc_ids:
                imgs = batch_images.get(doc_id, [])
                images_list.append(imgs)
        else:  # mode == "evidence_pages"
            for i in range(n):
                top_doc = batch[document_id_key[0]][i]
                try:
                    doc_id = _get_by_key(top_doc, document_id_key[1:])
                except ValueError:
                    raise ValueError(f"Missing 'document.id' in batch element {i}.")

                top_ev = batch[evidence_pages_key[0]][i]
                try:
                    pages = _get_by_key(top_ev, evidence_pages_key[1:])
                except ValueError:
                    raise ValueError(f"Missing 'evidence.pages' in batch element {i}.")

                if not isinstance(pages, list):
                    raise ValueError(
                        f"'evidence.pages' must be a list in batch element {i}."
                    )

                imgs = corpus_index.get_pages(doc_id, pages, return_images=True)
                images_list.append(imgs)

        return {"images": images_list}

    if batched:
        return dataset.map(
            add_images_batch, batched=True, batch_size=batch_size, **kwargs
        )
    else:
        return dataset.map(add_images_one, **kwargs)
