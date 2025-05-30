from collections import defaultdict
from collections.abc import Sequence

from PIL import Image
from datasets import Dataset
from datasets.features import ClassLabel, Sequence as SeqFeature, Features


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
            dict | Image.Image | None: The full row, image, or None if not found.
        """
        idx = self.doc_page_to_idx.get((doc_id, page_number))
        if idx is None:
            return None

        row = self.dataset[idx]
        return row["image"] if return_images else row

    def get_all_pages(
        self, doc_id: str, return_images: bool = True
    ) -> list | Dataset | None:
        """
        Retrieve all pages/images for a document, sorted by page_number.

        Args:
            doc_id (str): Document identifier.
            return_images (bool, optional): If True, return a list of image objects.
                                            If False, return a HF Dataset of rows. Defaults to True.

        Returns:
            Union[list[Image.Image], Dataset, None]: List of images, subset Dataset, or None if not found.
        """
        indices = self.doc_to_indices.get(doc_id)
        if not indices:
            return None

        # Load and sort by page_number
        subset = self.dataset.select(indices)
        sorted_subset = sorted(subset, key=lambda ex: ex["page_number"])

        if return_images:
            return [row["image"] for row in sorted_subset]
        else:
            return Dataset.from_list(sorted_subset)


def _resolve_feature(features: Features, path: Sequence[str]):
    """
    Locate a leaf feature in a nested Features spec.

    Unwraps any intermediate `Sequence` wrappers or raw Python lists
    and follows `path` keys through dict-like feature mappings.

    Args:
        features: The top-level `Features` object of a dataset.
        path: A sequence of keys leading to the desired leaf feature.

    Returns:
        The final `Feature` object at `features[path[0]][path[1]]…`.
    """
    feat = features
    for key in path:
        if isinstance(feat, list):
            feat = feat[0]
        if isinstance(feat, SeqFeature):
            feat = feat.feature
            feat = feat[key]
    if isinstance(feat, list):
        feat = feat[0]
    if isinstance(feat, SeqFeature):
        feat = feat.feature
    return feat


def filter_dataset(
    dataset: Dataset,
    *,
    tag_path: Sequence[str] | None = None,
    tag_name: str | None = None,
    tag_target: str | None = None,
    field_path: Sequence[str] | None = None,
    field_value: object | None = None,
    select_fields: str | Sequence[str] | None = None,
    batch_size: int = 1000,
    num_proc: int | None = None,
) -> Dataset:
    """
    Filter and (optionally) project nested fields from a Hugging Face dataset.

    You must specify either a tag filter or a field filter:
        Tag filter: keeps rows where `tag_path + ['name'] == tag_name`
        and `tag_path + ['target'] == tag_target`.
        Field filter: keeps rows where `field_path` equals `field_value`.

    If any of the features along the filter path are `ClassLabel`, string
    values are automatically mapped to label IDs.

    After filtering, if `select_fields` is given (e.g. `"question.text"`
    or `["question","answer.format"]`), only those nested fields are
    projected into new top-level columns and everything else is dropped.

    Args:
        dataset: A `Dataset` to filter.
        tag_path: Sequence of keys leading to a `tags` list (e.g., `["document","tags"]`).
        tag_name: Tag `"name"` to match (will be converted to ClassLabel ID).
        tag_target: Tag `"target"` string to match.
        field_path: Sequence of keys leading to a primitive or ClassLabel field.
        field_value: Value to match (string → ID if ClassLabel).
        select_fields: A dotted string or list of dotted strings describing
            nested fields to keep in the final output.
        batch_size: Batch size for `dataset.filter()`.
        num_proc: Number of processes for parallel filtering.

    Returns:
        A filtered `Dataset` with only the requested fields (if `select_fields`
        was supplied), or full rows otherwise.

    Raises:
        AssertionError: If neither a tag filter nor a field filter is specified.
    """
    # Ensure we have exactly one mode of filtering
    assert (tag_path and tag_name and tag_target) or (
        field_path and field_value
    ), "Must specify either tag_path+tag_name+tag_target or field_path+field_value"

    # Precompute ClassLabel → int mappings if needed
    if tag_path:
        assert tag_name is not None and tag_target is not None
        name_path = list(tag_path) + ["name"]
        name_feat = _resolve_feature(dataset.features, name_path)
        tag_name_idx = name_feat.str2int(tag_name)  # e.g. "missing" → 0

    if field_path:
        field_feat = _resolve_feature(dataset.features, field_path)
        if isinstance(field_feat, ClassLabel):
            field_value_idx = field_feat.str2int(field_value)
        else:
            field_value_idx = field_value

    def matches_tag(tags: list[dict]) -> bool:
        return any(
            (t["name"] == tag_name_idx) and (t["target"] == tag_target) for t in tags
        )

    def filter_fn(batch: dict[str, list]) -> list[bool]:
        if tag_path:
            data = batch[tag_path[0]]
            for key in tag_path[1:]:
                data = [item[key] for item in data]
            return [matches_tag(tag_list) for tag_list in data]

        data = batch[field_path[0]]
        for key in field_path[1:]:
            data = [item[key] for item in data]
        return [val == field_value_idx for val in data]

    # 1) Apply the filter
    filtered = dataset.filter(
        filter_fn,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
    )

    # 2) Project nested fields if requested
    if select_fields is not None:
        # Normalize to a list of dotted-path strings
        fields = (
            [select_fields] if isinstance(select_fields, str) else list(select_fields)
        )

        def project(example: dict) -> dict:
            output = {}
            for dotted in fields:
                keys = dotted.split(".")
                val = example
                for k in keys:
                    val = val[k]
                    output[dotted] = val
            return output

        filtered = filtered.map(
            project,
            batched=False,
            remove_columns=filtered.column_names,
        )

    return filtered
