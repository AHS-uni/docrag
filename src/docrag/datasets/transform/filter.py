from typing import Any

from datasets import ClassLabel, Dataset, Sequence

from .utils import get_by_key, get_by_key_from_batch

__all__ = [
    "filter_dataset",
]


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
                field_feature = get_by_key(dataset.features, field_key)
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
                tags_list_feature = get_by_key(dataset.features, tags_list_key)
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
                val = get_by_key(example, field_key)
            except ValueError:
                return False
            if val != expected_value:
                return False

        # Tag-based checks
        for tag_filter in compiled_tag_filters:
            try:
                tags_list = get_by_key(example, tag_filter["tags_list_key"])
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
                    val = get_by_key_from_batch(batch, field_key, i)
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
                    tags_list = get_by_key_from_batch(batch, filt["tags_list_key"], i)
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
