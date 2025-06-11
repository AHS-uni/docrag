from typing import Any

from datasets import Dataset

from .utils import build_features, get_by_key, get_by_key_from_batch

__all__ = [
    "project_fields",
]


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

    projected_features = build_features(dataset.features, select_map)

    # Define projection functions for unbatched and batched modes
    def project_one(example: dict) -> dict[str, Any]:
        output: dict[str, object] = {}
        for new_column_name, field_key in select_map.items():
            output[new_column_name] = get_by_key(example, field_key)
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
                    val = get_by_key_from_batch(batch, field_key, i)
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
