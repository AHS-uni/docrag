from typing import Any

from datasets import Features

__all__ = [
    "get_by_key",
    "get_by_key_from_batch",
    "build_features",
]

def get_by_key(obj: dict, key: tuple) -> Any:
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


def get_by_key_from_batch(batch: dict, key: tuple, idx: int) -> Any:
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


def build_features(
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
        feature_type = get_by_key(original_features, field_key)
        features_spec[column_name] = feature_type

    return Features(features_spec)
