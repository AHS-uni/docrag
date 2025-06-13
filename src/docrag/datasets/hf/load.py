from pathlib import Path

from datasets import (
    Dataset,
    DatasetDict,
    Image,
    load_dataset,
)

from .features import CORPUS_FEATURES, QA_FEATURES

__all__ = [
    "load_corpus_dataset",
    "load_qa_dataset",
]


def load_corpus_dataset(
    dataset_root: str | Path,
    corpus_file: str = "corpus.jsonl",
    **kwargs,
) -> Dataset:
    """
    Load the page-level corpus as a Hugging Face Dataset.

    Args:
        dataset_root (str | Path): root folder containing corpus.jsonl and documents/
        corpus_file (str): name of the JSONL manifest (default "corpus.jsonl")

    Returns:
        Dataset with columns ['doc_id', 'page_number', 'image'].
    """
    root = Path(dataset_root)
    corpus_path = root / corpus_file
    if not corpus_path.exists():
        raise FileNotFoundError(corpus_path)

    ds = load_dataset(
        "json",
        data_files=str(corpus_path),
        features=CORPUS_FEATURES,
        split="train",
        **kwargs,
    )

    ds = ds.rename_column("image_path", "image")
    ds = ds.cast_column("image", Image())

    return ds


def load_qa_dataset(
    dataset_root: str | Path,
    splits: dict[str, str],
    **kwargs,
) -> DatasetDict:
    """
    Load QA splits into a DatasetDict,
    with optional 'evidence_images'.

    Args:
        dataset_root: root folder containing unified_qas/ and documents/
        splits: dict mapping split names → unified_qas/*.jsonl

    Returns:
        DatasetDict with keys specified in `splits`.
    """
    root = Path(dataset_root)
    splits = splits

    ds = load_dataset(
        "json",
        data_files={k: str(root / v) for k, v in splits.items()},
        features=QA_FEATURES,
        **kwargs,
    )

    return ds
