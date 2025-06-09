"""
Helpers for loading unified DocRAG datasets as 🤗 `datasets.Dataset`
objects in either standard or streaming mode.
"""

from pathlib import Path

from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Image,
    Sequence,
    Value,
    load_dataset,
)

from docrag.schema import (
    AnswerFormat,
    AnswerType,
    DocumentType,
    EvidenceSource,
    QuestionType,
    TagName,
)

__all__ = [
    "load_corpus_dataset",
    "load_qa_dataset",
]


def _build_corpus_features() -> Features:
    """
    Features spec for a page-level corpus (one JSONL line per page).
    """
    return Features(
        {
            "document_id": Value("string"),
            "page_number": Value("int32"),
            "image_path": Value("string"),
        }
    )


def _build_qa_features() -> Features:
    """
    Features spec for QA entries matching unified schema.
    """
    q_types = [e.value for e in QuestionType]
    d_types = [e.value for e in DocumentType]
    e_sources = [e.value for e in EvidenceSource]
    a_formats = [e.value for e in AnswerFormat]
    a_types = [e.value for e in AnswerType]
    tag_names = [e.value for e in TagName]

    tag_features = Features(
        {
            "name": ClassLabel(names=tag_names),
            "target": Value("string"),
            "comment": Value("string"),
        }
    )

    return Features(
        {
            "id": Value("string"),
            "question": {
                "id": Value("string"),
                "text": Value("string"),
                "type": ClassLabel(names=q_types),
                "tags": [tag_features],
            },
            "document": {
                "id": Value("string"),
                "type": ClassLabel(names=d_types),
                "count_pages": Value("int32"),
                "tags": [tag_features],
            },
            "evidence": {
                "pages": Sequence(Value("int32")),
                "sources": Sequence(ClassLabel(names=e_sources)),
                "tags": [tag_features],
            },
            "answer": {
                "type": ClassLabel(names=a_types),
                "variants": Sequence(Value("string")),
                "rationale": Value("string"),
                "format": ClassLabel(names=a_formats),
                "tags": [tag_features],
            },
            "tags": [tag_features],
        }
    )


_CORPUS_FEATURES: Features = _build_corpus_features()
_QA_FEATURES: Features = _build_qa_features()

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
        features=_CORPUS_FEATURES,
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
    Load QA splits into a DatasetDict (or IterableDatasetDict),
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
        features=_QA_FEATURES,
        **kwargs,
    )

    return ds
