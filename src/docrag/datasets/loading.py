from pathlib import Path

from datasets import (
    load_dataset,
    Dataset,
    IterableDataset,
    DatasetDict,
    IterableDatasetDict,
    Features,
    Value,
    Sequence,
    ClassLabel,
    Image,
)

from docrag.schema import (
    AnswerType,
    QuestionType,
    DocumentType,
    EvidenceSource,
    AnswerFormat,
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
            "doc_id": Value("string"),
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
                "num_pages": Value("int32"),
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


def load_corpus_dataset(
    dataset_root: str | Path,
    corpus_file: str = "corpus.jsonl",
    cast_image: bool = True,
    streaming: bool = False,
    **kwargs,
) -> Dataset | IterableDataset:
    """
    Load the page-level corpus as a Hugging Face Dataset.

    Args:
        dataset_root (str | Path): root folder containing corpus.jsonl and documents/
        corpus_file (str): name of the JSONL manifest (default "corpus.jsonl")
        cast_image (bool): whether to cast 'image_path' → Image()
        streaming (bool): if True, returns an IterableDataset

    Returns:
        Dataset or IterableDataset with columns ['doc_id', 'page_number', 'image']
        if cast_image is True, otherwise ['doc_id', 'page_number', 'image_path'].
    """
    root = Path(dataset_root)
    features = _build_corpus_features()
    ds = load_dataset(
        "json",
        data_files=str(root / corpus_file),
        features=features,
        split="train",
        streaming=streaming,
        **kwargs,
    )
    if cast_image:
        ds = ds.rename_column("image_path", "image")
        ds = ds.cast_column("image", Image())

    return ds


def load_qa_dataset(
    dataset_root: str | Path,
    splits: dict[str, str] | None = None,
    include_images: bool = False,
    streaming: bool = False,
    **kwargs,
) -> DatasetDict | IterableDatasetDict:
    """
    Load QA splits into a DatasetDict (or IterableDatasetDict),
    with optional 'evidence_images'.

    Args:
        dataset_root: root folder containing unified_qas/ and documents/
        splits: dict mapping split names → unified_qas/*.jsonl
        include_images: if True, adds and casts an 'evidence_images' column
        streaming: if True, returns an IterableDatasetDict

    Returns:
        DatasetDict or IterableDatasetDict with keys train/val/test.
    """
    root = Path(dataset_root)
    default = {
        "train": "unified_qas/train.jsonl",
        "val": "unified_qas/val.jsonl",
        "test": "unified_qas/test.jsonl",
    }
    splits = splits or default
    features = _build_qa_features()

    ds = load_dataset(
        "json",
        data_files={k: str(root / v) for k, v in splits.items()},
        features=features,
        streaming=streaming,
        **kwargs,
    )

    if include_images:

        def _add_images(example):
            base = root / "documents" / example["document"]["id"]
            example["evidence_images"] = [
                str(base / f"{p:03d}.jpg") for p in example["evidence"]["pages"]
            ]
            return example

        ds = ds.map(_add_images)
        ds = ds.cast_column("evidence_images", Image())

    return ds


# def load_dataset_dict(
#     dataset_root: Union[str, Path],
#     corpus_file: str = "corpus.jsonl",
#     qa_splits: Optional[Dict[str, str]] = None,
#     include_images: bool = False,
#     streaming: bool = False,
# ) -> DatasetDict:
#     """
#     Load both corpus and QA datasets into one DatasetDict:

#         {
#           "corpus": Dataset,
#           "qa":     DatasetDict(...)
#         }

#     Args:
#         dataset_root:   root folder path
#         corpus_file:    name of the corpus manifest JSONL
#         qa_splits:      override default QA splits if desired
#         include_images: whether to include 'evidence_images'
#     """
#     corpus = load_corpus_dataset(dataset_root, corpus_file)
#     qa = load_qa_dataset(dataset_root, qa_splits, include_images)
#     return DatasetDict({"corpus": corpus, "qa": qa})
