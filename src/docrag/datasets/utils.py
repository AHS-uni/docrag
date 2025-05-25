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
from huggingface_hub import HfApi, CommitInfo

from docrag.schema import (
    AnswerType,
    QuestionType,
    DocumentType,
    EvidenceSource,
    AnswerFormat,
)

__all__ = [
    "build_corpus_features",
    "build_qa_features",
    "load_corpus_dataset",
    "load_qa_dataset",
    "push_dataset_to_hub",
]


def build_corpus_features() -> Features:
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


def build_qa_features() -> Features:
    """
    Features spec for QA entries matching unified schema.
    """
    q_types = [e.value for e in QuestionType]
    d_types = [e.value for e in DocumentType]
    e_sources = [e.value for e in EvidenceSource]
    a_formats = [e.value for e in AnswerFormat]
    a_types = [e.value for e in AnswerType]

    return Features(
        {
            "id": Value("string"),
            "question": {
                "id": Value("string"),
                "text": Value("string"),
                "type": ClassLabel(names=q_types),
            },
            "document": {
                "id": Value("string"),
                "type": ClassLabel(names=d_types),
                "num_pages": Value("int32"),
            },
            "evidence": {
                "pages": Sequence(Value("int32")),
                "sources": Sequence(ClassLabel(names=e_sources)),
            },
            "answer": {
                "type": ClassLabel(names=a_types),
                "variants": Sequence(Value("string")),
                "rationale": Value("string"),
                "format": ClassLabel(names=a_formats),
            },
        }
    )


def load_corpus_dataset(
    dataset_root: str | Path,
    corpus_file: str = "corpus.jsonl",
    cast_image: bool = True,
    streaming: bool = False,
) -> Dataset | IterableDataset:
    """
    Load the page-level corpus as a Hugging Face Dataset.

    Args:
        dataset_root: root folder containing corpus.jsonl and documents/
        corpus_file: name of the JSONL manifest (default "corpus.jsonl")
        cast_image: whether to cast 'image_path' → Image()
        streaming: if True, returns an IterableDataset

    Returns:
        Dataset or IterableDataset with columns [doc_id, page_number, image_path].
    """
    root = Path(dataset_root)
    features = build_corpus_features()
    ds = load_dataset(
        "json",
        data_files=str(root / corpus_file),
        features=features,
        split="train",
        streaming=streaming,
    )
    if cast_image:
        ds = ds.cast_column("image_path", Image())
    return ds


def load_qa_dataset(
    dataset_root: str | Path,
    splits: dict[str, str] | None = None,
    include_images: bool = False,
    streaming: bool = False,
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
    features = build_qa_features()

    ds = load_dataset(
        "json",
        data_files={k: str(root / v) for k, v in splits.items()},
        features=features,
        streaming=streaming,
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


def push_dataset_to_hub(
    dataset: Dataset | DatasetDict,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
    commit_message: str | None = None,
    revision: str | None = None,
) -> CommitInfo:
    """
    Pushes a Hugging Face Dataset or DatasetDict to the Hugging Face Hub.

    Args:
        dataset: Dataset or DatasetDict to push.
        repo_id: The repository ID (e.g., "username/dataset_name").
        token: Optional HF access token.
        private: Whether to create a private dataset.
        commit_message: Message for the commit (defaults to "Upload dataset").
        revision: Branch to push to (defaults to "main").
    """
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        private=private,
        exist_ok=True,
    )

    return dataset.push_to_hub(
        repo_id=repo_id,
        token=token,
        private=private,
        commit_message=commit_message,
        revision=revision,
    )
