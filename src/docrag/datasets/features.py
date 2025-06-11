from datasets import (
    ClassLabel,
    Features,
    Sequence,
    Value,
)

from docrag.schema.entries import (
    AnswerFormat,
    AnswerType,
    DocumentType,
    EvidenceSource,
    QuestionType,
    TagName,
)

__all__ = [
    "CORPUS_FEATURES",
    "QA_FEATURES",
]


def _build_corpus_features() -> Features:
    """
    Features spec for corpus datasets.
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
    Features spec for QA datasets.
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


CORPUS_FEATURES: Features = _build_corpus_features()
QA_FEATURES: Features = _build_qa_features()
