from pydantic import BaseModel, Field

__all__ = [
    "MPDocVQARawEntry",
    "DUDERawEntry",
    "MMLongBenchDocRawEntry",
    "ArxivQARawEntry",
    "TATDQARawEntry",
    "SlideVQARawEntry",
]


class RawEntry(BaseModel):
    """
    Base schema for one raw QA example before unification.
    Subclasses define fields *only*.
    """

    ...


### MPDocVQA ###


class MPDocVQARawEntry(RawEntry):
    """
    Schema for a single MP-DocVQA example.
    """

    question_id: int = Field(alias="questionId")
    question: str
    doc_id: str
    page_ids: list[str]
    answers: list[str] | None = None
    answer_page_idx: int | None = None
    data_split: str

    model_config = {
        "populate_by_name": True,
        "frozen": True,
    }


### DUDE ###


class AnswerBoundingBox(BaseModel):
    left: int
    top: int
    width: int
    height: int
    page: int

    model_config = {
        "populate_by_name": True,
        "frozen": True,
    }


class DUDERawEntry(RawEntry):
    """
    Schema for a single DUDE example.
    """

    question_id: str = Field(alias="questionId")
    question: str
    answers: list[str] | None = None
    answers_page_bounding_boxes: list[list[AnswerBoundingBox]] | None = None
    answers_variants: list[str] | None = None
    answer_type: str | None = None
    doc_id: str = Field(alias="docId")
    data_split: str

    model_config = {
        "populate_by_name": True,
        "frozen": True,
    }


### MMLongBenchDoc ###


class MMLongBenchDocRawEntry(RawEntry):
    """
    Schema for a single MMLongBench-Doc example.
    """

    question_id: str
    question: str
    doc_id: str
    doc_type: str
    answer: str
    evidence_pages: str
    evidence_sources: str
    answer_format: str

    model_config = {
        "populate_by_name": True,
        "frozen": True,
    }


### ArxivQA ###


class ArxivQARawEntry(RawEntry):
    """
    Schema for a single ArxivQA example.
    """

    id: str
    image: str
    options: list[str]
    question: str
    label: str
    rationale: str

    model_config = {
        "populate_by_name": True,
        "frozen": True,
    }


### TATDQA ###


class TATDQARawEntry(RawEntry):
    """
    Schema for a single TATDQA example.
    """

    doc_uid: str
    doc_page: int
    doc_source: str
    question_uid: str
    order: int
    question: str
    answer: list[str] | str | int | float
    derivation: str
    answer_type: str
    scale: str
    req_comparison: bool
    facts: list[str]
    block_mapping: list[dict[str, list[int]]]

    model_config = {
        "populate_by_name": True,
        "frozen": True,
    }


### SlideVQA ###


class SlideVQARawEntry(RawEntry):
    """
    Schema for a single SlideVQA example.
    """

    qa_id: int
    question: str
    answer: str
    arithmetic_expression: str | None = None
    evidence_pages: list[int]
    deck_name: str
    deck_url: str
    image_urls: list[str]
    answer_type: str | None = None
    resoning_type: str | None = Field(default=None, alias="reasoning_type")

    model_config = {
        "populate_by_name": True,
        "frozen": True,
    }
