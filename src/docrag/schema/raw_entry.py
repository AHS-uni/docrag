from pydantic import BaseModel, Field


class BaseRawEntry(BaseModel):
    """
    Base schema for one raw QA example before unification.
    Subclasses define fields *only*.
    """

    ...


### MPDocVQA ###


class MPDocVQARaw(BaseRawEntry):
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


class DUDERaw(BaseRawEntry):
    """
    Schema for a single DUDE QA example.
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


class MMLongBenchDocRaw(BaseRawEntry):
    """
    Schema for a single MMLongBench-Doc example.
    """

    doc_id: str
    doc_type: str
    question: str
    answer: str | float | int | list | None
    evidence_pages: list[int]
    evidence_sources: list[str]
    answer_format: str
    data_split: str

    model_config = {
        "populate_by_name": True,
        "frozen": True,
    }
