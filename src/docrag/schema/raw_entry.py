from pydantic import BaseModel, Field


class BaseRawEntry(BaseModel):
    """
    Base schema for one raw QA example before unification.
    Subclasses define fields *only*.
    """

    ...


class MPDocVQARaw(BaseRawEntry):
    """
    Schema for a single MP-DocVQA example, exactly mirroring the JSON keys.
    """

    question_id: int = Field(..., alias="questionId")
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
