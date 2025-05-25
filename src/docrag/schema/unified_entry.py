"""
Pydantic models for a single VQA example in the unified dataset.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from .enums import QuestionType, DocumentType, EvidenceSource, AnswerFormat, AnswerType

__all__ = [
    "Question",
    "Document",
    "Evidence",
    "Answer",
    "UnifiedEntry",
]


class Question(BaseModel):
    """
    A question posed in a QA example.

    Attributes:
        id:   Unique identifier for the question.
        text: The natural language text of the question.
        type: High-level question category (defaults to "missing").
    """

    id: str
    text: str
    type: QuestionType = QuestionType.MISSING


class Document(BaseModel):
    """
    A document providing context for the question.

    Attributes:
        id: Unique identifier for the document.
        type: Primary document category (defaults to "missing").
        num_pages: Number of pages in the document.
    """

    id: str
    type: DocumentType = DocumentType.MISSING
    num_pages: int


class Evidence(BaseModel):
    """
    Content from the document used to support the answer.

    Attributes:
        pages: Page numbers forming the evidence (defaults to empty list).
        sources: Source types within the page(s) (defaults to empty list).
    """

    pages: list[int] = Field(default_factory=list)
    sources: list[EvidenceSource] = Field(default_factory=list)


class Answer(BaseModel):
    """
    The acceptable answer(s) to a question.

    Attributes:
        answerable: Whether the question is answerable.
        variants:   List of valid answer variants (defaults to empty list).
        rationale:  Explanation or justification (defaults to empty list).
        format:     Expected answer format (defaults to "none").
    """

    type: AnswerType = AnswerType.NONE
    variants: list[str] = Field(default_factory=list)
    rationale: str = Field(default_factory=str)
    format: AnswerFormat = AnswerFormat.NONE

    @model_validator(mode="after")
    def validate_answer_fields(self):
        # If answerable, variants must exist
        if self.type == AnswerType.ANSWERABLE:
            if self.variants == []:
                raise ValueError(
                    "`variants` must contain at least one item when `type` is 'answerable'"
                )
        # If not answerable, other fields must be empty
        if self.type == AnswerType.NOT_ANSWERABLE:
            if self.variants != []:
                raise ValueError(
                    "`variants` must be empty when `type` is 'not_answerable'"
                )
            if self.rationale != "":
                raise ValueError(
                    "`rationale` must be empty when `type` is 'not_answerable'"
                )
            if self.format is not AnswerFormat.NONE:
                raise ValueError(
                    "`format` must be 'none' when `type` is 'not_answerable'"
                )
        return self


class UnifiedEntry(BaseModel):
    """
    A single QA example in the unified dataset.

    Attributes:
        id:       Unique identifier for the entry.
        question: The associated question.
        document: The associated document.
        evidence: Supporting evidence.
        answer:   The answer object.
    """

    id: str
    question: Question
    document: Document
    evidence: Evidence
    answer: Answer

    @model_validator(mode="after")
    def validate_entry_fields(self):
        # If answerable, evidence pages must exist
        if self.answer.type == AnswerType.ANSWERABLE:
            if self.evidence.pages == []:
                raise ValueError(
                    "`evidence.pages` must be non-empty when `answer.type` is 'answerable'"
                )
        # If not answerable, evidence should be empty
        if self.answer.type == AnswerType.NOT_ANSWERABLE:
            if self.evidence.pages != [] and self.evidence.sources != []:
                raise ValueError(
                    "`evidence.pages` and 'evidence.sources' must be empty when `answer.type` is 'not_answerable'"
                )
        return self
