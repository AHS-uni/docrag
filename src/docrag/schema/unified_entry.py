"""
Pydantic models for a single VQA example in the unified dataset.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from .enums import QuestionType, DocumentType, EvidenceSource, AnswerFormat

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
    type: QuestionType = Field(default=QuestionType.MISSING)


class Document(BaseModel):
    """
    A document providing context for the question.

    Attributes:
        id: Unique identifier for the document.
        type: Primary document category (defaults to "missing").
        num_pages: Number of pages in the document.
    """

    id: str
    type: DocumentType = Field(default=DocumentType.MISSING)
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

    answerable: bool
    variants: list[str] = Field(default_factory=list)
    rationale: str = Field(default_factory=str)
    format: AnswerFormat = Field(default=AnswerFormat.NONE)

    @model_validator(mode="after")
    def validate_answer_fields(self):
        # If answerable, variants must exist
        if self.answerable and not self.variants:
            raise ValueError(
                "`variants` must contain at least one item when `answerable` is True"
            )
        # If not answerable, other fields must be empty or None
        if not self.answerable:
            if self.variants:
                raise ValueError("`variants` must be empty when `answerable` is False")
            if self.rationale:
                raise ValueError("`rationale` must be empty when `answerable` is False")
            if self.format is not AnswerFormat.NONE:
                raise ValueError('`format` must be "none" when `answerable` is False')
        return self


class UnifiedEntry(BaseModel):
    """
    A single QA example in the unified dataset.

    Attributes:
        id:       Unique identifier for the entry.
        question: The associated question.
        document: The associated document.
        evidence: Supporting evidence (defaults to None).
        answer:   The answer object (defaults to None).
    """

    id: str
    question: Question
    document: Document
    evidence: Evidence | None = None
    answer: Answer | None = None
