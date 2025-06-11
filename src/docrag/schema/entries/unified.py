"""
Pydantic models for a single VQA example in the unified dataset.
"""

from pydantic import BaseModel, Field, model_validator

from .enums import (
    QuestionType,
    DocumentType,
    EvidenceSource,
    AnswerFormat,
    AnswerType,
    TagName,
)

__all__ = ["Question", "Document", "Evidence", "Answer", "UnifiedEntry", "Tag"]


class Tag(BaseModel):
    """
    Annotation for an object or field.

    Attributes:
        name (TagName): Name of the annotation.
        target (str): Associated field or object for the tag.
        comment (str): Additional information for the tag (defaults to empty string).
    """

    name: TagName
    target: str
    comment: str = ""


class Question(BaseModel):
    """
    A question posed in a QA example.

    Attributes:
        id (str): Unique identifier for the question.
        text (str): The natural language text of the question.
        type (QuestionType): High-level question category (defaults to "other").
        tags (list[Tag]): List of tags for a field.
    """

    id: str
    text: str
    type: QuestionType = QuestionType.OTHER
    tags: list[Tag] = Field(default_factory=list)


class Document(BaseModel):
    """
    A document providing context for the question.

    Attributes:
        id (str): Unique identifier for the document.
        type (DocumentType): Primary document category (defaults to "other").
        count_pages (int): Number of pages in the document.
        tags (list[Tag]): List of tags for a field.
    """

    id: str
    type: DocumentType = DocumentType.OTHER
    count_pages: int
    tags: list[Tag] = Field(default_factory=list)


class Evidence(BaseModel):
    """
    Content from the document used to support the answer.

    Attributes:
        pages (list[int]): Page numbers forming the evidence (defaults to empty list).
        sources (list[EvidenceSource]): Source types within the page(s) (defaults to ["other"]).
        tags (list[Tag]): List of tags for a field.
    """

    pages: list[int] = Field(default_factory=list)
    sources: list[EvidenceSource] = Field(
        default_factory=lambda: [EvidenceSource.OTHER]
    )
    tags: list[Tag] = Field(default_factory=list)


class Answer(BaseModel):
    """
    The acceptable answer(s) to a question.

    Attributes:
        answerable (AnswerType): Whether the question is answerable (defaults to "none").
        variants (list[str]): List of valid answer variants (defaults to empty list).
        rationale (str): Explanation or justification (defaults to empty string).
        format (AnswerFormat): Expected answer format (defaults to "none").
        tags (list[Tag]): List of tags for a field.
    """

    type: AnswerType = AnswerType.NONE
    variants: list[str] = Field(default_factory=list)
    rationale: str = ""
    format: AnswerFormat = AnswerFormat.NONE
    tags: list[Tag] = Field(default_factory=list)

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
        id (str): Unique identifier for the entry.
        question (Question): The associated question.
        document (Document): The associated document.
        evidence (Evidence): Supporting evidence.
        answer (Answer): The answer object.
        tags (list[Tag]): List of tags for an object.
    """

    id: str
    question: Question
    document: Document
    evidence: Evidence
    answer: Answer
    tags: list[Tag] = Field(default_factory=list)

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
