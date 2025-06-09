"""
Enumerations used for labelling documents, questions, answers, and evidence
"""

from enum import Enum

__all__ = [
    "QuestionType",
    "DocumentType",
    "AnswerFormat",
    "EvidenceSource",
]


class QuestionType(str, Enum):
    """
    High-level categories of questions.

    Attributes:
        EXTRACTIVE: Ask to copy a span exactly as it appears in the document.
        VERIFICATION: Confirm a stated fact with a Yes/No answer.
        COUNTING: Ask to count explicit items.
        ARITHMETIC: Ask to compute or compare numbers.
        ABSTRACTIVE: Request a summary, description, or trend.
        PROCEDURAL: Request a workflow or set of steps.
        REASONING: Require multi-hop logic or causal explanation.
        OTHER: Catch-all for uncategorized questions.
        MISSING: Marks the field as needing downstream imputation.
    """

    EXTRACTIVE = "extractive"
    VERIFICATION = "verification"
    COUNTING = "counting"
    ARITHMETIC = "arithmetic"
    ABSTRACTIVE = "abstractive"
    PROCEDURAL = "procedural"
    REASONING = "reasoning"
    OTHER = "other"
    MISSING = "missing"


class DocumentType(str, Enum):
    """
    High-level genres or categories of documents.

    Attributes:
        LEGAL: Contracts, pleadings, statutes, licenses.
        FINANCIAL: Invoices, reports, tax forms.
        SCIENTIFIC: Research articles, white papers.
        TECHNICAL: Manuals, specifications, datasheets.
        POLICY: Internal or governmental policies.
        CORRESPONDENCE: Letters, e-mails, memos.
        MARKETING: Brochures, advertisements.
        PERSONAL_RECORD: IDs, resumes, academic records.
        NEWS: Newspaper or magazine articles.
        OTHER: Uncategorized or out-of-scope documents.
        MISSING: Marks the field as needing downstream imputation.
    """

    LEGAL = "legal"
    FINANCIAL = "financial"
    SCIENTIFIC = "scientific"
    TECHNICAL = "technical"
    POLICY = "policy"
    CORRESPONDENCE = "correspondence"
    MARKETING = "marketing"
    PERSONAL_RECORD = "personal_record"
    NEWS = "news"
    OTHER = "other"
    MISSING = "missing"


class AnswerFormat(str, Enum):
    """
    Data type of an answer value.

    Attributes:
        STRING: Free-form text or quoted span.
        REFERENCE: Reference to a document object.
        INTEGER: Whole-number result.
        FLOAT: Real-valued numeric result.
        BOOLEAN: True/False or Yes/No.
        LIST: Ordered or unordered list of items.
        OTHER: Any other format not listed above.
        MISSING: Marks the field as needing downstream imputation.
        NONE: Reserved for non-answerable questions.
    """

    STRING = "string"
    REFERENCE = "reference"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    OTHER = "other"
    MISSING = "missing"
    NONE = "none"


class AnswerType(str, Enum):
    """
    Attributes:
        ANSWERABLE: Question should be answered.
        NOT_ANSWERABLE: Question should not be answered.
        NONE: Reserved as default value
    """

    ANSWERABLE = "answerable"
    NOT_ANSWERABLE = "not_answerable"
    NONE = "none"


class EvidenceSource(str, Enum):
    """
    Source type for answer evidence within a document.

    Attributes:
        SPAN: Linear text (sentence, paragraph, heading).
        TABLE: Tabular structure.
        CHART: Plot, graph, or quantitative visual.
        IMAGE: Non-chart visual (diagram, photo).
        LAYOUT: Page layout or structure.
        NONE: No identifiable evidence source.
        OTHER: Any other source type.
        MISSING: Marks the field as needing downstream imputation.
    """

    SPAN = "span"
    TABLE = "table"
    CHART = "chart"
    IMAGE = "image"
    LAYOUT = "layout"
    NONE = "none"
    OTHER = "other"
    MISSING = "missing"
