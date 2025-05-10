"""
Enumerations used for labelling documents, questions, answers, and evidence
"""

from enum import Enum

__all__ = [
    "DatasetSplit",
    "QuestionType",
    "DocumentType",
    "AnswerFormat",
    "EvidenceSource",
]


class DatasetSplit(str, Enum):
    """
    Dataset subdivision to which an example belongs.

    Members
    -------
    TRAIN
        Example is used during model optimisation.
    DEV
        Held-out set for hyper-parameter tuning or early-stopping.
    VAL
        Alternative name for *DEV*; included for interoperability.
    TEST
        Final, unseen set for reporting headline metrics.
    """

    TRAIN = "train"
    DEV = "dev"
    VAL = "val"
    TEST = "test"


class QuestionType(str, Enum):
    """
    High-level categories of questions.

    Members
    -------
    EXTRACTIVE
        Ask to copy a span or short list exactly as it appears in the document
        (e.g. “Who signed the contract?”).
    VERIFICATION
        Ask to confirm a stated fact with a Yes/No answer
        (e.g. “Does clause 4 allow early termination?”).
    COUNTING
        Ask to count explicit items (e.g. “How many references are cited?”).
    ARITHMETIC
        Ask to compute or compare numbers found in the context
        (e.g. averages, differences, maxima).
    ABSTRACTIVE
        Ask for a summary, description, or trend that generalises the content
        (e.g. “Describe the overall sales trend.”).
    PROCEDURAL
        Ask for a workflow or set of steps documented in the text
        (e.g. “Outline the anonymisation protocol.”).
    REASONING
        Require multi-hop logic, causal explanation, or hypothetical prediction
        (e.g. “If revenue grows at 5 %, project earnings for 2027.”).
    OTHER
        Catch-all for questions that do not fit any defined category.
    """

    EXTRACTIVE = "extractive"
    VERIFICATION = "verification"
    COUNTING = "counting"
    ARITHMETIC = "arithmetic"
    ABSTRACTIVE = "abstractive"
    PROCEDURAL = "procedural"
    REASONING = "reasoning"
    OTHER = "other"


class DocumentType(str, Enum):
    """
    High-level categories (genres) of documents

    Members
    -------
    LEGAL
        Contracts, pleadings, court opinions, statutes, licences.
    FINANCIAL
        Invoices, balance sheets, annual or quarterly reports, tax forms.
    SCIENTIFIC
        Research articles, technical papers, lab reports, white papers.
    TECHNICAL
        Manuals, specifications, standard-operating procedures, datasheets.
    POLICY
        Internal or governmental policy statements, regulations, guidelines.
    CORRESPONDENCE
        Letters, e-mails, memoranda—any directed communication between parties.
    MARKETING
        Brochures, ads, press releases, product catalogues.
    PERSONAL_RECORD
        Certificates, résumés/CVs, ID cards, medical or academic records.
    NEWS
        Newspaper or magazine articles, newswire stories, bulletins.
    OTHER
        Any document whose semantic category is out of scope or uncertain.
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


class AnswerFormat(str, Enum):
    """
    Data type of an answer value.

    Members
    -------
    STRING
        Free-form text or quoted span.
    REFERENCE
        Pointer to a document object (e.g. “figure 2”, “table A-3”, “page 4”).
    INTEGER
        Whole-number result (counts, rankings).
    FLOAT
        Real-valued numeric result (averages, percentages, monetary values).
    BOOLEAN
        True/False or Yes/No.
    LIST
        Ordered or unordered collection of homogeneous items.
    OTHER
        Any answer representation not covered above.
    """

    STRING = "string"
    REFERENCE = "object"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    OTHER = "other"


class EvidenceSource(str, Enum):
    """
    Where in the document the evidence for an answer was located.

    Members
    -------
    SPAN
        Linear text span (sentence, paragraph, heading).
    TABLE
        Tabular structure.
    CHART
        Plot, graph, or other visual quantitative representation.
    IMAGE
        Non-chart visual (diagram, photograph, scanned figure).
    NONE
        No identifiable source of evidence in the document.
    OTHER
        Any source type not enumerated above.
    """

    SPAN = "span"
    TABLE = "table"
    CHART = "chart"
    IMAGE = "image"
    NONE = "none"
    OTHER = "other"
