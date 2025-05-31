from pathlib import Path
import json
from collections import OrderedDict

from docrag.schema.enums import AnswerFormat, AnswerType, EvidenceSource, QuestionType, DocumentType
from docrag.schema.raw_entry import SlideVQARaw
from docrag.unify.base import BaseUnifier
from docrag.schema import (
    UnifiedEntry,
    Question,
    Document,
    Evidence,
    Answer,
)


class SlideVQAUnifier(BaseUnifier[SlideVQARaw]):
    """
    Unifier for the SlideVQA dataset, which features questions about slide decks
    requiring reasoning across multiple images.
    """

    def _discover_raw_qas(self) -> list[Path]:
        """
        Find all SlideVQA JSONL files in the raw_qas directory.
        """
        return sorted(self.raw_qas_dir.glob("*.jsonl"))

    def _load_raw_qas(self, path: Path) -> list[SlideVQARaw]:
        """
        Load SlideVQA entries from a JSONL file.
        """
        raw_entries = []
        split = path.stem  # Extract split name from filename (train, dev, test)
        
        # Read the JSONL file line by line
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                    
                try:
                    item = json.loads(line.strip())
                    item["data_split"] = split
                    raw_entries.append(SlideVQARaw.model_validate(item))
                except Exception as e:
                    self.logger.warning(f"Error parsing entry in {path}: {e}")
                    continue
                    
        return raw_entries

    @staticmethod
    def _normalize_answer(answer: str | float | int | list) -> tuple[list[str], AnswerFormat]:
        """
        Normalize answer to a list of strings and determine its format.
        
        Args:
            answer: The raw answer value which could be a string, number or list
            
        Returns:
            tuple[list[str], AnswerFormat]:
                - variants - list of normalized answer strings
                - fmt - the detected format of the answer
        """
        if isinstance(answer, list):
            # For list answers, join them with a sentinel
            variants = [" ".join(f"<item> {item}" for item in answer)]
            fmt = AnswerFormat.LIST
        elif isinstance(answer, int):
            variants = [str(answer)]
            fmt = AnswerFormat.INTEGER
        elif isinstance(answer, float):
            variants = [str(answer)]
            fmt = AnswerFormat.FLOAT
        elif isinstance(answer, str):
            variants = [answer]
            if answer.lower() in ["yes", "no", "true", "false"]:
                fmt = AnswerFormat.BOOLEAN
            else:
                fmt = AnswerFormat.STRING
        else:
            variants = []
            fmt = AnswerFormat.NONE
            
        return variants, fmt

    def _convert_qa_entry(self, raw: SlideVQARaw) -> UnifiedEntry | None:
        """
        Map a raw SlideVQA entry into the unified schema.
        """
        split = raw.data_split.lower()
        
        # Build the Question model
        question_type = self._determine_question_type(raw.question, raw.reasoning_type)
        question = Question(
            id=raw.question_id,
            text=raw.question,
            type=question_type,
        )

        # Build the Document model
        # Use corpus records to count slides in the deck
        slide_numbers = [p for (doc, p, _) in self._corpus_records if doc == raw.doc_id]
        num_slides = len(set(slide_numbers))
        
        document = Document(
            id=raw.doc_id,
            type=self._map_document_type(raw.doc_id),
            num_pages=num_slides,
        )

        # Build the Evidence model
        evidence = Evidence()
        if split != "test" and raw.evidence_slide_indices:
            evidence = Evidence(
                pages=raw.evidence_slide_indices,
                sources=[EvidenceSource.MISSING]
            )

        # Build the Answer model
        if split == "test":
            # For test set, leave answer as default
            answer = Answer()
        else:
            variants, fmt = self._normalize_answer(raw.answer)
            
            # If there's an arithmetic expression, override the format
            if raw.arithmetic_expression:
                # Numerical answer with explicit arithmetic expression
                fmt = AnswerFormat.FLOAT if isinstance(raw.answer, float) else AnswerFormat.INTEGER
            
            answer = Answer(
                type=AnswerType.ANSWERABLE,
                variants=variants,
                format=fmt,
            )

        # If answerable but no evidence pages found, use all document pages
        if evidence.pages == [] and answer.type == AnswerType.ANSWERABLE:
            evidence = Evidence(pages=slide_numbers)
            if evidence.pages == []:  # still no evidence
                self.logger.debug(
                    "Skipping entry with ID '%s'. Unable to find evidence slides for document %s.",
                    raw.question_id,
                    raw.doc_id,
                )
                return None  # Skip due to missing evidence

        return UnifiedEntry(
            id=f"{raw.doc_id}_{raw.question_id}",
            question=question,
            document=document,
            evidence=evidence,
            answer=answer,
        )
    
    def _determine_question_type(self, question: str, reasoning_type: str | None) -> QuestionType:
        """
        Determine the question type based on the question text and reasoning type.
        """
        if reasoning_type is None:
            # Fallback to simple heuristics based on question text
            question = question.lower()
            if any(word in question for word in ["how many", "count", "number of"]):
                return QuestionType.COUNTING
            elif any(word in question for word in ["calculate", "sum", "total", "average", "difference"]):
                return QuestionType.ARITHMETIC
            elif any(word in question for word in ["why", "how does", "explain", "reason"]):
                return QuestionType.REASONING
            else:
                return QuestionType.EXTRACTIVE
        
        # Map reasoning types to question types
        reasoning_map = {
            "single-hop": QuestionType.EXTRACTIVE,
            "multi-hop": QuestionType.REASONING,
            "numerical": QuestionType.ARITHMETIC,
        }
        
        return reasoning_map.get(reasoning_type.lower(), QuestionType.MISSING)
    
    def _map_document_type(self, doc_id: str) -> DocumentType:
        """
        Map SlideVQA document IDs to DocumentType enum values based on filename patterns.
        """
        # Map common document types to their corresponding enum values
        doc_id_lower = doc_id.lower()
        
        if "real" in doc_id_lower and "estate" in doc_id_lower:
            return DocumentType.FINANCIAL
        elif "fraud" in doc_id_lower:
            return DocumentType.LEGAL
        elif "experiment" in doc_id_lower:
            return DocumentType.SCIENTIFIC
        elif "formwork" in doc_id_lower:
            return DocumentType.TECHNICAL
        elif "pathophisof" in doc_id_lower:
            return DocumentType.SCIENTIFIC
        elif "sigma" in doc_id_lower:
            return DocumentType.TECHNICAL
        elif "presentation" in doc_id_lower:
            return DocumentType.MARKETING
        elif "project" in doc_id_lower:
            return DocumentType.TECHNICAL
        elif "valuation" in doc_id_lower:
            return DocumentType.FINANCIAL
        elif "mobile" in doc_id_lower:
            return DocumentType.TECHNICAL
        else:
            return DocumentType.OTHER 
