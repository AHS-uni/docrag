"""
Abstract base class for retrievers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from datasets import Dataset


class BaseRetriever(ABC):  # SKELETON
    """
    TODO
    """

    def __init__(self, *, top_k: int = 5, sim_threshold: float = 0.8) -> None:
        """
        Args:
            top_k (int): Number of documents to retrieve from the index.
            sim_threshold (int): Minimum similarity for retrieved chunk.
        """
        self.top_k = top_k
        self.sim_threshold = sim_threshold

    def build_index(self, *, corpus_dataset: Dataset, fields: dict[str, str]) -> None:
        """
        Create or load an index from a HuggingFace Dataset.

        Args:
            corpus (Dataset): HF Dataset with features:
                    - doc_id (string)
                    - page_number (int)
                    - image_path (PIL.Image or path string mapped to PIL)
            field: name of the feature for document IDs
            page_field: name of the feature for page numbers
            image_field: name of the feature for PIL page images
        """
        pass
