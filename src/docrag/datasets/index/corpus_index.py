from collections import defaultdict
import logging
from pathlib import Path

from PIL import Image
from datasets import Dataset

from docrag.utils import get_logger

__all__ = [
    "CorpusIndex",
]


class CorpusIndex:
    """
    Index for fast lookup of images in a corpus Dataset by document ID and page number.

    All methods return Image objects, or an empty list/None if no images are found.
    """

    def __init__(self, dataset: Dataset) -> None:
        """
        Initialize the index and build lookup tables.

        Args:
            dataset (Dataset): A Hugging Face Dataset containing at least these columns:
                - "document_id" (str): Document identifier
                - "page_number" (int): Page number
                - "image" (PIL.Image.Image): The image on that page
        """
        self.dataset = dataset
        # Map (document_id, page_number) → dataset index
        self.document_page_index: dict[tuple[str, int], int] = {}
        # Map document_id → list of dataset indices for that document
        self.document_to_indices: dict[str, list[int]] = defaultdict(list)

        self.logger = get_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG,
            log_file_path=Path("logs/corpus_index.log"),
        )

        for idx, example in enumerate(dataset):
            doc_id = example["document_id"]
            page_number = example["page_number"]
            self.document_page_index[(doc_id, page_number)] = idx
            self.document_to_indices[doc_id].append(idx)

    def get_page(self, document_id: str, page_number: int) -> Image.Image | None:
        """
        Retrieve a single image by (document_id, page_number).

        Args:
            document_id (str): Document identifier.
            page_number (int): Page number within the document.

        Returns:
            PIL.Image.Image | None: The image if found; otherwise None.
        """
        idx = self.document_page_index.get((document_id, page_number))
        if idx is None:
            self.logger.warning(
                f"get_page: Missing page {page_number} for document '{document_id}'."
            )
            return None

        row = self.dataset[idx]
        return row["image"]

    def get_pages(
        self, document_id: str, page_numbers: list[int]
    ) -> list[Image.Image | None]:
        """
        Retrieve a list of images for a document by looping over get_page().

        Args:
            document_id (str): Document identifier.
            page_numbers (list[int]): Page numbers to fetch.

        Returns:
            list[PIL.Image.Image | None]: One entry per page_number in order.
                If a page is missing, its entry is None.
        """
        results: list[Image.Image | None] = []
        for page_number in page_numbers:
            image = self.get_page(document_id, page_number)
            results.append(image)
        return results

    def get_document_pages(self, document_id: str) -> list[Image.Image]:
        """
        Retrieve all images for a document, sorted by page_number.

        Args:
            document_id (str): Document identifier.

        Returns:
            list[PIL.Image.Image]: List of images sorted by page_number.
                Returns an empty list if no pages are found.
        """
        indices = self.document_to_indices.get(document_id)
        if not indices:
            self.logger.warning(
                f"get_document_pages: No pages found for document '{document_id}'."
            )
            return []

        # Use HF Dataset sorting to ensure proper order
        subset = self.dataset.select(indices)
        sorted_subset = subset.sort("page_number")

        images: list[Image.Image] = [ex["image"] for ex in sorted_subset]
        if not images:
            self.logger.warning(
                f"get_document_pages: Document '{document_id}' had no images after sorting."
            )
        return images

    def get_batch_document_pages(
        self, document_ids: list[str]
    ) -> dict[str, list[Image.Image]]:
        """
        Retrieve images for multiple documents in one call by looping over get_document_pages().

        Args:
            document_ids (list[str]): List of document identifiers.

        Returns:
            dict[str, list[PIL.Image.Image]]: Mapping from each document_id to its list of images.
                If a document_id has no pages, its value is an empty list.
        """
        batch_result: dict[str, list[Image.Image]] = {}
        for document_id in document_ids:
            images = self.get_document_pages(document_id)
            batch_result[document_id] = images
        return batch_result
