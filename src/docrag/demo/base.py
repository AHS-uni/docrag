"""DocRAG demo – base interfaces."""

from abc import ABC, abstractmethod
import torch
from PIL import Image


class Retriever(ABC):
    """Abstract base class for retriever backends.

    Attributes:
        name: Unique key for this retriever.
        model_name: HuggingFace repo name of the model.
    """

    name: str
    model_name: str

    @abstractmethod
    def embed_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Embed a list of images into feature vectors.

        Args:
            images: List of PIL.Image instances.

        Returns:
            Tensor of shape [num_images, embedding_dim].
        """
        pass

    @abstractmethod
    def embed_queries(self, queries: list[str]) -> torch.Tensor:
        """Embed a list of text queries into feature vectors.

        Args:
            queries: List of query strings.

        Returns:
            Tensor of shape [num_queries, embedding_dim].
        """
        pass

class Generator(ABC):
    """Abstract base class for generation backends.

    Attributes:
        name: Unique key for this generator.
        model_name: HuggingFace repo ID.
    """

    name: str
    model_name: str

    @abstractmethod
    def generate(
        self,
        query: str,
        images: list[Image.Image] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Generate an answer for a text query (and optional images).

        Args:
            query: The user's question.
            images: Optional list of page images.
            system_prompt: Optional system-level prompt.

        Returns:
            The generated answer text.
        """
        pass
