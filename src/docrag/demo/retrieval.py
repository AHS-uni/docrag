"""DocRAG demo – concrete retriever backends and factory."""

from urllib.parse import uses_fragment
import torch
from PIL import Image

from .registry import RETRIEVERS
from .base import Retriever

def _as_2d(t):
    """Return (B, D) matrix for FAISS."""
    if not isinstance(t, torch.Tensor):  # unwrap ColModelOutput
        return t.embeddings  # .embeddings holds the tensor
    if t.ndim == 3:  # (B, S, D)  →  average over S
        return t.mean(dim=1)

@RETRIEVERS.register("colqwen")
class ColQwenetriever(Retriever):
    """Retriever using nomic-ai/colnomic-embed-multimodal-3b.

    Args:
        device: Optional device (e.g. "cuda:0" or "cpu").
    """

    name = "colqwen"
    model_name = "nomic-ai/colnomic-embed-multimodal-3b"

    def __init__(self, device: str | torch.device | None = None):
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        self.model = ColQwen2_5.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=device or "auto",
            # attn_implementation="flash_attention_2"
        ).eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(self.model_name, use_fast=True)

    @torch.inference_mode()
    def embed_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Embed a list of images into feature vectors.

        Args:
            images: List of PIL.Image page images.

        Returns:
            Tensor of shape [num_images, embedding_dim].
        """
        batch = self.processor.process_images(images).to(self.model.device)
        return _as_2d(self.model(**batch))

    @torch.inference_mode()
    def embed_queries(self, queries: list[str]) -> torch.Tensor:
        """Embed a list of text queries into feature vectors.

        Args:
            queries: List of strings.

        Returns:
            Tensor of shape [num_queries, embedding_dim].
        """
        batch = self.processor.process_queries(queries).to(self.model.device)
        return _as_2d(self.model(**batch))


@RETRIEVERS.register("colpali")
class ColPaliRetriever(Retriever):
    """Retriever using vidore/colpali-v1.3.

    Args:
        device: Optional device (e.g. "cuda:0" or "cpu").
    """

    name = "colpali"
    model_name = "vidore/colpali-v1.3"

    def __init__(self, device: str | torch.device | None = None):
        from colpali_engine.models import ColPali, ColPaliProcessor

        self.model = ColPali.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=device or "auto",
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(self.model_name, use_fast=True)

    @torch.inference_mode()
    def embed_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Embed a list of images into feature vectors.

        Args:
            images: List of PIL.Image page images.

        Returns:
            Tensor of shape [num_images, embedding_dim].
        """
        batch = self.processor.process_images(images).to(self.model.device)
        return _as_2d(self.model(**batch))

    @torch.inference_mode()
    def embed_queries(self, queries: list[str]) -> torch.Tensor:
        """Embed a list of text queries into feature vectors.

        Args:
            queries: List of strings.

        Returns:
            Tensor of shape [num_queries, embedding_dim].
        """
        batch = self.processor.process_queries(queries).to(self.model.device)
        return _as_2d(self.model(**batch))


def get_retriever(name: str, device: str | None = None) -> Retriever:
    """Load (if needed) and return a retriever instance.

    Args:
        name: Key name of the retriever.
        device: Optional device for model placement.

    Returns:
        Instance of the requested Retriever.
    """
    return RETRIEVERS.load(name, device=device)
