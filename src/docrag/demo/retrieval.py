"""docrag demo – retrieval wrappers

Two lightweight retriever wrappers around:
  • vidore/colpali-v1.3-hf
  • nomic-ai/colnomic-embed-multimodal-3b

The wrappers expose a minimal, common interface so the rest of the demo
(app & API layer) can treat them identically.
"""

from functools import lru_cache
from typing import List, Protocol

import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

# ---------------------------------------------------------------------------
# Common interface
# ---------------------------------------------------------------------------


class Retriever(Protocol):
    """Protocol every concrete retriever must satisfy."""

    name: str

    # ------- embedding helpers --------------------------------------------
    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:  # [N, D]
        ...

    def embed_queries(self, queries: List[str]) -> torch.Tensor:  # [M, D]
        ...

    # ------- scoring -------------------------------------------------------
    def score(
        self, query_embeds: torch.Tensor, image_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Return a (Q × I) similarity matrix."""
        ...

    # ------- convenience ---------------------------------------------------
    def rank(self, queries: List[str], images: List[Image.Image]) -> torch.Tensor:
        """Shortcut that embeds and scores in one call."""
        q_emb = self.embed_queries(queries)
        i_emb = self.embed_images(images)
        return self.score(q_emb, i_emb)


# ---------------------------------------------------------------------------
# ColNomic 3B retriever (ColQwen2.5 backbone)
# ---------------------------------------------------------------------------


class ColNomicRetriever:
    """Wrapper around *nomic-ai/colnomic-embed-multimodal-3b* (a ColQwen2.5 model)."""

    name = "colnomic-3b"
    _MODEL_NAME = "nomic-ai/colnomic-embed-multimodal-3b"

    def __init__(self, device: str | None = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        from colpali_engine.models import (
            ColQwen2_5,
            ColQwen2_5_Processor,
        )  # lazy import

        self.model = ColQwen2_5.from_pretrained(
            self._MODEL_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            attn_implementation="flash_attention_2"
            if is_flash_attn_2_available()
            else None,
        ).eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(self._MODEL_NAME)

    # ---------------------------------------------------------------------
    # API methods
    # ---------------------------------------------------------------------

    @torch.inference_mode()
    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        batch = self.processor.process_images(images).to(self.model.device)
        return self.model(**batch)  # shape: [N, D]

    @torch.inference_mode()
    def embed_queries(self, queries: List[str]) -> torch.Tensor:
        batch = self.processor.process_queries(queries).to(self.model.device)
        return self.model(**batch)  # shape: [M, D]

    def score(
        self, query_embeds: torch.Tensor, image_embeds: torch.Tensor
    ) -> torch.Tensor:
        # Uses the processor utility to compute dot‑product similarity.
        return self.processor.score_multi_vector(query_embeds, image_embeds)


# ---------------------------------------------------------------------------
# ColPali v1.3 retriever
# ---------------------------------------------------------------------------


class ColPaliRetriever:
    """Wrapper around *vidore/colpali-v1.3-hf*"""

    name = "colpali-v1.3"
    _MODEL_NAME = "vidore/colpali-v1.3-hf"

    def __init__(self, device: str | None = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        from transformers import ColPaliForRetrieval, ColPaliProcessor  # type: ignore

        self.model = ColPaliForRetrieval.from_pretrained(
            self._MODEL_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(self._MODEL_NAME)

    # ---------------------------------------------------------------------
    # API methods
    # ---------------------------------------------------------------------

    @torch.inference_mode()
    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        batch = self.processor(images=images).to(self.model.device)
        return self.model(**batch).embeddings  # shape: [N, D]

    @torch.inference_mode()
    def embed_queries(self, queries: List[str]) -> torch.Tensor:
        batch = self.processor(text=queries).to(self.model.device)
        return self.model(**batch).embeddings  # shape: [M, D]

    def score(
        self, query_embeds: torch.Tensor, image_embeds: torch.Tensor
    ) -> torch.Tensor:
        return self.processor.score_retrieval(query_embeds, image_embeds)


# ---------------------------------------------------------------------------
# Simple registry util
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def _get_retriever(name: str) -> Retriever:
    """Factory with memoisation (so heavy weights load once per process)."""
    name = name.lower()
    if name in {ColNomicRetriever.name, "colnomic"}:
        return ColNomicRetriever()  # type: ignore[return-value]
    if name in {ColPaliRetriever.name, "colpali"}:
        return ColPaliRetriever()  # type: ignore[return-value]
    raise ValueError(f"Unknown retriever '{name}' – available: colnomic, colpali")


def get_retriever(name: str) -> Retriever:
    """Public helper – thin wrapper around *_get_retriever* so the rest of the app
    doesn’t need to know about @lru_cache.
    """
    return _get_retriever(name)


# ---------------------------------------------------------------------------
# CLI test – quick smoke‑test when running this file directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke‑test the retriever wrappers")
    parser.add_argument("image_path", help="Path to a sample image file (jpg/png)")
    parser.add_argument("query", help="Text query to embed and score")
    parser.add_argument(
        "--backend",
        choices=["colpali", "colnomic"],
        default="colpali",
        help="Which retriever backend to use",
    )
    args = parser.parse_args()

    from PIL import Image

    retriever = get_retriever(args.backend)
    img_emb = retriever.embed_images([Image.open(args.image_path)])
    qry_emb = retriever.embed_queries([args.query])
    sim = retriever.score(qry_emb, img_emb)
    print(f"Similarity score: {sim.item():.4f}")
