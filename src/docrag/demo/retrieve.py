"""docrag demo – /retrieve router

Given a *doc_id* (produced by /ingest) and a text *query*, compute similarity
scores between the query and every page image of the document using one of the
registered retriever back‑ends.

Results are returned as a list sorted by descending similarity so the front‑end
can display the top‑k pages.
"""

import glob
from pathlib import Path
from typing import List
import tempfile

import torch
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from PIL import Image

from docrag.demo.ingestion import EMBED_STORE  # reuse cached page embeddings
from docrag.demo.retrieval import get_retriever  # wrapper factory

# ---------------------------------------------------------------------------
router = APIRouter(prefix="/retrieve", tags=["retrieval"])


class PageScore(BaseModel):
    page_number: int = Field(..., description="0‑based page index")
    score: float = Field(..., description="Similarity score (higher = closer)")


class RetrieveResponse(BaseModel):
    doc_id: str
    results: List[PageScore]


# ---------------------------------------------------------------------------


@router.get("/", response_model=RetrieveResponse)
async def retrieve(
    doc_id: str = Query(..., description="Document ID returned by /ingest"),
    query: str = Query(..., description="Text query to embed and compare"),
    retriever: str = Query("colpali", description="Which retriever back‑end to use"),
    top_k: int = Query(5, ge=1, le=50, description="How many top pages to return"),
):
    """Compute similarity scores for *query* against every page of *doc_id*."""

    # ---------------------------------------------------------------------
    # 1. Obtain / compute page embeddings
    # ---------------------------------------------------------------------

    if doc_id in EMBED_STORE:
        page_embeds = EMBED_STORE[doc_id]  # shape: (num_pages, dim)
    else:
        # Fall back to embedding on‑the‑fly.
        doc_dir = Path(tempfile.gettempdir()) / "docrag_docs" / doc_id
        if not doc_dir.exists():
            raise HTTPException(status_code=404, detail="Unknown doc_id – ingest first")

        image_paths = sorted(glob.glob(str(doc_dir / "*.jpg")))
        if not image_paths:
            raise HTTPException(
                status_code=404, detail="No page images found for doc_id"
            )

        retr = get_retriever(retriever)
        imgs = [Image.open(p) for p in image_paths]
        with torch.no_grad():
            page_embeds = retr.embed_images(imgs).cpu()
        EMBED_STORE[doc_id] = page_embeds  # cache for future calls

    num_pages = page_embeds.shape[0]

    # ---------------------------------------------------------------------
    # 2. Embed query text & score
    # ---------------------------------------------------------------------

    retr = get_retriever(retriever)
    query_emb = retr.embed_queries([query]).cpu()  # shape: (1, dim)

    scores = retr.score(query_emb, page_embeds)  # (1, num_pages)
    scores = scores.squeeze(0)  # -> (num_pages,)

    # ---------------------------------------------------------------------
    # 3. Top‑k & response formatting
    # ---------------------------------------------------------------------

    top_k = min(top_k, num_pages)
    best_scores, best_idx = torch.topk(scores, k=top_k, largest=True)

    results = [
        PageScore(page_number=int(i), score=float(s))
        for s, i in zip(best_scores.tolist(), best_idx.tolist())
    ]

    return RetrieveResponse(doc_id=doc_id, results=results)
