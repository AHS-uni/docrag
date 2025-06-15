"""
DocRAG demo – retrieval module.

Given a document ID and a text query, searches the per-document FAISS index
and returns the top-k most similar pages.
"""

import glob
import tempfile
from pathlib import Path
from typing import List

import faiss
from numpy import float32
import torch
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from PIL import Image

from .index_manager import get_index, create_index
from .retrieval import get_retriever

router = APIRouter(prefix="/retrieve", tags=["retrieval"])


class PageScore(BaseModel):
    """
    Similarity score for an individual document page.

    Attributes:
        page_number: 0-based index of the page.
        score: Similarity score (higher = more similar).
    """

    page_number: int = Field(..., description="0-based page index")
    score: float = Field(..., description="Similarity score (higher is closer)")


class RetrieveResponse(BaseModel):
    """
    Response model for the `/retrieve` endpoint.

    Attributes:
        doc_id: The document identifier.
        results: A list of PageScore entries, sorted by score descending.
    """

    doc_id: str
    results: List[PageScore]


@router.get("/", response_model=RetrieveResponse)
async def retrieve(
    doc_id: str = Query(..., description="Document ID returned by /ingest"),
    query: str = Query(..., description="Text query to search"),
    retriever: str = Query("colnomic-3b", description="Retriever backend key"),
    top_k: int = Query(5, ge=1, le=50, description="Number of top pages to return"),
) -> RetrieveResponse:
    """
    Retrieve the top-k most similar pages in a document for a text query.

    Args:
        doc_id: Identifier of a previously ingested document.
        query: Text query string.
        retriever: Which retriever backend to use.
        top_k: How many top pages to return.

    Returns:
        A RetrieveResponse containing the doc_id and a sorted list of PageScore.

    Raises:
        HTTPException: If the document is unknown or no page images exist.
    """
    # 1) Get or build the FAISS index for this document
    try:
        index = get_index(doc_id)
    except KeyError:
        # Build index on-the-fly if missing
        doc_dir = Path(tempfile.gettempdir()) / "docrag_docs" / doc_id
        if not doc_dir.exists():
            raise HTTPException(status_code=404, detail="Unknown doc_id")
        image_paths = sorted(glob.glob(str(doc_dir / "*.jpg")))
        if not image_paths:
            raise HTTPException(status_code=404, detail="No page images found")

        retr = get_retriever(retriever)
        imgs = [Image.open(p) for p in image_paths]
        with torch.no_grad():
            emb = (
                retr.embed_images(imgs)
                .to(dtype=torch.float32, device="cpu")
                .contiguous()
                .numpy()
            )
        create_index(doc_id, emb)
        index = get_index(doc_id)

    # 2) Embed the query and perform FAISS search
    retr = get_retriever(retriever)
    with torch.no_grad():
        q_emb = (
            retr.embed_queries([query])
            .to(dtype=torch.float32, device="cpu")
            .contiguous()
            .numpy()
        )
    faiss.normalize_L2(q_emb)
    distances, indices = index.search(q_emb, top_k)  # shapes: (1, top_k)

    # 3) Format the top-k results
    results = [
        PageScore(page_number=int(idx), score=float(dist))
        for dist, idx in zip(distances[0].tolist(), indices[0].tolist())
    ]

    return RetrieveResponse(doc_id=doc_id, results=results)
