"""
DocRAG demo – end-to-end RAG endpoint.

Runs retrieval to pick top-K pages, then feeds those page images + query
into the generator (with optional system prompt & prompt template).
"""

import glob
import tempfile
from pathlib import Path
from typing import List, Optional

import faiss
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

from .index_manager import get_index, create_index
from .retrieval import get_retriever
from .retrieve import PageScore
from .generation import get_generator

router = APIRouter(prefix="/rag", tags=["rag"])


class RAGRequest(BaseModel):
    """
    Payload for the `/rag` endpoint.

    Attributes:
        doc_id: Document ID from `/ingest`.
        query: The user’s question.
        retriever: Retriever key (e.g. "colnomic-3b").
        generator: Generator key (e.g. "internvl-3b").
        top_k: Number of pages to retrieve & pass through.
        system_prompt: Optional system prompt for generation.
        prompt_template: Optional format string using "{text}".
    """

    doc_id: str = Field(..., description="Document ID for RAG mode")
    query: str = Field(..., description="User’s question")
    retriever: str = Field("colnomic-3b", description="Retriever backend key")
    generator: str = Field("internvl-3b", description="Generator backend key")
    top_k: int = Field(5, description="How many top pages to retrieve")
    system_prompt: Optional[str] = Field(
        None, description="System prompt for the generator"
    )
    prompt_template: Optional[str] = Field(
        None, description="Template string, use '{text}' to inject the query"
    )


class RAGResponse(BaseModel):
    """
    Response model for the `/rag` endpoint.

    Attributes:
        doc_id: Echoed document ID.
        query: Echoed user query.
        retrieval_results: List of PageScore entries (page_number + score).
        answer: The generated answer text.
    """

    doc_id: str
    query: str
    retrieval_results: List[PageScore]
    answer: str


@router.post("/", response_model=RAGResponse)
async def rag(req: RAGRequest) -> RAGResponse:
    """
    End-to-end Retrieval-Augmented Generation.

    1. Ensures a FAISS index exists (builds on-the-fly if needed).
    2. Embeds the query, searches top_k pages.
    3. Loads those page images.
    4. Applies prompt_template (if any) to the query.
    5. Calls the generator with the system_prompt and images.
    6. Returns both the retrieval scores and the generated answer.

    Raises:
        HTTPException: If doc_id is invalid or pages missing.
    """
    # ─── 1) Ensure FAISS index ────────────────────────────────────
    try:
        index = get_index(req.doc_id)
    except KeyError:
        # build index on-the-fly
        tempdir = Path(tempfile.gettempdir())
        doc_dir = tempdir / "docrag_docs" / req.doc_id
        if not doc_dir.exists():
            raise HTTPException(status_code=404, detail="Unknown doc_id")
        image_paths = sorted(glob.glob(str(doc_dir / "*.jpg")))
        if not image_paths:
            raise HTTPException(status_code=404, detail="No page images for doc_id")

        retr = get_retriever(req.retriever)
        imgs_all = [Image.open(p) for p in image_paths]
        with torch.no_grad():
            emb_all = (
                retr.embed_images(imgs_all)
                .to(dtype=torch.float32, device="cpu")
                .contiguous()
                .numpy()
            )
        create_index(req.doc_id, emb_all)
        index = get_index(req.doc_id)

    # ─── 2) Embed & search top_k ───────────────────────────────────
    retr = get_retriever(req.retriever)
    with torch.no_grad():
        q_emb = (
            retr.embed_queries([req.query])
            .to(dtype=torch.float32, device="cpu")
            .contiguous()
            .numpy()
        )
    faiss.normalize_L2(q_emb)
    distances, indices = index.search(q_emb, req.top_k)  # (1, top_k)

    page_idxs = indices[0].tolist()
    retrieval_results = [
        PageScore(page_number=int(i), score=float(d))
        for d, i in zip(distances[0].tolist(), page_idxs)
    ]

    # ─── 3) Load just those images ────────────────────────────────
    tempdir = Path(tempfile.gettempdir())
    doc_dir = tempdir / "docrag_docs" / req.doc_id
    imgs = []
    for pi in page_idxs:
        img_path = doc_dir / f"{pi:04}.jpg"
        if not img_path.exists():
            raise HTTPException(status_code=404, detail=f"Page {pi} not found")
        imgs.append(Image.open(img_path))

    # ─── 4) Apply prompt template ─────────────────────────────────
    text = req.query
    if req.prompt_template:
        try:
            text = req.prompt_template.format(text=text)
        except Exception:
            text = req.prompt_template.replace("{text}", text)

    # ─── 5) Generate ──────────────────────────────────────────────
    gen = get_generator(req.generator)
    answer = gen.generate(text, images=imgs, system_prompt=req.system_prompt)

    return RAGResponse(
        doc_id=req.doc_id,
        query=req.query,
        retrieval_results=retrieval_results,
        answer=answer,
    )
