"""
DocRAG demo – single `/generate` endpoint, with global system prompt and template.
"""

import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

from .generation import get_generator

router = APIRouter(prefix="/generate", tags=["generation"])


class GenerateRequest(BaseModel):
    """
    Payload for the `/generate` endpoint.

    Attributes:
        doc_id: Optional document ID for RAG mode.
        pages: Optional list of page indices to use as evidence.
        query: The user’s question.
        generator: Key of the generator backend.
        system_prompt: Optional system-level instruction.
        prompt_template: Optional format string using '{text}'.
    """

    doc_id: Optional[str] = Field(None, description="Document ID for RAG mode")
    pages: Optional[List[int]] = Field(None, description="0-based page indices")
    query: str = Field(..., description="User’s question")
    generator: str = Field("internvl-3b", description="Generator backend key")
    system_prompt: Optional[str] = Field(
        None, description="Override system-level instruction"
    )
    prompt_template: Optional[str] = Field(
        None, description="Format string template, use '{text}' for query"
    )


class GenerateResponse(BaseModel):
    """
    Response model for `/generate`.

    Attributes:
        doc_id: Echoed document ID, if any.
        pages: Echoed page list, if any.
        query: Echoed user query.
        answer: The generated answer text.
    """

    doc_id: Optional[str]
    pages: Optional[List[int]]
    query: str
    answer: str


@router.post("/", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """
    Generate an answer, optionally in RAG mode with page images.

    Applies the global `system_prompt` and `prompt_template` from the request.

    Args:
        req: The GenerateRequest payload.

    Returns:
        A GenerateResponse with the answer text.

    Raises:
        HTTPException: If `doc_id`/`pages` are invalid.
    """
    # 1) Load evidence images if RAG mode
    imgs: List[Image.Image] = []
    if req.doc_id:
        doc_dir = Path(tempfile.gettempdir()) / "docrag_docs" / req.doc_id
        if not doc_dir.exists():
            raise HTTPException(status_code=404, detail="Unknown doc_id")
        if not req.pages:
            raise HTTPException(status_code=400, detail="Must supply pages for RAG")
        for pi in req.pages:
            img_path = doc_dir / f"{pi:04}.jpg"
            if not img_path.exists():
                raise HTTPException(status_code=404, detail=f"Page {pi} not found")
            imgs.append(Image.open(img_path))

    # 2) Apply prompt template if given
    text = req.query
    if req.prompt_template:
        try:
            text = req.prompt_template.format(text=text)
        except Exception:
            text = req.prompt_template.replace("{text}", text)


    # 4) Invoke the generator
    gen = get_generator(req.generator, device=None)
    answer = gen.generate(text, images=imgs or None, system_prompt=req.system_prompt)

    return GenerateResponse(
        doc_id=req.doc_id,
        pages=req.pages,
        query=req.query,
        answer=answer,
    )
