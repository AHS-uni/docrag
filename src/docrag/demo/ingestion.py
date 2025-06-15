"""
DocRAG demo – ingestion module.

Uploads a PDF, renders pages to JPEG, and optionally pre-builds a per-document FAISS
index of page embeddings to enable fast intra-document retrieval.
"""

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List

import pymupdf
import torch
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
from PIL import Image

from .retrieval import get_retriever
from .index_manager import create_index

router = APIRouter(prefix="/ingest", tags=["ingestion"])


class IngestResponse(BaseModel):
    """
    Response model for the `/ingest` endpoint.

    Attributes:
        doc_id: Unique identifier for the ingested document.
        num_pages: Number of pages rendered.
        page_paths: File paths to the generated JPEGs.
    """

    doc_id: str
    num_pages: int
    page_paths: List[str]


def _save_pdf(upload: UploadFile, work_dir: Path) -> Path:
    """
    Save the uploaded PDF to disk.

    Args:
        upload: Uploaded PDF file.
        work_dir: Directory in which to save the PDF.

    Returns:
        Path to the saved PDF file.
    """
    pdf_path = work_dir / "document.pdf"
    with pdf_path.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return pdf_path


def _pdf_to_images(pdf_path: Path, dpi: int = 224) -> List[Path]:
    """
    Render each page of a PDF to a JPEG image.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering.

    Returns:
        List of paths to the rendered JPEG images.
    """
    doc = pymupdf.open(pdf_path)
    out_paths: List[Path] = []
    for i in range(len(doc)):
        pix = doc.load_page(i).get_pixmap(dpi=dpi)
        dst = pdf_path.parent / f"{i:04}.jpg"
        pix.save(dst)
        out_paths.append(dst)
    return out_paths


@router.post("/", response_model=IngestResponse, status_code=201)
@router.post("", response_model=IngestResponse, status_code=201)
async def ingest_pdf(
    file: UploadFile = File(..., description="PDF to ingest"),
    retriever: str = Query(
        "colnomic-3b", description="Retriever backend key for embedding"
    ),
    do_embed: bool = Query(True, description="Whether to build FAISS index now"),
) -> IngestResponse:
    """
    Ingest a PDF document: render pages and optionally build a FAISS index.

    Args:
        file: PDF file uploaded by the client.
        retriever: Which retriever backend to use for embeddings.
        do_embed: If True, embed pages and build the FAISS index immediately.

    Returns:
        An IngestResponse containing the document ID, number of pages, and page image paths.

    Raises:
        HTTPException: If the uploaded file is not a PDF.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="File must be a PDF")

    # Prepare working directory
    doc_id = uuid.uuid4().hex
    work_dir = Path(tempfile.gettempdir()) / "docrag_docs" / doc_id
    work_dir.mkdir(parents=True, exist_ok=True)

    # Save PDF and render to images
    pdf_path = _save_pdf(file, work_dir)
    page_paths = _pdf_to_images(pdf_path)

    # Optionally build per-document FAISS index
    if do_embed:
        retr = get_retriever(retriever)
        imgs = [Image.open(p) for p in page_paths]
        with torch.no_grad():
            emb = (
                retr.embed_images(imgs)
                .to(dtype=torch.float32, device="cpu")
                .contiguous()
                .numpy()
            )
        create_index(doc_id, emb)

    return IngestResponse(
        doc_id=doc_id,
        num_pages=len(page_paths),
        page_paths=[str(p) for p in page_paths],
    )
