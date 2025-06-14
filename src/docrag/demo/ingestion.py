"""docrag demo – ingestion module (updated)

Uploads a PDF, renders pages to JPEG with PyMuPDF, and **optionally** embeds those
page images immediately using the lightweight retriever wrappers so that the
subsequent /retrieve request can work without recomputing.

Embeddings are cached in a simple in‑memory dict (`EMBED_STORE`) keyed by
`doc_id`. This avoids pulling in FAISS for the quick demo.
"""

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from docrag.demo.retrieval import get_retriever  # local wrapper factory

# ---------------------------------------------------------------------------
# Runtime‑level cache for embeddings – shared across the FastAPI process
# ---------------------------------------------------------------------------

from typing import Dict  # noqa: E402
import torch  # noqa: E402

EMBED_STORE: Dict[str, torch.Tensor] = {}

# ---------------------------------------------------------------------------
router = APIRouter(prefix="/ingest", tags=["ingestion"])


class IngestResponse(BaseModel):
    doc_id: str
    num_pages: int
    page_paths: List[str]


# ---------------------------------------------------------------------------
# helper fns
# ---------------------------------------------------------------------------


def _save_pdf(upload: UploadFile, work_dir: Path) -> Path:
    pdf_path = work_dir / "document.pdf"
    with pdf_path.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return pdf_path


def _pdf_to_images(pdf_path: Path, dpi: int = 224) -> List[Path]:
    doc = fitz.open(pdf_path)
    out_paths: List[Path] = []
    for i in range(len(doc)):
        pix = doc.load_page(i).get_pixmap(dpi=dpi)
        dst = pdf_path.parent / f"{i:04}.jpg"
        pix.save(dst)
        out_paths.append(dst)
    return out_paths


# ---------------------------------------------------------------------------
# endpoint
# ---------------------------------------------------------------------------


@router.post("/", response_model=IngestResponse, status_code=201)
@router.post(
    "", response_model=IngestResponse, status_code=201
)  # allow /ingest without slash
async def ingest_pdf(
    file: UploadFile = File(..., description="PDF to ingest"),
    retriever: str = Query("colpali", description="Which retriever to pre‑embed with"),
    do_embed: bool = Query(
        True, description="Skip if you prefer to embed at /retrieve time"
    ),
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="File must be a PDF")

    doc_id = uuid.uuid4().hex
    work_dir = Path(tempfile.gettempdir()) / "docrag_docs" / doc_id
    work_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = _save_pdf(file, work_dir)
    page_paths = _pdf_to_images(pdf_path)

    # Optional eager embedding ------------------------------------------------
    if do_embed:
        from PIL import Image  # local import to avoid PIL overhead if not needed

        retr = get_retriever(retriever)
        imgs = [Image.open(p) for p in page_paths]
        with torch.no_grad():
            page_embeds = retr.embed_images(imgs).cpu()
        EMBED_STORE[doc_id] = page_embeds  # shape: (num_pages, dim)

    return IngestResponse(
        doc_id=doc_id,
        num_pages=len(page_paths),
        page_paths=[str(p) for p in page_paths],
    )
