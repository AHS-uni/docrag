"""
Model-management endpoints (list / load / unload / snapshot-download).

They expose both retrievers and generators registered in demo.registry.
"""

from typing import List, Literal

from fastapi import APIRouter, HTTPException, Query
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field

from .registry import RETRIEVERS, GENERATORS

router = APIRouter(prefix="/models", tags=["models"])
ModelType = Literal["generator", "retriever"]


class ModelEntry(BaseModel):
    name: str
    type: ModelType
    loaded: bool = Field(
        ..., description="True if a singleton instance is already in memory"
    )


class ModelList(BaseModel):
    models: List[ModelEntry]


# ────────────────────────────────────────────────────────────
@router.get("/", response_model=ModelList)
async def list_models() -> ModelList:
    """Return every registered model (both kinds) with load-state."""
    models: list[ModelEntry] = []
    for k in RETRIEVERS.available():
        models.append(
            ModelEntry(name=k, type="retriever", loaded=k in RETRIEVERS.loaded())
        )
    for k in GENERATORS.available():
        models.append(
            ModelEntry(name=k, type="generator", loaded=k in GENERATORS.loaded())
        )
    return ModelList(models=models)


# ────────────────────────────────────────────────────────────
@router.post("/load")
async def load_model(
    name: str = Query(..., description="Registry key, e.g. 'colpali'"),
    type: ModelType = Query(..., description="'retriever' or 'generator'"),
):
    (RETRIEVERS if type == "retriever" else GENERATORS).load(name)
    return {"status": "loaded", "name": name, "type": type}


# ────────────────────────────────────────────────────────────
@router.post("/unload")
async def unload_model(
    name: str = Query(..., description="Registry key"),
    type: ModelType = Query(...),
):
    (RETRIEVERS if type == "retriever" else GENERATORS).unload(name)
    return {"status": "unloaded", "name": name, "type": type}


# ────────────────────────────────────────────────────────────
@router.post("/download")
async def download_snapshot(
    name: str = Query(..., description="Registry key"),
    type: ModelType = Query(...),
):
    reg = RETRIEVERS if type == "retriever" else GENERATORS
    cls = reg._classes.get(name.lower())
    if cls is None:
        raise HTTPException(status_code=404, detail="Unknown model key")

    repo_id = cls.model_name
    cache_path = snapshot_download(repo_id, resume_download=True)
    return {"status": "downloaded", "repo": repo_id, "path": cache_path}
