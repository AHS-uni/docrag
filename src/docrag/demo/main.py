"""DocRAG demo – FastAPI app entry point."""

from fastapi import FastAPI

from .ingestion import router as ingest_router
from .retrieve import router as retrieve_router
from .generate import router as generate_router
from .rag import router as rag_router
from .models import router as models_router

app = FastAPI(title="DocRAG Demo")
app.include_router(ingest_router)
app.include_router(retrieve_router)
app.include_router(generate_router)
app.include_router(rag_router)
app.include_router(models_router)
