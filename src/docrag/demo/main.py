from fastapi import FastAPI
from .ingestion import router as ingest_router

app = FastAPI(title="DocRAG Demo")
app.include_router(ingest_router)
