"""
FastAPI HTTP service for the RAG QA app.

Run:
    uvicorn app:app --reload --port 8000

Try:
    curl -X POST localhost:8000/query -H "Content-Type: application/json" \
      -d '{"question": "What is the refund window?", "top_k": 5}'
"""
from __future__ import annotations
import logging
import time

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import get_settings
from rag.generation import generate_answer
from rag.logging_utils import log_query
from rag.retrieval import retrieve
from rag.store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag.app")

settings = get_settings()
app = FastAPI(title="Cost-Efficient RAG Service", version="1.0")

_store: VectorStore | None = None
_client: anthropic.Anthropic | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore(settings.chroma_persist_dir, settings.chroma_collection)
    return _store


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not settings.anthropic_api_key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not set")
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1, le=50)
    metadata_filter: dict | None = None


class Citation(BaseModel):
    chunk_id: str
    source: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    used_no_context_fallback: bool
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    num_chunks_retrieved: int
    input_tokens: int
    output_tokens: int
    model: str


@app.get("/health")
def health():
    return {"status": "ok", "collection_size": get_store().count()}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    top_k = req.top_k or settings.default_top_k

    t0 = time.perf_counter()
    chunks = retrieve(get_store(), req.question, top_k, req.metadata_filter)
    retrieval_latency_ms = (time.perf_counter() - t0) * 1000

    result = generate_answer(get_client(), settings.generation_model, req.question, chunks)

    chunk_lookup = {c.chunk_id: c for c in chunks}
    citations = [
        Citation(
            chunk_id=cid,
            source=chunk_lookup[cid].metadata.get("source", "unknown"),
            score=chunk_lookup[cid].score,
        )
        for cid in result.cited_chunk_ids
        if cid in chunk_lookup
    ]

    log_query(
        question=req.question,
        top_k=top_k,
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=result.latency_ms,
        num_chunks_retrieved=len(chunks),
        num_chunks_cited=len(citations),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        used_no_context_fallback=result.used_no_context_fallback,
    )

    return QueryResponse(
        answer=result.answer,
        citations=citations,
        used_no_context_fallback=result.used_no_context_fallback,
        retrieval_latency_ms=round(retrieval_latency_ms, 2),
        generation_latency_ms=round(result.latency_ms, 2),
        total_latency_ms=round(retrieval_latency_ms + result.latency_ms, 2),
        num_chunks_retrieved=len(chunks),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        model=result.model,
    )
