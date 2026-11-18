"""Local HTTP API: retrieve sources only, or retrieve and answer."""
import threading
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.services.rag_services import RAGService
from src.base_patterns import MinimalSearchResults, MinimalSource

rag_service = RAGService()
generation_lock = threading.Lock()  # one LLM generation at a time
ANSWER_TOKENS = 100  # same limit as the CLI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the index at start-up, save the cache at shutdown."""
    rag_service.initialize_resources()
    yield
    rag_service.save_cache()


app = FastAPI(title="vLLM RAG", lifespan=lifespan)


class UserQuery(BaseModel):
    """Request body shared by both endpoints."""

    question: str = Field(min_length=3)
    k: int = Field(gt=0, le=50, default=5)


class SourcesResponse(BaseModel):
    """Response of /sources."""

    received_question: str
    resource_locations: List[MinimalSource]


class AnswerResponse(SourcesResponse):
    """Response of /ask: sources plus the generated answer."""

    answer: str


def _search(query: UserQuery) -> MinimalSearchResults:
    """Run retrieval and turn failures into HTTP errors."""
    if rag_service.retriever is None:
        raise HTTPException(503, "index not loaded. Run `index` first.")
    try:
        return rag_service.retriever.get_matching_chunk(
            question=query.question, k=query.k)
    except Exception as e:
        raise HTTPException(500, f"Search failed: {e}") from e


@app.post("/sources", response_model=SourcesResponse)
def get_sources(query: UserQuery) -> SourcesResponse:
    """Return only the retrieved source locations (fast, no LLM)."""
    result = _search(query)
    return SourcesResponse(
        received_question=query.question,
        resource_locations=result.retrieved_sources)


@app.post("/ask", response_model=AnswerResponse)
def ask_question(query: UserQuery) -> AnswerResponse:
    """Retrieve sources, then generate an answer from them."""
    result = _search(query)
    try:
        generator = rag_service.get_answer_generator()
        with generation_lock:
            answer = generator.generate_answer(
                search_result=result, tokens_limit=ANSWER_TOKENS,
                refiner=None)
    except Exception as e:
        raise HTTPException(500, f"Answer failed: {e}") from e
    return AnswerResponse(
        received_question=query.question,
        resource_locations=result.retrieved_sources,
        answer=answer.answer)
