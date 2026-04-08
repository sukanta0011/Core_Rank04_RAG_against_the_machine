from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.api.services.rag_services import RAGService


rag_service = RAGService()


async def lifespan(app: FastAPI):
    rag_service.initialize_resources()
    yield


app = FastAPI(
    title="vLLM RAG",
    lifespan=lifespan
    )


class UserQuery(BaseModel):
    question: str
    k: int = Field(gt=0, default=5)


@app.post("/ask")
async def ask_question(query: UserQuery):
    search_result = rag_service.retriever.get_matching_chunk(
        question=query.question,
        k=query.k
    )
    answer = rag_service.answer_generator.generate_answer(
        search_result=search_result
    )
    return {
        "received_question": query.question,
        "resource_locations": search_result.retrieved_sources,
        "answer": answer.answer
    }
