from pydantic import BaseModel, Field
from typing import List, Sequence
import uuid


class MinimalSource(BaseModel):
    """The MinimalSource model represents a minimal source of information"""
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """The RagDataset model represents a dataset of RAG questions"""
    rag_questions: Sequence[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """The MinimalSearchResults and MinimalAnswer models represent the search
       results and an answer"""
    question_id: str
    question_str: str
    retrieved_sources_indexes: List[int]
    retrieved_sources_scores: List[float]
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    """The StudentSearchResults and StudentSearchResultsAndAnswer models
    represent search results and search results with answers"""
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: List[MinimalAnswer]
