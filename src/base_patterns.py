"""Pydantic data models exchanged between the pipeline stages.

The names and required fields follow the project specification. Extra
fields are allowed by the spec, but the required ones must not change.
"""
import uuid
from typing import List, Sequence

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A location in the corpus: a file and a character range.

    Attributes:
        file_path: Path exactly as in the corpus, e.g.
            `data/raw/vllm-0.10.1/docs/features/lora.md`. The grader
            compares it verbatim.
        first_character_index: Start of the range (inclusive).
        last_character_index: End of the range (exclusive, as produced
            by the chunker: start + length).
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without ground truth.

    Attributes:
        question_id: Unique id (a UUID is generated if missing).
        question: The question text.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its ground-truth sources and reference answer.

    Attributes:
        sources: The correct source locations.
        answer: The reference answer.
    """

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions.

    Attributes:
        rag_questions: Answered and/or unanswered questions.
    """

    rag_questions: Sequence[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """The retrieval result for one question.

    Attributes:
        question_id: Id of the question.
        question: The question text.
        retrieved_sources_scores: One score per source, best first.
            Scale depends on the method (BM25 score, distance, RRF).
        retrieved_sources: The top-k sources, best first.
    """

    question_id: str
    question: str
    retrieved_sources_scores: List[float]
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """A retrieval result plus the generated answer.

    Attributes:
        answer: Answer text generated from the retrieved sources.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Search results for a whole dataset (output of `search_dataset`).

    Attributes:
        search_results: One entry per question.
        k: Number of sources requested per question.
    """

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Search results with answers (output of `answer_dataset`).

    Attributes:
        search_results: One entry per question, with its answer.
        k: Number of sources requested per question.
    """

    search_results: List[MinimalAnswer]
    k: int
