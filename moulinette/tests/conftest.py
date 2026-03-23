# ABOUTME: Shared pytest fixtures for moulinette tests.
# ABOUTME: Provides helper functions and sample data for retrieval evaluation tests.

import pytest
from moulinette.models import (
    MinimalSource,
    MinimalSearchResults,
    AnsweredQuestion,
    RagDataset,
    StudentSearchResults,
)


def make_source(file_path: str, start: int, end: int) -> MinimalSource:
    """Create a MinimalSource with the given file path and character range."""
    return MinimalSource(
        file_path=file_path,
        first_character_index=start,
        last_character_index=end,
    )


@pytest.fixture
def sample_rag_dataset():
    """3 questions with varying source counts:
    - q1: 2 sources (file_a 0-100, file_b 0-50)
    - q2: 1 source  (file_c 0-200)
    - q3: 1 source  (file_d 0-80)
    """
    return RagDataset(
        rag_questions=[
            AnsweredQuestion(
                question_id="q1",
                question="How does vLLM handle batching?",
                sources=[
                    make_source("file_a.py", 0, 100),
                    make_source("file_b.py", 0, 50),
                ],
                answer="It uses continuous batching.",
            ),
            AnsweredQuestion(
                question_id="q2",
                question="What is PagedAttention?",
                sources=[
                    make_source("file_c.py", 0, 200),
                ],
                answer="A memory-efficient attention mechanism.",
            ),
            AnsweredQuestion(
                question_id="q3",
                question="How does prefix caching work?",
                sources=[
                    make_source("file_d.py", 0, 80),
                ],
                answer="It caches KV blocks for shared prefixes.",
            ),
        ]
    )


@pytest.fixture
def sample_student_results_all_found():
    """Student results that match ALL ground-truth sources for all 3 questions."""
    return StudentSearchResults(
        k=10,
        search_results=[
            MinimalSearchResults(
                question_id="q1",
                question_str="How does vLLM handle batching?",
                retrieved_sources=[
                    make_source("file_a.py", 0, 100),
                    make_source("file_b.py", 0, 50),
                    make_source("file_x.py", 0, 30),  # extra, irrelevant
                ],
            ),
            MinimalSearchResults(
                question_id="q2",
                question_str="What is PagedAttention?",
                retrieved_sources=[
                    make_source("file_c.py", 0, 200),
                ],
            ),
            MinimalSearchResults(
                question_id="q3",
                question_str="How does prefix caching work?",
                retrieved_sources=[
                    make_source("file_d.py", 0, 80),
                ],
            ),
        ],
    )


@pytest.fixture
def sample_student_results_partial():
    """Student results with partial matches:
    - q1: finds file_a but NOT file_b  -> recall = 0.5
    - q2: finds file_c                 -> recall = 1.0
    - q3: finds nothing                -> recall = 0.0
    """
    return StudentSearchResults(
        k=10,
        search_results=[
            MinimalSearchResults(
                question_id="q1",
                question_str="How does vLLM handle batching?",
                retrieved_sources=[
                    make_source("file_a.py", 0, 100),
                    make_source("file_x.py", 0, 30),
                ],
            ),
            MinimalSearchResults(
                question_id="q2",
                question_str="What is PagedAttention?",
                retrieved_sources=[
                    make_source("file_c.py", 0, 200),
                ],
            ),
            MinimalSearchResults(
                question_id="q3",
                question_str="How does prefix caching work?",
                retrieved_sources=[
                    make_source("file_z.py", 0, 999),
                ],
            ),
        ],
    )
