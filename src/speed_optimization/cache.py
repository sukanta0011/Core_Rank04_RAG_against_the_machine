from __future__ import annotations
from pydantic import BaseModel
from typing import List, Tuple, TYPE_CHECKING
from src.data_retrieval.helper_classes import (
    DataManager, prepare_storage)
from src.base_patterns import MinimalSource

if TYPE_CHECKING:
    from ..data_retrieval.abstract_classes import RetrieveMethod


class AnswerCacheStructure(BaseModel):
    question_id: str | None = None,
    retrieved_sources_scores: List[float] | None = None
    retrieved_sources: List[MinimalSource] | None = None
    answer: str | None = None


class IndexCache:
    pass


class AnswerCache:
    def __init__(self, path: str):
        self.path = path
        prepare_storage(self.path)
        try:
            self._answer_from_cache =\
                DataManager.load_data(self.path)
        except Exception:
            self._answer_from_cache = {}

    def get_sources_from_cache(
        self,
        question: str,
        resource_numbers: int,
        method: str,
            ) -> Tuple[str, None | AnswerCacheStructure]:
        question_hash = DataManager.generate_hash_from_text(
            f"{question}:{resource_numbers}:{method}")
        resources: AnswerCacheStructure | None =\
            self._answer_from_cache.get(question_hash)
        return question_hash, resources

    def update_answer_cache(
        self,
        hash: str,
        scores=None,
        resources=None,
        answer=None,
        question_id=None
            ) -> None:
        self._answer_from_cache.update(
            {hash: AnswerCacheStructure(
                question_id=question_id,
                retrieved_sources_scores=scores,
                retrieved_sources=resources,
                answer=answer
            )}
        )
        # print(self._answer_from_cache)

    def save_to_answer_cache(self):
        DataManager.save_data(
            self.path, self._answer_from_cache
        )
