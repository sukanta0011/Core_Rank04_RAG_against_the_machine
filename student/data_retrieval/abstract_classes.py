from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import List, Annotated
from student.data_retrieval.helper_classes import PrepareStorageFolder
from student.base_patterns import MinimalSearchResults, MinimalSource


class Retriever(BaseModel, ABC):
    storage_path: PrepareStorageFolder
    data: List[str]
    all_minimal_resource: List[MinimalSource]

    @abstractmethod
    def create_corpus_index(self) -> None:
        pass

    @abstractmethod
    def load_corpus_index(self) -> None:
        pass

    @abstractmethod
    def get_matching_chunk(
        self,
        question: Annotated[str, Field(min_length=3)],
        k: Annotated[int, Field(gt=0)],
        question_id: str | None = None
            ) -> MinimalSearchResults:
        pass
