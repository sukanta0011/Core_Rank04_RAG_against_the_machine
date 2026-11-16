from abc import ABC, abstractmethod
from enum import Enum
import uuid
from pydantic import BaseModel, Field, InstanceOf
from typing import List, Annotated, Dict
from langchain_core.documents import Document
from src.base_patterns import MinimalSearchResults, MinimalSource
from src.data_retrieval.helper_classes import (
    PrepareStorageFolder, ValidatedStoragePath)
from ..speed_optimization.cache import (
    AnswerCacheStructure, AnswerCache)


class DataChunk(ABC, BaseModel):
    chunk_size: int = Field(gt=20, default=500)

    @abstractmethod
    def process(self, document: str,
                chunk_overlap: int) -> List[Document]:
        pass


class Update(BaseModel):
    add: Dict = Field(default_factory={})
    remove: Dict = Field(default_factory={})


class RetrieveMethod(Enum):
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class Retriever(BaseModel, ABC):
    data: Dict
    update: Update
    cache: InstanceOf[AnswerCache]
    method: str

    @abstractmethod
    def create_and_save_corpus_index(
        self, storage_path: PrepareStorageFolder
            ) -> None:
        pass

    @abstractmethod
    def load_corpus_index(
            self, storage_path: ValidatedStoragePath) -> None:
        pass

    def get_matching_chunk(
        self,
        question: Annotated[str, Field(min_length=3)],
        k: Annotated[int, Field(gt=0)],
        question_id: str | None = None
            ) -> MinimalSearchResults:
        if question_id is None:
            question_id = str(uuid.uuid4())

        hash, resources = self.cache.get_sources_from_cache(
            question=question,
            resource_numbers=k,
            method=self.method)
        if resources is None:
            return self.retrieve_from_raw_data(
                question=question,
                k=k,
                question_id=question_id,
                hash=hash
            )
        else:
            cash_source = AnswerCacheStructure(**resources)
            return MinimalSearchResults(
                question=question,
                question_id=cash_source.question_id,
                retrieved_sources_scores=\
                    cash_source.retrieved_sources_scores,
                retrieved_sources=cash_source.retrieved_sources
            )

    @abstractmethod
    def retrieve_from_raw_data(
        self,
        hash: str,
        question: Annotated[str, Field(min_length=3)],
        k: Annotated[int, Field(gt=0)],
        question_id: str | None = None
            ) -> MinimalSearchResults:
        pass