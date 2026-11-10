from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import List, Annotated, Dict
from langchain_core.documents import Document
from src.base_patterns import MinimalSearchResults, MinimalSource
from src.data_retrieval.helper_classes import (
    PrepareStorageFolder, ValidatedStoragePath)


class DataChunk(ABC, BaseModel):
    chunk_size: int = Field(gt=20, default=500)

    @abstractmethod
    def process(self, document: str,
                chunk_overlap: int) -> List[Document]:
        pass


class Update(BaseModel):
    add: Dict = Field(default_factory={})
    remove: Dict = Field(default_factory={})


class Retriever(BaseModel, ABC):
    data: Dict
    update: Update

    @abstractmethod
    def create_and_save_corpus_index(
        self, storage_path: PrepareStorageFolder
            ) -> None:
        pass

    @abstractmethod
    def load_corpus_index(self, storage_path: ValidatedStoragePath) -> None:
        pass

    @abstractmethod
    def get_matching_chunk(
        self,
        question: Annotated[str, Field(min_length=3)],
        k: Annotated[int, Field(gt=0)],
        question_id: str | None = None
            ) -> MinimalSearchResults:
        pass

