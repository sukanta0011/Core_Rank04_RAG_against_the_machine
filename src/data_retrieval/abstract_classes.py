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


class Retriever(BaseModel, ABC):
    data: Dict
    update: Dict

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

    def get_all_minimal_source(self, ids: List) -> List[MinimalSource]:
        keys = list(self.data.keys())
        all_minimal_sources: List[MinimalSource] = []
        for i in ids:
            parts = str(keys[i]).split(":")
            if len(parts) > 3:
                path = ":".join(parts[:-2])
            else:
                path = parts[0]
            all_minimal_sources.append(
                MinimalSource(
                    file_path=path,
                    first_character_index=int(parts[-2]),
                    last_character_index=int(parts[-1])
                )
            )
        return all_minimal_sources

