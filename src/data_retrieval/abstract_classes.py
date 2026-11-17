"""Base types shared by every retriever (lexical, semantic, hybrid)."""
from abc import ABC, abstractmethod
from enum import Enum
import uuid
from pydantic import BaseModel, Field, InstanceOf
from typing import List, Annotated, Dict
from langchain_core.documents import Document
from src.base_patterns import MinimalSearchResults
from src.data_retrieval.helper_classes import (
    PrepareStorageFolder, ValidatedStoragePath)
from ..speed_optimization.cache import AnswerCache


class DataChunk(ABC, BaseModel):
    """Base class of the chunking strategies.

    Attributes:
        chunk_size: Maximum number of characters per chunk.
    """

    chunk_size: int = Field(gt=20, default=500)

    @abstractmethod
    def process(self, document: str,
                chunk_overlap: int) -> List[Document]:
        """Split a document into chunks.

        Args:
            document: Full text of one file.
            chunk_overlap: Characters shared by neighbouring chunks.

        Returns:
            Chunks with a `start_index` entry in their metadata.
        """


class Update(BaseModel):
    """Changes to apply to the vector store during indexing.

    Attributes:
        reset_flag: True when every old chunk is invalid (settings
            changed), so the collection must be dropped first.
        add: chunk id -> text to embed and store.
        remove: chunk id -> old text whose id must be deleted.
    """

    reset_flag: bool = False
    add: Dict = Field(default_factory=dict)
    remove: Dict = Field(default_factory=dict)


class RetrieveMethod(Enum):
    """Names of the supported retrieval methods."""

    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class Retriever(BaseModel, ABC):
    """Common interface and cache handling for all retrievers.

    Attributes:
        data: chunk id -> chunk text.
        update: Pending changes for the index (used when indexing).
        cache: Cache of earlier retrieval results.
        method: Retrieval method name, part of the cache key.
        corpus_digest: Digest of the indexed corpus. Part of the cache
            key, so entries from an older index never match. Needs the
            `corpus_digest` parameter of `AnswerCache`.
    """

    data: Dict
    update: Update
    cache: InstanceOf[AnswerCache]
    method: str
    corpus_digest: str = ""

    @abstractmethod
    def create_and_save_corpus_index(
        self, storage_path: PrepareStorageFolder
            ) -> None:
        """Build the index and save it.

        Args:
            storage_path: Folder where the index is stored.
        """

    @abstractmethod
    def load_corpus_index(
            self, storage_path: ValidatedStoragePath) -> None:
        """Open a previously saved index without modifying it.

        Args:
            storage_path: Folder where the index was stored.
        """

    def get_matching_chunk(
        self,
        question: Annotated[str, Field(min_length=3)],
        k: Annotated[int, Field(gt=0)],
        question_id: str | None = None
            ) -> MinimalSearchResults:
        """Return the top-k sources, using the cache when possible.

        Args:
            question: Question text.
            k: Number of sources to return.
            question_id: Id to put in the result. A new UUID is used if
                it is None.

        Returns:
            The search result, from the cache or from the index.
        """
        if question_id is None:
            question_id = str(uuid.uuid4())

        hash, resources = self.cache.get_sources_from_cache(
            question=question,
            resource_numbers=k,
            method=self.method,
            corpus_digest=self.corpus_digest)
        if resources is None or resources.retrieved_sources is None:
            return self.retrieve_from_raw_data(
                question=question,
                k=k,
                question_id=question_id,
                hash=hash
            )
        return MinimalSearchResults(
            question=question,
            question_id=question_id,
            retrieved_sources_scores=resources.retrieved_sources_scores
            or [],
            retrieved_sources=resources.retrieved_sources
        )

    @abstractmethod
    def retrieve_from_raw_data(
        self,
        hash: str,
        question: Annotated[str, Field(min_length=3)],
        k: Annotated[int, Field(gt=0)],
        question_id: str | None = None
            ) -> MinimalSearchResults:
        """Search the index (no cache lookup) and store the result.

        Args:
            hash: Cache key, to pass to `cache.update_answer_cache`.
            question: Question text.
            k: Number of sources to return.
            question_id: Id to put in the result.

        Returns:
            The search result.
        """
