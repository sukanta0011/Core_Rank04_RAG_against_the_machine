"""Re-chunk retrieved sources into smaller pieces and re-rank them."""
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field

from src.base_patterns import MinimalSearchResults, MinimalSource
from src.data_retrieval.abstract_classes import (
    DataChunk, Retriever, Update)
from src.data_retrieval.helper_classes import (
    FULL_CODE_SYNTAX, FileFormats)
from src.speed_optimization.cache import AnswerCache


class ResourceRefiner(BaseModel):
    """Split the top retrieved chunks into smaller ones and re-search.

    Used to shrink the context given to the language model: only the
    best small pieces of the retrieved chunks are kept.

    Attributes:
        chunk_size: Maximum characters of a refined chunk.
        chunk_overlap: Overlap between refined chunks.
        k: Number of refined chunks to return.
        text_chunk: Chunker class for text files.
        code_chunk: Chunker class for code files.
        retriever: Retriever class used on the refined chunks.
        cache: Cache passed to the temporary retriever. A throwaway
            cache is created if None.
        method: Method name for the temporary retriever.
        temp_folder: Where the temporary index is written.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk_size: int = Field(gt=0, default=500)
    chunk_overlap: int = Field(ge=0, default=0)
    k: int = Field(gt=0, default=10)
    text_chunk: Type[DataChunk]
    code_chunk: Type[DataChunk]
    retriever: Type[Retriever]
    cache: Any = None
    method: str = "lexical"
    temp_folder: str = "data/temp/"

    def get_refined_sources(
            self,
            data: List[str],
            minimal_resource: List[MinimalSource],
            question: str, question_id: str | None = None,
                ) -> Tuple[MinimalSearchResults, Dict]:
        """Refine the retrieved chunks for one question.

        Args:
            data: Text of each retrieved chunk.
            minimal_resource: Source of each chunk, same order as `data`.
            question: Question text.
            question_id: Id for the result.

        Returns:
            (search result over the refined chunks, refined chunk
            id -> text).
        """
        new_chunks, updates = self.create_new_data_chunks(
            data=data, minimal_resource=minimal_resource)
        new_search_result = self.retrieved_from_new_data_chunks(
            new_chunks=new_chunks,
            updates=updates,
            question=question,
            k=self.k,
            question_id=question_id,
        )
        return new_search_result, new_chunks

    def create_new_data_chunks(
            self,
            data: List[str],
            minimal_resource: List[MinimalSource]
                ) -> Tuple[Dict, Update]:
        """Split every retrieved chunk into smaller chunks.

        Offsets are shifted by the parent chunk's start, so they point
        into the real file, not into the parent chunk.

        Args:
            data: Text of each retrieved chunk.
            minimal_resource: Source of each chunk, same order as `data`.

        Returns:
            (refined chunk id -> text, update listing the same chunks).
        """
        new_data: Dict[str, str] = {}
        updates = Update()

        text_chunk_instance = self.text_chunk(chunk_size=self.chunk_size)
        code_chunk_instance = self.code_chunk(chunk_size=self.chunk_size)

        for idx, resource in enumerate(minimal_resource):
            docs: List[Document] = []
            file_type = Path(resource.file_path).suffix[1:]
            file_type = FULL_CODE_SYNTAX.get(file_type, file_type)

            if file_type in FileFormats.TXT_LANGUAGES.value:
                docs = text_chunk_instance.process(
                    document=data[idx], chunk_overlap=self.chunk_overlap)
            elif file_type in FileFormats.CODING_LANGUAGES.value:
                set_language = getattr(
                    code_chunk_instance, "set_language", None)
                if set_language is not None:
                    set_language(file_type)
                docs = code_chunk_instance.process(
                    document=data[idx], chunk_overlap=self.chunk_overlap)

            for doc in docs:
                start = (doc.metadata["start_index"]
                         + resource.first_character_index)
                end = start + len(doc.page_content)
                key = f"{resource.file_path}:{start}:{end}"
                new_data[key] = doc.page_content
                updates.add[key] = doc.page_content

        return new_data, updates

    def retrieved_from_new_data_chunks(
            self,
            new_chunks: Dict,
            updates: Update,
            question: str,
            k: int, question_id: str | None = None,
            ) -> MinimalSearchResults:
        """Index the refined chunks temporarily and search them.

        Args:
            new_chunks: Refined chunk id -> text.
            updates: Update listing the refined chunks.
            question: Question text.
            k: Number of results (clamped to the number of chunks).
            question_id: Id for the result.

        Returns:
            The search result over the refined chunks.
        """
        cache = self.cache or AnswerCache(f"{self.temp_folder}cache.json")
        new_retriever = self.retriever(
            data=new_chunks,
            update=updates,
            cache=cache,
            method=self.method,
        )
        new_retriever.create_and_save_corpus_index(self.temp_folder)
        return new_retriever.retrieve_from_raw_data(
            hash="refiner", question=question,
            k=max(1, min(k, len(new_chunks))), question_id=question_id)
