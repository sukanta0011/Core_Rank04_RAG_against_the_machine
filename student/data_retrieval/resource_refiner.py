from abc import ABC
from pathlib import Path
from typing import List, Type, Tuple
from pydantic import BaseModel, Field
from student.base_patterns import MinimalSource, MinimalSearchResults
from student.data_retrieval.chunk_data import (
    DataChunk)
from student.data_retrieval.helper_classes import (
    FileFormats, FULL_CODE_SYNTAX)
from student.data_retrieval.abstract_classes import (
    Retriever
)
from langchain_core.documents import Document


class ResourceRefiner(BaseModel, ABC):
    chunk_size: int = Field(gt=0, default=500)
    chunk_overlap: int = Field(ge=0, default=0)
    k: int = Field(gt=0, default=10)
    text_chunk: Type[DataChunk]
    code_chunk: Type[DataChunk]
    retriever: Type[Retriever]

    def get_refined_sources(
            self,
            data: List[str],
            minimal_resource: List[MinimalSource],
            question: str, question_id=None,
                ) -> Tuple[MinimalSearchResults, List[str]]:
        new_minimal_sources, new_chunks = self.create_new_data_chunks(
            data=data,
            minimal_resource=minimal_resource
        )
        new_search_result = self.retrieved_from_new_data_chunks(
            new_minimal_sources=new_minimal_sources,
            new_chunks=new_chunks,
            question=question,
            k=self.k,
            question_id=question_id,
        )
        print(len(new_chunks))
        print(new_search_result.retrieved_sources_scores)
        return new_search_result, new_chunks

    def create_new_data_chunks(
            self,
            data: List[str],
            minimal_resource: List[MinimalSource]
                ) -> Tuple[List[MinimalSource], List[str]]:
        new_data: List[str] = []
        new_minimal_source: List[MinimalSource] = []

        text_chunk_instance = self.text_chunk(
            chunk_size=self.chunk_size)
        code_chunk_instance = self.text_chunk(
            chunk_size=self.chunk_size)

        for idx, resource in enumerate(minimal_resource):
            docs: List[Document] = []
            path = Path(resource.file_path)
            # print(path)
            file_type = path.suffix[1:]

            if file_type in FULL_CODE_SYNTAX:
                file_type = FULL_CODE_SYNTAX[file_type]

            if file_type in FileFormats.TXT_LANGUAGES.value:
                docs = text_chunk_instance.process(
                    document=data[idx],
                    chunk_overlap=self.chunk_overlap
                )
            elif file_type in FileFormats.CODING_LANGUAGES.value:
                docs = code_chunk_instance.process(
                    document=data[idx],
                    chunk_overlap=self.chunk_overlap
                )
            # new_data.extend([chunk.page_content for chunk in docs])
            # print(docs)
            for doc in docs:
                new_data.append(doc.page_content)
                start = doc.metadata["start_index"]
                new_minimal_source.append(
                    MinimalSource(
                        file_path=resource.file_path,
                        first_character_index=start,
                        last_character_index=\
                            start + len(doc.page_content)
                    )
                )
        return new_minimal_source, new_data
    
    def retrieved_from_new_data_chunks(
            self, new_minimal_sources, new_chunks, question, k, question_id=None,
            ) -> MinimalSearchResults:
        new_retriever = self.retriever(
            data=new_chunks,
            all_minimal_resource=new_minimal_sources,
        )
        new_retriever.create_corpus_index()
        new_result = new_retriever.get_matching_chunk(
            question=question,
            k=k,
            question_id=question_id
        )
        return new_result
