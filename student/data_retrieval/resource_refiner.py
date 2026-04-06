from abc import ABC
from pathlib import Path
from typing import List, Type
from pydantic import BaseModel, Field
from student.base_patterns import MinimalSource
from student.data_retrieval.chunk_data import (
    DataChunk)
from student.data_retrieval.helper_classes import (
    FileFormats, FULL_CODE_SYNTAX)
from student.data_retrieval.abstract_classes import (
    Retriever
)
from langchain_core.documents import Document


class ResourceRefiner(BaseModel, ABC):
    data: List[str]
    minimal_resource: List[MinimalSource]
    chunk_size: int = Field(gt=0, default=500)
    chunk_overlap: int = Field(ge=0, default=0)
    text_chunk: Type[DataChunk]
    code_chunk: Type[DataChunk]
    retriever: Type[Retriever]

    def create_new_data_chunks(self) -> List[str]:
        new_data = []
        text_chunk_instance = self.text_chunk(
            chunk_size=self.chunk_size)
        code_chunk_instance = self.text_chunk(
            chunk_size=self.chunk_size)

        for idx, resource in enumerate(self.minimal_resource):
            docs: List[Document] = []
            path = Path(resource.file_path)
            # print(path)
            file_type = path.suffix[1:]

            if file_type in FULL_CODE_SYNTAX:
                file_type = FULL_CODE_SYNTAX[file_type]

            if file_type in FileFormats.TXT_LANGUAGES.value:
                docs = text_chunk_instance.process(
                    document=self.data[idx],
                    chunk_overlap=self.chunk_overlap
                )
            elif file_type in FileFormats.CODING_LANGUAGES.value:
                docs = code_chunk_instance.process(
                    document=self.data[idx],
                    chunk_overlap=self.chunk_overlap
                )
            new_data.extend([chunk.page_content for chunk in docs])
        return new_data
    
    def _get_new_minimal_resources(self):
        pass