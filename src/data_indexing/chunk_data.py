from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import (
    BaseModel, Field,
    PrivateAttr, validate_call)
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from .helper_classes import (
    DataManager,
    FileFormats, ValidatedStoragePath)
from src.base_patterns import MinimalSource


class DataChunk(ABC, BaseModel):
    chunk_size: int = Field(gt=20, default=500)

    @abstractmethod
    def process(self, document: str,
                chunk_overlap: int) -> List[Document]:
        pass


class TextChunk(DataChunk):
    @validate_call
    def process(self, document: str,
                chunk_overlap: int = 0) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True)

        docs = text_splitter.create_documents([document])
        return docs


class CodeChunk(DataChunk):
    coding_language: Language = Field(default=Language.PYTHON)

    @validate_call
    def set_language(self, language: str) -> None:
        self.coding_language = Language(language)

    @validate_call
    def process(self, document: str, chunk_overlap: int = 0) -> List[Document]:
        code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=self.coding_language,
            chunk_size=self.chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True)

        docs = code_splitter.create_documents([document])
        return docs


class SplitDataByChunks(BaseModel):
    all_paths: Dict[str, List[Path]]
    chunk_size: int = Field(gt=20, default=200)
    code_overlap: int = Field(ge=0, default=0)
    txt_overlap: int = Field(ge=0, default=0)

    # Private attributes
    _all_chunks: List[MinimalSource] = PrivateAttr(default=list())
    _all_chunk_txt: List[str] = PrivateAttr(default=list())

    def get_all_minimal_sources(self) -> List[MinimalSource]:
        return self._all_chunks

    def get_all_data_chunks(self) -> List[str]:
        return self._all_chunk_txt

    @validate_call
    def save_chunked_data(self, storage_path: ValidatedStoragePath) -> None:
        DataManager.save_data(storage_path, self._all_chunk_txt)

    @validate_call
    def chunk_all_files(self) -> None:
        text_chunk = TextChunk(chunk_size=self.chunk_size)
        code_chunk = CodeChunk(chunk_size=self.chunk_size)

        for syntax, paths in self.all_paths.items():
            try:
                if syntax in FileFormats.CODING_LANGUAGES.value:
                    # print(f"Coding_language: {syntax}")
                    for path in paths:
                        data = DataManager.load_data(str(path))
                        code_chunk.set_language(syntax)
                        data_chunks = code_chunk.process(
                            document=data,
                            chunk_overlap=self.code_overlap)
                        self.store_data_chunks(
                            self._all_chunks, data_chunks,
                            str(path))
                elif syntax in FileFormats.TXT_LANGUAGES.value:
                    for path in paths:
                        data = DataManager.load_data(str(path))
                        data_chunks = text_chunk.process(
                            document=data,
                            chunk_overlap=self.txt_overlap)
                        self.store_data_chunks(
                            self._all_chunks, data_chunks,
                            str(path))
            except Exception as e:
                raise Exception(f"{e}")

    def store_data_chunks(self, storage: List[MinimalSource],
                          data_chunks: List[Document], path: str) -> None:
        for doc in data_chunks:
            start = doc.metadata["start_index"]
            end = start + len(doc.page_content)
            minimal_source = MinimalSource(
                file_path=path,
                first_character_index=start,
                last_character_index=end
            )
            self._all_chunk_txt.append(doc.page_content)
            storage.append(minimal_source)


# ------------- Tests---------------------

def test_text_chunking() -> None:
    text = (
        "Welcome to the world of False Positives! This is a classic challenge "
        "in RAG. Because BM25 is a 'keyword matcher' and not a "
        "meaning understander, it can easily get distracted by a chunk"
        " that has the same words but a completely different context."
    )
    text_splitter = TextChunk(chunk_size=100)
    docs = text_splitter.process(text, 10)
    for doc in docs:
        print(f"start: {doc.metadata.get("start_index")}, "
              f"stop: {doc.metadata.get("start_index", 0)
                       + len(doc.page_content)}, "
              f"content: {doc.page_content}")


if __name__ == "__main__":
    test_text_chunking()
