from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import (
    BaseModel, Field,
    PrivateAttr, validate_call)
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from .helper_classes import (
    DataManager, FilesInDir,
    FileFormats, ExistingPath, ValidatedStoragePath)


class MinimalSource(BaseModel):
    """The MinimalSource model represents a minimal source of information"""
    file_path: str
    first_character_index: int
    last_character_index: int


class DataChunk(ABC, BaseModel):
    chunk_size: int = Field(default=500)

    @abstractmethod
    def process(self, document: str) -> List[str]:
        pass


class TextChunk(DataChunk):
    def process(self, document: str) -> List[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=0)

        texts = text_splitter.split_text(document)
        return texts


class CodeChunk(DataChunk):
    coding_language: Language = Field(default=Language.PYTHON)

    def set_language(self, language: str) -> None:
        self.coding_language = Language(language)

    def process(self, document: str) -> List[str]:
        code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=self.coding_language,
            chunk_size=self.chunk_size,
            chunk_overlap=0)

        texts = code_splitter.split_text(document)
        return texts


class SplitDataByChunks(BaseModel):
    all_paths: Dict[str, List[Path]]
    chunk_size: int = Field(gt=20, default=200)

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
                        data_chunks = code_chunk.process(data)
                        self.store_data_chunks(
                            self._all_chunks, data_chunks,
                            str(path))
                elif syntax in FileFormats.TXT_LANGUAGES.value:
                    for path in paths:
                        data = DataManager.load_data(str(path))
                        data_chunks = text_chunk.process(data)
                        self.store_data_chunks(
                            self._all_chunks, data_chunks,
                            str(path))
            except Exception as e:
                raise Exception(f"{e}")

    def store_data_chunks(self, storage: List[MinimalSource],
                          data_chunks: List[str], path: str) -> None:
        position: int = 0
        for text in data_chunks:
            start = position
            end = position + len(text)
            minimal_source = MinimalSource(
                file_path=path,
                first_character_index=start,
                last_character_index=end
            )
            self._all_chunk_txt.append(text)
            storage.append(minimal_source)
            position += len(text)
