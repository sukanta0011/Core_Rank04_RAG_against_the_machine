from pathlib import Path
from abc import ABC, abstractmethod
import time
from pydantic import (
    BaseModel, Field, field_validator,
    PrivateAttr)
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from .helper_classes import DataManager, FilesInDir, FileFormats


class MinimalSource(BaseModel):
    """The MinimalSource model represents a minimal source of information"""
    file_path: str
    first_character_index: int
    last_character_index: int


class DataChunk(ABC, BaseModel):
    chunk_size: int = Field(default=500)

    @abstractmethod
    def process(self, *args: Any) -> List[str]:
        pass


class TextChunk(DataChunk):
    def process(self, document: str) -> List[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=0)

        texts = text_splitter.split_text(document)
        return texts


class CodeChunk(DataChunk):
    # coding_language: Language = Field(default=Language.PYTHON)

    # def set_language(self, language: str) -> None:
    #     self.coding_language = Language(language)

    def process(self, document: str, language: str) -> List[str]:
        coding_language = Language(language)
        code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=coding_language,
            chunk_size=self.chunk_size,
            chunk_overlap=0)

        texts = code_splitter.split_text(document)
        return texts


class SplitDataByChunks(BaseModel):
    file_path: str
    storage_path: str
    chunk_size: int = Field(default=200)

    # Private attributes
    _all_chunks: List[MinimalSource] = PrivateAttr(default=list())
    _all_paths: Dict[str, List[Path]] = PrivateAttr(default=dict())
    _all_chunk_txt: List[str] = PrivateAttr(default=list())

    @field_validator('file_path', mode='before')
    def validate_path(cls, path_str: str) -> str:
        path = Path(path_str)
        if path.exists():
            return path_str
        else:
            raise FileNotFoundError(f"{path} does not exists")

    @field_validator('storage_path', mode='after')
    def validate_storage_path(cls, path_str: str) -> str:
        path = Path(path_str)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.touch()
            return path_str
        except PermissionError:
            raise PermissionError(
                f"Cannot create output file at '{path}': permission denied"
            )
        except OSError as e:
            raise OSError(f"Cannot create output path: {e}")

    def model_post_init(self, __context: Any) -> None:
        self._all_paths = FilesInDir.extract_all_file_paths(
            self.file_path)
        self.chunk_all_files()
        DataManager.save_data(self.storage_path, self._all_chunk_txt)
        self._all_chunk_txt = []

    def get_all_data_chunks(self) -> List[MinimalSource]:
        return self._all_chunks

    def chunk_all_files(self) -> None:
        text_chunk = TextChunk(chunk_size=self.chunk_size)
        code_chunk = CodeChunk(chunk_size=self.chunk_size)

        for syntax, paths in self._all_paths.items():
            try:
                if syntax in FileFormats.CODING_LANGUAGES.value:
                    # print(f"Coding_language: {syntax}")
                    for path in paths:
                        data = DataManager.load_data(str(path))
                        data_chunks = code_chunk.process(data, syntax)
                        self.store_data_chunks(
                            self._all_chunks, data_chunks,
                            str(path), self.storage_path)
                elif syntax in FileFormats.TXT_LANGUAGES.value:
                    for path in paths:
                        data = DataManager.load_data(str(path))
                        data_chunks = text_chunk.process(data)
                        self.store_data_chunks(
                            self._all_chunks, data_chunks,
                            str(path), self.storage_path)
            except Exception as e:
                raise Exception(f"{e}")

    def store_data_chunks(self, storage: List[MinimalSource],
                          data_chunks: List[str], path: str,
                          storage_path: str) -> None:
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


def chunking_test() -> None:
    # file_path = "vllm-0.10.1/CODE_OF_CONDUCT.md"
    # file_path = "vllm-0.10.1/tools/validate_config.py"
    # with open(file_path, "r") as fl:
    #     data = fl.read()
    splitter = SplitDataByChunks(file_path="vllm-0.10.1",
                                 chunk_size=2000,
                                 storage_path="data/raw_chunks.json")
    # chunks = splitter.get_splitted_code(data, 'python')
    # print(chunks)
    # for chunk in chunks:
    #     print(f"[[{chunk}]]")
    # splitter.store_all_file_paths()
    # codes = {item.value for item in Language}
    # print(codes)
    # print(f"{splitter._all_paths.keys()}")
    # all_chunks = splitter.get_all_data_chunks()
    # print(f"Total Chunks: {len(all_chunks)}")


if __name__ == "__main__":
    start_time = time.time()
    chunking_test()
    # data_chunks = [{"name": "hello"}, {"name": "sukanta"}]
    # fl = open("data/raw_chunks.json", "a")
    # # for text in data_chunks:
    # #     json.dump(text, fl, indent=4)
    # #     fl.write(",/n")
    # # json.dump(data_chunks, fl)
    # fl.close()
    end_time = time.time()
    print(f"Time taken: {round((end_time - start_time), 3)}s")
