from pathlib import Path
import time
import json
from enum import Enum
from pydantic import BaseModel, Field, field_validator, PrivateAttr
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


FULL_CODE_SYNTAX = {
    'py': 'python'
}


class FileFormats(Enum):
    CODING_LANGUAGES = {item.value for item in Language}
    TXT_LANGUAGES = {'txt', 'md', 'json'}


class CodingLanguageMap(Enum):
    PYTHON = Language.PYTHON


class MinimalSource(BaseModel):
    """The MinimalSource model represents a minimal source of information"""
    file_path: str
    first_character_index: int
    last_character_index: int


class SplitDataByChunks(BaseModel):
    file_path: str
    storage_path: str
    chunk_size: int = Field(default=200)

    # Private attributes
    _all_chunks: List[MinimalSource] = PrivateAttr(default=list())
    _all_paths: Dict[str, List[Path]] = PrivateAttr(default=dict())

    @field_validator('file_path', mode='before')
    def validate_path(cls, path_str: str) -> str:
        path = Path(path_str)
        if path.exists():
            return path_str
        else:
            raise FileNotFoundError(f"{path} does not exists")

    @field_validator('storage_path', mode='after')
    def validate_storage_path(cls, path_str: str) -> None:
        path = Path(path_str)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                path.unlink()
            path.touch()
        except PermissionError:
            raise PermissionError(
                f"Cannot create output file at '{path}': permission denied"
            )
        except OSError as e:
            raise OSError(f"Cannot create output path: {e}")

    def model_post_init(self, __context: Any) -> None:
        self.store_all_file_paths()
        self.chunk_all_files()

    def get_all_data_chunks(self) -> List[MinimalSource]:
        return self._all_chunks

    def store_all_file_paths(self) -> None:
        parent_path = Path(self.file_path)
        for item in parent_path.rglob("*"):
            if item.is_file():
                suffix = item.suffix[1:]
                if len(suffix) == 0:
                    continue
                if suffix in FULL_CODE_SYNTAX:
                    suffix = FULL_CODE_SYNTAX[suffix]
                try:
                    self._all_paths[suffix].append(item)
                except KeyError:
                    self._all_paths[suffix] = []
                    self._all_paths[suffix].append(item)

        # for syntax, paths in self._all_paths.items():
        #     print(f"{syntax}: {len(paths)}: {paths[0]}")

    def chunk_all_files(self) -> None:
        for syntax, paths in self._all_paths.items():
            try:
                if syntax in FileFormats.CODING_LANGUAGES.value:
                    for path in paths:
                        data = self.get_data(str(path))
                        data_chunks = self.split_code(data, syntax)
                        self.store_data_chunks(
                            self._all_chunks, data_chunks, str(path), self.storage_path)
                elif syntax in FileFormats.TXT_LANGUAGES.value:
                    for path in paths:
                        data = self.get_data(str(path))
                        data_chunks = self.split_txt_file(data)
                        self.store_data_chunks(
                            self._all_chunks, data_chunks, str(path), self.storage_path)
            except Exception as e:
                raise Exception(f"{e}")

    def get_data(self, path: str) -> str:
        try:
            with open(path, 'r', errors='ignore') as fl:
                data = fl.read()
            return data
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Unable to read the document. Path: '{path}' does not exists."
            )
        except PermissionError:
            raise PermissionError(
                f"Unable to read the document. Path: '{path}' does not"
                " reading permission."
            )
        except Exception as e:
            raise Exception(
                f"Unknown error while trying to read '{path}'.\n"
                f"Error Message: {e}"
            )

    def store_data_chunks(self, storage: List[MinimalSource],
                          data_chunks: List[str], path: str,
                          storage_path: str) -> None:
        position: int = 0
        fl = open(storage_path, "a")
        for text in data_chunks:
            start = position
            end = position + len(text)
            minimal_source = MinimalSource(
                file_path=path,
                first_character_index=start,
                last_character_index=end
            )
            storage.append(minimal_source)
            position += len(text)
            json.dump(text, fl)
        fl.close()

    def split_txt_file(self, document: str):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=0)

        texts = text_splitter.split_text(document)
        return texts

    def split_code(self, document: str, coding_language: str):
        code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=coding_language,
            chunk_size=self.chunk_size,
            chunk_overlap=0)

        texts = code_splitter.split_text(document)
        return texts


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
    # chunking_test()
    data_chunks = [{"name": "hello"}, {"name": "sukanta"}]
    fl = open("data/raw_chunks.json", "a")
    for text in data_chunks:
        json.dump(text, fl, indent=4)
    fl.close()
    end_time = time.time()
    print(f"Time taken: {round((end_time - start_time), 3)}s")
