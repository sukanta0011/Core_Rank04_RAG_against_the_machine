import json
from tqdm import tqdm
from pathlib import Path
from pydantic import (
    BaseModel, Field,
    PrivateAttr, validate_call)
from typing import List, Dict, Tuple, Any
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, Language)
from langchain_core.documents import Document
from .helper_classes import (
    DataManager, ExistingPath,
    FileFormats, prepare_storage_folder)
from src.base_patterns import MinimalSource
from .abstract_classes import DataChunk


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
    folder: str
    chunk_size: int = Field(gt=20, default=200)
    code_overlap: int = Field(ge=0, default=0)
    txt_overlap: int = Field(ge=0, default=0)
    use_indexing: bool = Field(default=False)

    # Private attributes
    _resaving_required: bool = True
    _indexing_storage_path: str = PrivateAttr(default=str)
    _exist_indexing_storage: bool = False
    _all_chunks: List[MinimalSource] = PrivateAttr(default=list())
    _all_chunk_txt: List[str] = PrivateAttr(default=list())
    _indexing_storage: Dict = PrivateAttr(default=dict())
    _all_chunks_with_id: Dict = PrivateAttr(default=dict())
    _chunks_to_be_updated: Dict = PrivateAttr(default=dict())

    def model_post_init(self, __context) -> Any:
        if self.use_indexing:
            self._indexing_storage_path = "/".join([self.folder, 'indexed_data.json'])
            self._chunks_to_be_updated = {
                "add": {},
                "remove": {}
            }
            self.prepare_indexing_storage()

    def get_all_data(self) -> Tuple[Dict, Dict]:
        return self._all_chunks_with_id, self._chunks_to_be_updated

    def prepare_indexing_storage(self) -> None:
        try:
            with open(
                    self._indexing_storage_path, "r",
                    encoding="utf-8", newline="") as fl:
                self._indexing_storage = json.load(fl)
            if self.validate_settings(
                    self._indexing_storage.get("settings")):
                self._exist_indexing_storage = True
            else:
                self._prepare_fresh_storage()
        except (OSError, json.JSONDecodeError, FileNotFoundError):
            self._prepare_fresh_storage()

    def _prepare_fresh_storage(self) -> Dict:
        path = Path(self._indexing_storage_path)
        prepare_storage_folder(self.folder)
        path.touch()
        self._indexing_storage = {}
        self._indexing_storage["settings"] = {
            "chunk_size": self.chunk_size,
            "code_overlap": self.code_overlap,
            "txt_overlap": self.txt_overlap,
        }
        self._indexing_storage["files"] = {}

    def validate_settings(self, settings: Dict) -> bool:
        existing_chunk_size = settings.get("chunk_size")
        existing_text_overlap = settings.get("code_overlap")
        existing_code_overlap = settings.get("txt_overlap")
        return (
            self.chunk_size == existing_chunk_size and
            self.code_overlap == existing_code_overlap and
            self.txt_overlap == existing_text_overlap
        )

    def save_chunked_data(self) -> None:
        if self._resaving_required:
            if self.use_indexing:
                DataManager.save_data(
                    self._indexing_storage_path, self._indexing_storage)
            data_file = "/".join([self.folder, 'data.json'])
            DataManager.save_data(data_file, self._all_chunks_with_id)
            data_file = "/".join([self.folder, 'update.json'])
            DataManager.save_data(data_file, self._chunks_to_be_updated)

    @staticmethod
    @validate_call
    def load_from_files(
        storage_folder: ExistingPath
            ) -> Tuple[Dict, Dict]:

        all_data_source = "/".join([storage_folder, 'data.json'])
        all_update_source = "/".join([storage_folder, 'update.json'])

        # all_minimum_source = []
        # all_text_data = []
        all_data = {}
        all_update = {}

        with open(all_data_source, 'r', encoding="utf-8", newline="") as fl:
            all_data = json.load(fl)

        with open(all_update_source, 'r', encoding="utf-8", newline="") as fl:
            all_update = json.load(fl)

        return all_data, all_update

    def chunk_all_files(self) -> None:
        if self.use_indexing:
            self._chunk_files_with_indexing()
        else:
            self._chunk_files()

    def _extract_allowed_file_paths(self) -> List[Dict]:
        allowed_paths = []
        for syntax, paths in self.all_paths.items():
            if syntax in FileFormats.CODING_LANGUAGES.value or \
                    syntax in FileFormats.TXT_LANGUAGES.value:
                for path in paths:
                    allowed_paths.append({syntax: str(path)})
        return allowed_paths

    def chunk_to_be_removed(self, files: Dict) -> None:
        self._chunks_to_be_updated["remove"].update(
            {k: v["text"] for k, v in files.items()}
        )

    def chunk_to_be_added(self, files: Dict) -> None:
        self._chunks_to_be_updated["add"].update(
            {k: v["text"] for k, v in files.items()}
        )

    def _chunk_files_with_indexing(self) -> None:
        text_chunk = TextChunk(chunk_size=self.chunk_size)
        code_chunk = CodeChunk(chunk_size=self.chunk_size)

        allowed_paths = self._extract_allowed_file_paths()

        for path_info in tqdm(allowed_paths, desc="Chunking Documents"):
            syntax, path = list(path_info.items())[0]
            try:
                existing_data = self._indexing_storage.get("files").get(str(path))
                if existing_data:
                    hash = DataManager.generate_hash(path)
                    existing_hash = existing_data.get("hash")
                    if existing_hash is not None and existing_hash == hash:
                        continue
                    else:
                        self._indexing_storage[
                            "files"][str(path)]["hash"] = hash
                        self.chunk_to_be_removed(
                            self._indexing_storage[
                                "files"][str(path)]["chunks"]
                        )
                else:
                    self._indexing_storage["files"][str(path)] = {}
                    self._indexing_storage[
                        "files"][str(path)]["hash"] =\
                            DataManager.generate_hash(path)

                data = DataManager.load_data(str(path))
                if syntax in FileFormats.CODING_LANGUAGES.value:
                    code_chunk.set_language(syntax)
                    data_chunks = code_chunk.process(
                        document=data,
                        chunk_overlap=self.code_overlap)
                else:
                    data_chunks = text_chunk.process(
                        document=str(data),
                        chunk_overlap=self.txt_overlap)

                self._indexing_storage["files"][str(path)]["chunks"] = {}
                for doc in data_chunks:
                    start = doc.metadata["start_index"]
                    self._indexing_storage[
                        "files"][str(path)]["chunks"].update(
                        {   f"{path}:{start}:{start + len(doc.page_content)}":
                            {
                            "first": start,
                            "last": start + len(doc.page_content),
                            "text": doc.page_content,
                        }})
                self.chunk_to_be_added(
                    self._indexing_storage["files"][str(path)]["chunks"])

            except Exception as e:
                raise Exception(f"{e}")

        self._rebuild_flat_lists()

    def _rebuild_flat_lists(self) -> None:
        self._all_chunks = []
        self._all_chunk_txt = []
        for p in sorted(self._indexing_storage["files"]):
            for id, c in self._indexing_storage["files"][p]["chunks"].items():
                # id, c = list(info.items())[0]
                self._all_chunks.append(MinimalSource(
                    file_path=p,
                    first_character_index=c["first"],
                    last_character_index=c["last"]))
                self._all_chunk_txt.append(c["text"])
                self._all_chunks_with_id.update(
                    {id: c["text"]}
                )


    def _chunk_files(self) -> None:
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
                            document=str(data),
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
