"""Chunking of the corpus and incremental bookkeeping of what changed."""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, Language)
from pydantic import (
    BaseModel, Field,
    PrivateAttr, validate_call)
from tqdm import tqdm

from src.base_patterns import MinimalSource
from .abstract_classes import DataChunk, Update
from .helper_classes import (
    DataManager, ExistingPath,
    FileFormats, prepare_storage_folder)


class TextChunk(DataChunk):
    """Chunker for plain text and Markdown."""

    @validate_call
    def process(self, document: str,
                chunk_overlap: int = 0) -> List[Document]:
        """Split text on paragraphs, lines, then words.

        Args:
            document: Full file text.
            chunk_overlap: Characters shared by neighbouring chunks.

        Returns:
            Chunks, each with `start_index` in its metadata.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True)
        docs: List[Document] = text_splitter.create_documents([document])
        return docs


class CodeChunk(DataChunk):
    """Chunker that cuts source code at language-aware boundaries.

    Attributes:
        coding_language: Language whose separators are used.
    """

    coding_language: Language = Field(default=Language.PYTHON)

    @validate_call
    def set_language(self, language: str) -> None:
        """Choose the language.

        Args:
            language: A value of `langchain_text_splitters.Language`.
        """
        self.coding_language = Language(language)

    @validate_call
    def process(self, document: str, chunk_overlap: int = 0) -> List[Document]:
        """Split source code on class and function boundaries first.

        Args:
            document: Full file text.
            chunk_overlap: Characters shared by neighbouring chunks.

        Returns:
            Chunks, each with `start_index` in its metadata.
        """
        code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=self.coding_language,
            chunk_size=self.chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True)
        docs: List[Document] = code_splitter.create_documents([document])
        return docs


class SplitDataByChunks(BaseModel):
    """Chunk the corpus, optionally only the files that changed.

    With `use_indexing=True`, a manifest (`indexed_data.json`) stores
    each file's hash and chunks. Only added or changed files are
    re-chunked, and `Update` lists what the vector store must delete
    and add.

    Attributes:
        all_paths: extension -> files, from `FilesInDir`.
        folder: Folder for the manifest, `data.json`, `update.json`.
        method: Method used for indexing
        chunk_size: Maximum characters per chunk.
        code_overlap: Overlap for code chunks.
        txt_overlap: Overlap for text chunks.
        use_indexing: Enable the incremental mode.
    """

    all_paths: Dict[str, List[Path]]
    folder: str
    method: str
    chunk_size: int = Field(gt=20, default=200)
    code_overlap: int = Field(ge=0, default=0)
    txt_overlap: int = Field(ge=0, default=0)
    use_indexing: bool = Field(default=False)

    # Private attributes (default_factory: never shared between instances)
    _method_used: List = PrivateAttr(default_factory=list)
    _resaving_required: bool = PrivateAttr(default=True)
    _indexing_storage_path: str = PrivateAttr(default="")
    _exist_indexing_storage: bool = PrivateAttr(default=False)
    _all_chunks: List[MinimalSource] = PrivateAttr(default_factory=list)
    _all_chunk_txt: List[str] = PrivateAttr(default_factory=list)
    _indexing_storage: Dict = PrivateAttr(default_factory=dict)
    _all_chunks_with_id: Dict = PrivateAttr(default_factory=dict)
    _chunks_to_be_updated: Update = PrivateAttr(default_factory=Update)

    def model_post_init(self, __context: Any) -> None:
        """Load (or create) the manifest when indexing is enabled."""
        if self.use_indexing:
            self._indexing_storage_path = \
                "/".join([self.folder, 'indexed_data.json'])
            self.prepare_indexing_storage()

    def get_all_data(self) -> Tuple[Dict, Update]:
        """Return (chunk id -> text, pending vector-store update)."""
        return self._all_chunks_with_id, self._chunks_to_be_updated

    def prepare_indexing_storage(self) -> None:
        """Load the manifest, or start a fresh one.

        A fresh start (missing or corrupt manifest, or other settings)
        also sets `reset_flag`: the manifest describes what the vector
        store holds, so without it the store cannot be trusted.
        """
        try:
            with open(
                    self._indexing_storage_path, "r",
                    encoding="utf-8", newline="") as fl:
                loaded = json.load(fl)
            if (isinstance(loaded, dict)
                    and isinstance(loaded.get("files"), dict)
                    and isinstance(loaded.get("settings"), dict)
                    and isinstance(loaded.get("indexed_methods"), list)
                    and self.validate_settings(
                        loaded["settings"], loaded["indexed_methods"])):
                self._indexing_storage = loaded
                self._exist_indexing_storage = True
                return
            self._method_used.extend(loaded["indexed_methods"])
        except (OSError, ValueError):  # missing, unreadable, bad JSON
            pass
        self._method_used.append(self.method)
        self._prepare_fresh_storage()
        self._chunks_to_be_updated.reset_flag = True

    def _prepare_fresh_storage(self) -> None:
        """Create an empty in-memory manifest with the current settings."""
        prepare_storage_folder(self.folder)
        self._indexing_storage = {
            "settings": {
                "chunk_size": self.chunk_size,
                "code_overlap": self.code_overlap,
                "txt_overlap": self.txt_overlap,
            },
            "indexed_methods": self._method_used,
            "files": {},
        }

    def validate_settings(
            self, settings: Dict, indexed_methods: List) -> bool:
        """Tell whether saved settings equal the current ones.

        Args:
            settings: The `settings` block of a saved manifest.

        Returns:
            True if chunk size and both overlaps match.
        """
        settings_check = False
        if (self.chunk_size == settings.get("chunk_size")
            and self.code_overlap == settings.get("code_overlap")
                and self.txt_overlap == settings.get("txt_overlap")):
            settings_check = True
        return (settings_check and self.method in indexed_methods)

    def save_chunked_data(self) -> None:
        """Write the manifest, `data.json` and `update.json`."""
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
        """Load the files written by `save_chunked_data`.

        Args:
            storage_folder: Folder holding `data.json`, `update.json`.

        Returns:
            (chunk id -> text, update as a dict).

        Raises:
            OSError: If a file is missing or unreadable.
            ValueError: If a file is not valid JSON.
        """
        all_data_source = "/".join([storage_folder, 'data.json'])
        all_update_source = "/".join([storage_folder, 'update.json'])

        with open(all_data_source, 'r', encoding="utf-8", newline="") as fl:
            all_data = json.load(fl)

        with open(all_update_source, 'r', encoding="utf-8", newline="") as fl:
            all_update = json.load(fl)

        return all_data, all_update

    def chunk_all_files(self) -> None:
        """Chunk everything (full mode) or only what changed."""
        if self.use_indexing:
            self._chunk_files_with_indexing()
        else:
            self._chunk_files()

    def _extract_allowed_file_paths(self) -> List[Dict]:
        """List `{syntax: path}` for every file type we can chunk."""
        allowed_paths = []
        for syntax, paths in self.all_paths.items():
            if syntax in FileFormats.CODING_LANGUAGES.value or \
                    syntax in FileFormats.TXT_LANGUAGES.value:
                for path in paths:
                    allowed_paths.append({syntax: str(path)})
        return allowed_paths

    def chunk_to_be_removed(self, files: Dict) -> None:
        """Mark chunks (id -> {"text": ...}) for deletion."""
        self._chunks_to_be_updated.remove.update(
            {k: v["text"] for k, v in files.items()}
        )

    def chunk_to_be_added(self, files: Dict) -> None:
        """Mark chunks (id -> {"text": ...}) for insertion."""
        self._chunks_to_be_updated.add.update(
            {k: v["text"] for k, v in files.items()}
        )

    def _chunk_one_file(self, syntax: str, path: str,
                        text_chunk: TextChunk,
                        code_chunk: CodeChunk) -> Dict[str, Dict]:
        """Chunk one file into `id -> {first, last, text}`.

        The text is read raw (no JSON parsing), so offsets match the
        file on disk.
        """
        data = DataManager.load_text(path)
        if syntax in FileFormats.CODING_LANGUAGES.value:
            code_chunk.set_language(syntax)
            docs = code_chunk.process(
                document=data, chunk_overlap=self.code_overlap)
        else:
            docs = text_chunk.process(
                document=data, chunk_overlap=self.txt_overlap)
        chunks: Dict[str, Dict] = {}
        for doc in docs:
            start = doc.metadata["start_index"]
            end = start + len(doc.page_content)
            chunks[f"{path}:{start}:{end}"] = {
                "first": start, "last": end, "text": doc.page_content}
        return chunks

    def _chunk_files_with_indexing(self) -> None:
        """Re-chunk added and changed files, drop removed ones.

        Raises:
            Exception: If a file cannot be chunked (names the file).
        """
        text_chunk = TextChunk(chunk_size=self.chunk_size)
        code_chunk = CodeChunk(chunk_size=self.chunk_size)
        files: Dict[str, Dict] = self._indexing_storage["files"]

        allowed_paths = self._extract_allowed_file_paths()
        on_disk = {list(p.values())[0] for p in allowed_paths}

        # Files deleted from disk: their chunks must leave the store.
        for gone in [p for p in files if p not in on_disk]:
            self.chunk_to_be_removed(files[gone].get("chunks", {}))
            del files[gone]

        for path_info in tqdm(allowed_paths, desc="Chunking Documents"):
            syntax, path = list(path_info.items())[0]
            try:
                new_hash = DataManager.generate_hash(path)
                existing = files.get(path)
                if existing and existing.get("hash") == new_hash:
                    continue  # unchanged
                if existing:
                    # Collect the old ids BEFORE overwriting the entry.
                    self.chunk_to_be_removed(existing.get("chunks", {}))
                chunks = self._chunk_one_file(
                    syntax, path, text_chunk, code_chunk)
                files[path] = {"hash": new_hash, "chunks": chunks}
                self.chunk_to_be_added(chunks)
            except Exception as e:
                raise Exception(f"Failed to chunk '{path}': {e}") from e

        self._rebuild_flat_lists()

    def _rebuild_flat_lists(self) -> None:
        """Derive the flat lists from the manifest (never saved)."""
        self._all_chunks = []
        self._all_chunk_txt = []
        self._all_chunks_with_id = {}
        for p in sorted(self._indexing_storage["files"]):
            for chunk_id, c in \
                    self._indexing_storage["files"][p]["chunks"].items():
                self._all_chunks.append(MinimalSource(
                    file_path=p,
                    first_character_index=c["first"],
                    last_character_index=c["last"]))
                self._all_chunk_txt.append(c["text"])
                self._all_chunks_with_id[chunk_id] = c["text"]

    def _chunk_files(self) -> None:
        """Chunk every file from scratch (no manifest, no update)."""
        text_chunk = TextChunk(chunk_size=self.chunk_size)
        code_chunk = CodeChunk(chunk_size=self.chunk_size)

        for syntax, paths in self.all_paths.items():
            if (syntax not in FileFormats.CODING_LANGUAGES.value
                    and syntax not in FileFormats.TXT_LANGUAGES.value):
                continue
            for path in paths:
                try:
                    chunks = self._chunk_one_file(
                        syntax, str(path), text_chunk, code_chunk)
                except Exception as e:
                    raise Exception(
                        f"Failed to chunk '{path}': {e}") from e
                for chunk_id, c in chunks.items():
                    self._all_chunks.append(MinimalSource(
                        file_path=str(path),
                        first_character_index=c["first"],
                        last_character_index=c["last"]))
                    self._all_chunk_txt.append(c["text"])
                    self._all_chunks_with_id[chunk_id] = c["text"]
