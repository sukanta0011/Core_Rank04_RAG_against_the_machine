"""Caches: skip re-indexing when nothing changed, reuse past answers."""
import os
import json
import hashlib
from pathlib import Path
from pydantic import BaseModel, ValidationError
from typing import Any, List, Tuple, Dict
from src.data_retrieval.helper_classes import (
    DataManager, prepare_storage)
from src.base_patterns import MinimalSource


class AnswerCacheStructure(BaseModel):
    """One cached entry: retrieval results and/or a generated answer.

    Attributes:
        question_id: Id of the question that created the entry.
        retrieved_sources_scores: Scores of the retrieved sources.
        retrieved_sources: The retrieved sources.
        answer: The generated answer, if any.
    """

    question_id: str | None = None
    retrieved_sources_scores: List[float] | None = None
    retrieved_sources: List[MinimalSource] | None = None
    answer: str | None = None


class IndexCache:
    """Detect whether the corpus or the settings changed since last index.

    A digest of every file's (mtime, size) plus the settings is stored.
    If the current digest equals the stored one, indexing can be skipped.
    """

    def __init__(
            self, path: str, chunk_size: int,
            overlap: int, method: str) -> None:
        """Create the cache and make sure its file exists.

        Args:
            path: File that stores the digest.
            chunk_size: Maximum chunk size used for indexing.
            overlap: Chunk overlap used for indexing.
            method: Retrieval method name.
        """
        self.path = path
        self.settings = {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "method": method
        }
        prepare_storage(self.path)

    def get_file_stats(self, paths: List[Path]) -> Dict:
        """Collect modification time and size of every file.

        Args:
            paths: Files to inspect.

        Returns:
            Mapping path -> {"time": mtime, "size": bytes}.
        """
        stats = {}
        for path in paths:
            st = os.stat(path)
            stats[str(path)] = {"time": st.st_mtime, "size": st.st_size}
        return stats

    def save_stats_digest(self, digest: str) -> None:
        """Store the digest in the cache file.

        Call it last, after the index has been built successfully.

        Args:
            digest: Value returned by `has_file_stats_changed`.
        """
        DataManager.save_data(self.path, {"hash": digest})

    def get_stat_digest(self, stats: Dict) -> str:
        """Hash the file stats together with the settings.

        Args:
            stats: Output of `get_file_stats`.

        Returns:
            SHA-256 hex digest, independent of dict ordering.
        """
        canonical = json.dumps(
            {"stats": stats, "settings": self.settings},
            sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def load_digest(self) -> str | None:
        """Return the stored digest.

        Returns:
            The digest, or None if the file is missing, unreadable or
            not in the expected shape.
        """
        try:
            old_digest = DataManager.load_data(self.path)
            if isinstance(old_digest, dict):
                old_hash = old_digest.get("hash")
            return old_hash if isinstance(old_hash, str) else None
        except Exception:
            return None

    def has_file_stats_changed(
            self, paths: List[Path]) -> Tuple[str, bool]:
        """Compare the current corpus and settings with the stored digest.

        Args:
            paths: Every corpus file.

        Returns:
            (current_digest, changed). A missing digest counts as
            changed.
        """
        current_stats = self.get_file_stats(paths)
        current_digest = self.get_stat_digest(current_stats)
        old_hash = self.load_digest()
        return current_digest, current_digest != old_hash

    @staticmethod
    def get_only_paths(paths: Dict[str, List[Path]]) -> List[Path]:
        """Flatten an extension -> files mapping into one list.

        Args:
            paths: Output of `FilesInDir.extract_all_file_paths`.

        Returns:
            All files in one list.
        """
        flat_path = []
        for _, path in paths.items():
            flat_path.extend(path)
        return flat_path


class AnswerCache:
    """Persistent cache of retrieval results and generated answers.

    Entries are keyed by a hash of question, number of sources and
    retrieval method (and optionally the corpus digest).
    """

    def __init__(self, path: str):
        """Load the cache file. A missing or corrupt file means empty.

        Args:
            path: JSON file that stores the cache.
        """
        self.path = path
        prepare_storage(self.path)
        self._answer_from_cache: Dict[str, Any] = {}
        try:
            data = DataManager.load_data(self.path)
            if isinstance(data, dict):
                self._answer_from_cache = data
        except Exception:
            pass

    @staticmethod
    def _as_structure(value: Any) -> AnswerCacheStructure | None:
        """Convert a stored value to an entry, or None if unusable."""
        if isinstance(value, AnswerCacheStructure):
            return value
        if isinstance(value, dict):
            try:
                return AnswerCacheStructure(**value)
            except (ValidationError, TypeError):
                return None
        return None

    def get_sources_from_cache(
        self,
        question: str,
        resource_numbers: int,
        method: str,
        corpus_digest: str = "",
            ) -> Tuple[str, None | AnswerCacheStructure]:
        """Look a question up in the cache.

        Args:
            question: Question text.
            resource_numbers: k, the number of requested sources.
            method: Retrieval method name.
            corpus_digest: Optional digest of the indexed corpus. When
                given, entries from an older index never match. When
                empty, the key is the same as before.

        Returns:
            (key, entry). The entry is None on a miss or a corrupt
            entry. The key is always returned, to be used by
            `update_answer_cache`.
        """
        text = f"{question}:{resource_numbers}:{method}"
        if corpus_digest:
            text += f":{corpus_digest}"
        question_hash = DataManager.generate_hash_from_text(text)
        return question_hash, self._as_structure(
            self._answer_from_cache.get(question_hash))

    def update_answer_cache(
        self,
        hash: str,
        scores: List | None = None,
        resources: List | None = None,
        answer: str | None = None,
        question_id: str | None = None
            ) -> None:
        """Store an entry in memory (call `save_to_answer_cache` to persist).

        Fields passed as None keep the value of an existing entry with
        the same key, so storing an answer does not erase the saved
        sources (and the reverse).

        Args:
            hash: Key returned by `get_sources_from_cache`.
            scores: Scores of the retrieved sources.
            resources: The retrieved sources.
            answer: The generated answer.
            question_id: Id of the question.
        """
        old = self._as_structure(
            self._answer_from_cache.get(hash)) or AnswerCacheStructure()
        self._answer_from_cache[hash] = AnswerCacheStructure(
            question_id=(
                question_id if question_id is not None
                else old.question_id),
            retrieved_sources_scores=(
                [float(s) for s in scores] if scores is not None
                else old.retrieved_sources_scores),
            retrieved_sources=(
                resources if resources is not None
                else old.retrieved_sources),
            answer=answer if answer is not None else old.answer,
        )

    def save_to_answer_cache(self) -> None:
        """Write the whole cache to its JSON file (atomically)."""
        tmp = self.path + ".tmp"
        DataManager.save_data(tmp, self._answer_from_cache)
        os.replace(tmp, self.path)
