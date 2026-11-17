"""Caches: skip re-indexing when nothing changed, reuse past answers."""
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ValidationError

from src.base_patterns import MinimalSource
from src.data_retrieval.helper_classes import (
    DataManager, prepare_storage)


class AnswerCacheStructure(BaseModel):
    """One cached entry: retrieval results and/or a generated answer."""

    question_id: str | None = None
    retrieved_sources_scores: List[float] | None = None
    retrieved_sources: List[MinimalSource] | None = None
    answer: str | None = None


class IndexCache:
    """Detect whether the corpus or the settings changed since last index.

    The digest covers every file's (mtime, size) and the settings.
    Save it LAST, after the index is fully built.
    """

    def __init__(
            self, path: str, chunk_size: int,
            overlap: int, method: str) -> None:
        """Create the cache.

        Args:
            path: File that stores the digest.
            chunk_size: Maximum chunk size used for indexing.
            overlap: Chunk overlap used for indexing.
            method: Retrieval method name (lexical, semantic, hybrid).
        """
        self.path = path
        self.settings = {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "method": method
        }
        prepare_storage(self.path)

    def get_file_stats(self, paths: List[Path]) -> Dict[str, Dict]:
        """Collect modification time and size of every file.

        Args:
            paths: Files to inspect.

        Returns:
            Mapping path -> {"time": mtime, "size": bytes}.
        """
        stats: Dict[str, Dict] = {}
        for path in paths:
            st = os.stat(path)
            stats[str(path)] = {"time": st.st_mtime, "size": st.st_size}
        return stats

    def save_stats_digest(self, digest: str) -> None:
        """Store the digest atomically (no half-written file).

        Args:
            digest: Value returned by `has_file_stats_changed`.
        """
        tmp = self.path + ".tmp"
        DataManager.save_data(tmp, {"hash": digest})
        os.replace(tmp, self.path)

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
        """Return the saved digest, or None if missing or unreadable."""
        try:
            data = DataManager.load_data(self.path)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        value = data.get("hash")
        return value if isinstance(value, str) else None

    def has_file_stats_changed(
            self, paths: List[Path]) -> Tuple[str, bool]:
        """Compare the current corpus and settings with the saved digest.

        Args:
            paths: Every corpus file.

        Returns:
            (current_digest, changed). A missing or corrupt saved digest
            counts as changed.
        """
        current_digest = self.get_stat_digest(self.get_file_stats(paths))
        return current_digest, current_digest != self.load_digest()

    @staticmethod
    def get_only_paths(paths: Dict[str, List[Path]]) -> List[Path]:
        """Flatten an extension -> files mapping into one list."""
        flat_path: List[Path] = []
        for group in paths.values():
            flat_path.extend(group)
        return flat_path


class AnswerCache:
    """Persistent cache of retrieval results and answers.

    The key is built from question, k, method and (optionally) the
    corpus digest, so a re-index makes old entries stop matching.
    """

    def __init__(self, path: str) -> None:
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
    def make_key(
            question: str, resource_numbers: int, method: str,
            corpus_digest: str = "") -> str:
        """Build the cache key.

        Args:
            question: Question text.
            resource_numbers: k, the number of requested sources.
            method: Retrieval method name.
            corpus_digest: Digest of the indexed corpus. Pass
                `IndexCache.load_digest()` so entries from an older
                index never match.

        Returns:
            SHA-256 hex digest of all inputs.
        """
        payload = json.dumps(
            [question, resource_numbers, method, corpus_digest],
            ensure_ascii=False)
        return DataManager.generate_hash_from_text(payload)

    @staticmethod
    def _as_structure(value: Any) -> AnswerCacheStructure | None:
        """Convert a stored value to a structure, or None if unusable."""
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
            ) -> Tuple[str, AnswerCacheStructure | None]:
        """Look a question up.

        Args:
            question: Question text.
            resource_numbers: k.
            method: Retrieval method name.
            corpus_digest: See `make_key`.

        Returns:
            (key, entry). The entry is None on a miss or a corrupt
            entry. On a hit, use the CURRENT question_id, not the
            cached one, since ids differ between datasets.
        """
        key = self.make_key(
            question, resource_numbers, method, corpus_digest)
        return key, self._as_structure(self._answer_from_cache.get(key))

    def update_answer_cache(
        self,
        hash: str,
        scores: Optional[Sequence[float]] = None,
        resources: Optional[List[MinimalSource]] = None,
        answer: Optional[str] = None,
        question_id: Optional[str] = None,
            ) -> None:
        """Add or complete an entry.

        Fields passed as None keep their previous value, so storing an
        answer does not erase the sources saved by the search step.

        Args:
            hash: Key from `get_sources_from_cache` or `make_key`.
            scores: Scores of the retrieved sources.
            resources: Retrieved sources.
            answer: Generated answer.
            question_id: Id of the question.
        """
        old = self._as_structure(self._answer_from_cache.get(hash))
        old = old or AnswerCacheStructure()
        self._answer_from_cache[hash] = AnswerCacheStructure(
            question_id=question_id or old.question_id,
            retrieved_sources_scores=(
                [float(s) for s in scores] if scores is not None
                else old.retrieved_sources_scores),
            retrieved_sources=(
                resources if resources is not None
                else old.retrieved_sources),
            answer=answer if answer is not None else old.answer,
        )

    def save_to_answer_cache(self) -> None:
        """Write the cache to disk atomically."""
        tmp = self.path + ".tmp"
        DataManager.save_data(tmp, self._answer_from_cache)
        os.replace(tmp, self.path)
