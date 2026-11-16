from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path
from pydantic import BaseModel
from typing import List, Tuple, TYPE_CHECKING, Dict
from src.data_retrieval.helper_classes import (
    DataManager, prepare_storage)
from src.base_patterns import MinimalSource

if TYPE_CHECKING:
    from ..data_retrieval.abstract_classes import RetrieveMethod


class AnswerCacheStructure(BaseModel):
    question_id: str | None = None,
    retrieved_sources_scores: List[float] | None = None
    retrieved_sources: List[MinimalSource] | None = None
    answer: str | None = None


class IndexCache:
    def __init__(self, path: str, chunk_size: int, overlap: int, method: int) -> None:
        self.path = path
        self.settings = {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "method": method
        }
        prepare_storage(self.path)

    def get_file_stats(self, paths: List[Path]) -> Dict:
        stats = {}
        for path in paths:
            stats.update({
                str(path): {
                    "time": os.stat(path).st_mtime,
                    "size": os.stat(path).st_size
                }
            })
        return stats

    def save_stats_digest(self, digest: str) -> None:
        DataManager.save_data(self.path, {"hash": digest})

    def get_stat_digest(self, stats: Dict) -> str:
        canonical = json.dumps(
            {"stats": stats, "settings": self.settings},
            sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def load_digest(self) -> str | None:
        try:
            old_digest = DataManager.load_data(self.path)
            old_hash = old_digest.get("hash")
            return old_hash
        except Exception:
            return None

    def has_file_stats_changed(
            self, paths: List[Path]) -> Tuple[Dict, bool]:
        current_stats = self.get_file_stats(paths)
        current_digest = self.get_stat_digest(current_stats)
        old_hash = self.load_digest()
        return current_digest, current_digest != old_hash

    @staticmethod
    def get_only_paths(paths: Dict[str, List[Path]]) -> List[Path]:
        flat_path = []
        for _, path in paths.items():
            flat_path.extend(path)
        return flat_path


class AnswerCache:
    def __init__(self, path: str):
        self.path = path
        prepare_storage(self.path)
        try:
            self._answer_from_cache =\
                DataManager.load_data(self.path)
        except Exception:
            self._answer_from_cache = {}

    def get_sources_from_cache(
        self,
        question: str,
        resource_numbers: int,
        method: str,
            ) -> Tuple[str, None | AnswerCacheStructure]:
        question_hash = DataManager.generate_hash_from_text(
            f"{question}:{resource_numbers}:{method}")
        retrieved_resources = self._answer_from_cache.get(question_hash)
        if isinstance(retrieved_resources, Dict):
            if retrieved_resources is not None:
                resources = AnswerCacheStructure(**retrieved_resources)
                return question_hash, resources
            return question_hash, None
        return question_hash, retrieved_resources

    def update_answer_cache(
        self,
        hash: str,
        scores=None,
        resources=None,
        answer=None,
        question_id=None
            ) -> None:
        self._answer_from_cache.update(
            {hash: AnswerCacheStructure(
                question_id=question_id,
                retrieved_sources_scores=scores,
                retrieved_sources=resources,
                answer=answer
            )}
        )
        # print(self._answer_from_cache)

    def save_to_answer_cache(self):
        DataManager.save_data(
            self.path, self._answer_from_cache
        )
