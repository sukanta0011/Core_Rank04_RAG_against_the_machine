"""Hybrid retrieval: fuse lexical (BM25) and semantic (Chroma) rankings."""
from collections import defaultdict
from typing import Annotated, Dict, List, Tuple

from pydantic import Field, PrivateAttr, model_validator

from src.base_patterns import MinimalSearchResults, MinimalSource
from ..data_retrieval.abstract_classes import Retriever
from ..data_retrieval.lexical_retriever import BM25Retriever
from ..data_retrieval.semantic_retriever import MiniLML6Retriever

RRF_CONSTANT = 60


class HybridRetriever(Retriever):
    """Combine BM25 and embedding search with Reciprocal Rank Fusion.

    Attributes:
        pool_factor: Each sub-retriever returns `k * pool_factor`
            candidates before fusion. 1 reproduces the previous
            behaviour; a larger pool gives the fusion more to choose from.
    """

    pool_factor: int = Field(default=1, ge=1)

    _semantic_retriever: MiniLML6Retriever = PrivateAttr()
    _lexical_retriever: BM25Retriever = PrivateAttr()

    @model_validator(mode='after')
    def initialize_sub_retrievers(self) -> 'HybridRetriever':
        """Create the two sub-retrievers on the same data and cache."""
        self._semantic_retriever = MiniLML6Retriever(
            data=self.data,
            update=self.update,
            cache=self.cache,
            method="semantic",
        )
        self._lexical_retriever = BM25Retriever(
            data=self.data,
            update=self.update,
            cache=self.cache,
            method="lexical",
        )
        return self

    def create_and_save_corpus_index(self, storage_path: str) -> None:
        """Build both indexes.

        Args:
            storage_path: Folder where both indexes are stored.
        """
        print("Semantic indexing under process")
        self._semantic_retriever.create_and_save_corpus_index(storage_path)
        print("Lexical indexing under process")
        self._lexical_retriever.create_and_save_corpus_index(storage_path)

    def load_corpus_index(self, storage_path: str) -> None:
        """Open both saved indexes.

        Args:
            storage_path: Folder where both indexes were stored.

        Raises:
            Exception: If either index cannot be loaded.
        """
        try:
            self._semantic_retriever.load_corpus_index(storage_path)
            self._lexical_retriever.load_corpus_index(storage_path)
        except Exception as e:
            raise Exception(
                f"Indexed corpus loading failed. Error: {e}") from e

    def retrieve_from_raw_data(
        self,
        hash: str,
        question: Annotated[str, Field(min_length=3)],
        k: Annotated[int, Field(gt=0)],
        question_id: str | None = None
            ) -> MinimalSearchResults:
        """Search both indexes, fuse the rankings, keep the top k.

        Args:
            hash: Cache key for this question.
            question: Question text.
            k: Number of sources to return.
            question_id: Id for the result.

        Returns:
            Fused result. Scores are RRF scores, not BM25 or distances.
        """
        pool = k * self.pool_factor
        semantic = self._semantic_retriever.get_matching_chunk(
            question=question, question_id=question_id, k=pool)
        lexical = self._lexical_retriever.get_matching_chunk(
            question=question, question_id=question_id, k=pool)

        # Keep the sources we already have; no id parsing needed.
        by_key: Dict[str, MinimalSource] = {}
        for source in (list(lexical.retrieved_sources)
                       + list(semantic.retrieved_sources)):
            by_key.setdefault(get_key(source), source)

        keys, scores = get_rrf_index([
            [get_key(s) for s in lexical.retrieved_sources],
            [get_key(s) for s in semantic.retrieved_sources],
        ])
        keys, scores = keys[:k], scores[:k]
        sources = [by_key[key] for key in keys]

        self.cache.update_answer_cache(
            hash=hash, scores=scores, resources=sources,
            question_id=question_id)

        return MinimalSearchResults(
            question=question,
            question_id=question_id,
            retrieved_sources_scores=scores,
            retrieved_sources=sources
        )


def get_key(source: MinimalSource) -> str:
    """Identity of a chunk: `path:first:last`."""
    return (f"{source.file_path}:{source.first_character_index}:"
            f"{source.last_character_index}")


def get_keys_from_source(sources: List[MinimalSource]) -> List[str]:
    """Return the identity key of each source, in order."""
    return [get_key(s) for s in sources]


def get_rrf_index(
        retrieved_sources: List[List[str]],
        flatter_val: int = RRF_CONSTANT) -> Tuple[List[str], List[float]]:
    """Reciprocal Rank Fusion of several ranked lists.

    Each item scores `1 / (flatter_val + rank + 1)` per list it appears
    in. Only ranks are used, never the raw scores. Ties keep the order
    of first appearance, so the result is deterministic.

    Args:
        retrieved_sources: Ranked key lists, best first.
        flatter_val: RRF constant (60 is the usual value).

    Returns:
        (keys, scores) sorted by fused score, best first.
    """
    fused_score: Dict[str, float] = defaultdict(float)
    for ranking in retrieved_sources:
        for rank, key in enumerate(ranking):
            fused_score[key] += 1 / (flatter_val + rank + 1)
    ordered = sorted(
        fused_score.items(), key=lambda x: x[1], reverse=True)
    return [k for k, _ in ordered], [s for _, s in ordered]
