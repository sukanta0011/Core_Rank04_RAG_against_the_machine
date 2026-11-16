import uuid
import torch
from pydantic import PrivateAttr, Field, model_validator
from typing import Annotated, List, Tuple
from collections import defaultdict
from sentence_transformers import CrossEncoder
from src.base_patterns import (MinimalSource,
    MinimalSearchResults, StudentSearchResults)
from ..data_retrieval.abstract_classes import Retriever
from ..base_patterns import MinimalSearchResults
from ..data_retrieval.semantic_retriever import MiniLML6Retriever
from ..data_retrieval.lexical_retriever import BM25Retriever


re_ranker = CrossEncoder(
    'cross-encoder/ms-marco-MiniLM-L-6-v2',
    device='cuda' if torch.cuda.is_available() else "cpu")


class HybridRetriever(Retriever):
    _semantic_retriever: MiniLML6Retriever = PrivateAttr()
    _lexical_retriever: BM25Retriever = PrivateAttr()

    @model_validator(mode='after')
    def initialize_sub_retrievers(self) -> 'HybridRetriever':
        self._semantic_retriever = MiniLML6Retriever(
            data=self.data,
            update=self.update,
            cache=self.cache,
            method="semantic"
        )
        self._lexical_retriever = BM25Retriever(
            data=self.data,
            update=self.update,
            cache=self.cache,
            method="lexical"
        )
        return self

    def create_and_save_corpus_index(self, storage_path):
        print("Spacial indexing under process")
        self._semantic_retriever.create_and_save_corpus_index(storage_path)
        print("Lexical indexing under process")
        self._lexical_retriever.create_and_save_corpus_index(storage_path)

    def load_corpus_index(self, storage_path):
        try:
            self._semantic_retriever.load_corpus_index(storage_path)
            self._lexical_retriever.load_corpus_index(storage_path)
        except Exception as e:
            raise Exception(f"Indexed corpus loading failed. Error: {e}")

    def retrieve_from_raw_data(
        self,
        hash: str,
        question: Annotated[str, Field(min_length=3)],
        k: Annotated[int, Field(gt=0)],
        question_id: str | None = None
            ) -> MinimalSearchResults:

        semantic_results = self._semantic_retriever.get_matching_chunk(
            question=question,
            question_id=question_id,
            k=k
        )

        lexical_results = self._lexical_retriever.get_matching_chunk(
            question=question,
            question_id=question_id,
            k=k
        )

        lexical_keys = get_keys_from_source(lexical_results.retrieved_sources)
        semantic_keys = get_keys_from_source(
            semantic_results.retrieved_sources)

        new_keys, scores = get_rrf_index([lexical_keys, semantic_keys])

        new_resources = self._semantic_retriever.get_all_minimal_source(
            new_keys[:k]
        )

        self.cache.update_answer_cache(
            hash=hash,
            scores=scores[:k],
            resources=new_resources,
            question_id=question_id
            )

        return MinimalSearchResults(
            question=question,
            question_id=question_id,
            retrieved_sources_scores=scores[:k],
            retrieved_sources=new_resources
        )


def get_keys_from_source(sources: List[MinimalSource]) -> List[str]:
    keys = []
    for source in sources:
        keys.append(f"{source.file_path}:{source.first_character_index}:{source.last_character_index}")
    return keys


def get_rrf_index(
        retrieved_sources: List[List[str]],
        flatter_val: int = 60) -> Tuple[List[int], List[float]]:
    fused_score = defaultdict(float)

    for retrieved_source in retrieved_sources:
        for rank, key in enumerate(retrieved_source):
            fused_score[key] += 1 / (
                flatter_val + rank + 1)

    sorted_dict = sorted(
        fused_score.items(), key=lambda x: x[1], reverse=True)

    return (
        [item[0] for item in sorted_dict],
        [item[1] for item in sorted_dict]
        )


if __name__ == "__main__":
    print(get_rrf_index([["a", "b", "c"], ["d", "a", "e"]]))
