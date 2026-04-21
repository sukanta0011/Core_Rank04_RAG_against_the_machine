import uuid
import torch
from pydantic import PrivateAttr, Field, model_validator
from typing import Annotated, List, Tuple
from collections import defaultdict
from sentence_transformers import CrossEncoder
from ..data_retrieval.abstract_classes import Retriever
from ..base_patterns import MinimalSearchResults
from ..data_retrieval.semantic_retriever import MiniLML6Retriever
from ..data_retrieval.lexical_retriever import BM25Retriever


re_ranker = CrossEncoder(
    'cross-encoder/ms-marco-MiniLM-L-6-v2',
    device='cuda' if torch.cuda.is_available() else "cpu")


class HybridRetriever(Retriever):
    _special_retriever: MiniLML6Retriever = PrivateAttr()
    _lexical_retriever: BM25Retriever = PrivateAttr()

    @model_validator(mode='after')
    def initialize_sub_retrievers(self) -> 'HybridRetriever':
        self._special_retriever = MiniLML6Retriever(
            data=self.data,
            all_minimal_resource=self.all_minimal_resource
        )
        self._lexical_retriever = BM25Retriever(
            data=self.data,
            all_minimal_resource=self.all_minimal_resource
        )
        return self

    def create_and_save_corpus_index(self, storage_path):
        print("Spacial indexing under process")
        self._special_retriever.create_and_save_corpus_index(storage_path)
        print("Lexical indexing under process")
        self._lexical_retriever.create_and_save_corpus_index(storage_path)

    def load_corpus_index(self, storage_path):
        try:
            self._special_retriever.load_corpus_index(storage_path)
            self._lexical_retriever.load_corpus_index(storage_path)
        except Exception as e:
            raise Exception(f"Indexed corpus loading failed. Error: {e}")

    def get_matching_chunk(
        self,
        question: Annotated[str, Field(min_length=3)],
        k: Annotated[int, Field(gt=0)],
        question_id: str | None = None
            ) -> MinimalSearchResults:

        spatial_results = self._special_retriever.get_matching_chunk(
            question=question,
            question_id=question_id,
            k=2*k
        )

        lexical_results = self._lexical_retriever.get_matching_chunk(
            question=question,
            question_id=question_id,
            k=2*k
        )

        if question_id is None:
            question_id = str(uuid.uuid4())

        # retrieve top k index from both search
        indexes, scores = get_rrf_index(
            [spatial_results.retrieved_sources_indexes,
             lexical_results.retrieved_sources_indexes])

        # indexes = []
        # indexes.extend(spatial_results.retrieved_sources_indexes)
        # indexes.extend(lexical_results.retrieved_sources_indexes)

        indexes, scores = re_rank_results(
            query=question,
            chunks=[self.data[i] for i in indexes[:k]],
            indexes=indexes[:k]
        )

        return MinimalSearchResults(
            question_str=question,
            question_id=question_id,
            retrieved_sources_indexes=indexes[:k],
            retrieved_sources_scores=scores[:k],
            retrieved_sources=[self.all_minimal_resource[i]
                               for i in indexes[:k]]
        )


def get_rrf_index(
        retrieved_sources: List[List[int]],
        flatter_val: int = 60) -> Tuple[List[int], List[float]]:
    fused_score = defaultdict(float)

    for retrieved_source in retrieved_sources:
        for rank, index in enumerate(retrieved_source):
            fused_score[index] += 1 / (
                flatter_val + rank + 1)

    sorted_dict = sorted(
        fused_score.items(), key=lambda x: x[1], reverse=True)

    return (
        [item[0] for item in sorted_dict],
        [item[1] for item in sorted_dict]
        )


def re_rank_results(
        query: str, chunks: List[str],
        indexes: List[int]) -> Tuple[List[int], List[float]]:

    pairs = [[query, chunk] for chunk in chunks]
    scores = re_ranker.predict(pairs)

    scored_index = sorted(
        zip(indexes, scores), key=lambda x: x[1], reverse=True)
    return (
        [item[0] for item in scored_index],
        [item[1] for item in scored_index]
        )


if __name__ == "__main__":
    print(get_rrf_index([[1, 10, 2], [5, 1, 20]]))
