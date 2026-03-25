from pathlib import Path
from typing import List, Tuple
from typing_extensions import Annotated
import bm25s
from bm25s import BM25
from pydantic import (
    BaseModel, field_validator,
    PrivateAttr, Field, validate_call)
from .helper_classes import PrepareStorageFolder


class BM25Indexer(BaseModel):
    storage_path: PrepareStorageFolder

    # Private class
    _retriever: BM25 = PrivateAttr()

    def create_corpus_index(self, data=str | List[str]):
        corpus_tokens = bm25s.tokenize(data, stopwords='en')
        self._retriever = BM25(corpus=data)
        self._retriever.index(corpus_tokens, show_progress=True)
        self._retriever.save(self.storage_path)

    def load_corpus_index(self):
        self._retriever = BM25.load(self.storage_path, load_corpus=True)

    @validate_call
    def get_matching_chunk(
        self,
        query: Annotated[str, Field(min_length=3)],
        k: Annotated[int, Field(gt=0)]
            ) -> Tuple[List[int], List[float]]:

        if not hasattr(self, '_retriever'):
            self.load_corpus_index()
        query_tokens = bm25s.tokenize(query)
        index, scores = self._retriever.retrieve(
            query_tokens, k=k, return_as="tuple")
        return (index[0], scores[0])


def bm25s_test():
    corpus = [
        "a cat is a feline and likes to purr",
        "a dog is the human's best friend and loves to play",
        "a bird is a beautiful animal that can fly unlike human",
    ]

    # Tokenize the corpus and index it
    corpus_tokens = bm25s.tokenize(corpus)
    # print(corpus_tokens)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    # You can now search the corpus with a query
    query = "how is the relation of dog and human?"
    query_tokens = bm25s.tokenize(query)
    index, score = retriever.retrieve(query_tokens, k=2, return_as="tuple")
    print(f"Best result score: {index[0]}")

    # Happy with your index? Save it for later...
    retriever.save("bm25s_index_animals")

    # ...and load it when needed
    # ret_loaded = bm25s.BM25.load("bm25s_index_animals", load_corpus=True)
    # print(ret_loaded.corpus)


if __name__ == "__main__":
    bm25s_test()
