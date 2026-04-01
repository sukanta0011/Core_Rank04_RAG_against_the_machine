from typing import List
import uuid
from typing_extensions import Annotated
import bm25s
from bm25s import BM25
from pydantic import PrivateAttr, Field, validate_call
from .helper_classes import PrepareStorageFolder
from src.data_retrieval.abstract_classes import Retrieve
from src.base_patterns import MinimalSearchResults

class BM25Retrieve(Retrieve):
    # Private class
    _retriever: BM25 = PrivateAttr()

    def create_corpus_index(self):
        # corpus_tokens: List[List[str]]
        if isinstance(self.data, list):
            corpus_tokens = bm25s.tokenize(self.data, stopwords='en')
            self._retriever = BM25(corpus=self.data)
            self._retriever.index(corpus_tokens)
            self._retriever.save(self.storage_path)
        else:
            raise TypeError(f"Unsupported data types: {type(self.data)},"
                            "Valid types are List[str].")

    def load_corpus_index(self):
        self._retriever = BM25.load(self.storage_path, load_corpus=True)

    @validate_call
    def get_matching_chunk(
        self,
        question: Annotated[str, Field(min_length=3)],
        k: Annotated[int, Field(gt=0)],
        question_id: str| None = None
            ) -> MinimalSearchResults:

        if not hasattr(self, '_retriever'):
            raise AttributeError(
                "Run 'create_corpus_index' or 'load_corpus_index' "
                "before running the 'get_matching_chunk' function")

        if question_id is None:
            question_id = str(uuid.uuid4())

        query_tokens = bm25s.tokenize(question)
        index, scores = self._retriever.retrieve(
            query_tokens, k=k, return_as="tuple")
        
        ids = [item['id'] for item in index[0]]
        
        return MinimalSearchResults(
            question=question,
            question_id=question_id,
            retrieved_sources_indexes=ids,
            retrieved_sources_scores=scores[0],
            retrieved_sources=[self.all_minimal_resource[i] for i in ids]
        )


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
