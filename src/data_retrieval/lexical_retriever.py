"""Lexical retrieval with BM25 (bm25s), plus the batch search runner."""
import uuid
from typing import Dict, List

import bm25s
from bm25s import BM25
from pydantic import (
    BaseModel, ConfigDict, Field, PrivateAttr, validate_call)
from tqdm import tqdm
from typing_extensions import Annotated

from src.base_patterns import (
    MinimalSearchResults, MinimalSource, StudentSearchResults)
from src.data_retrieval.abstract_classes import Retriever
from src.data_retrieval.helper_classes import ValidatedStoragePath


class BM25Retriever(Retriever):
    """BM25 search over the chunk texts.

    The position of a chunk in the BM25 index equals its position in
    `self.data` (insertion order), so `data` must keep the same order
    at indexing time and at search time.
    """

    _retriever: BM25 = PrivateAttr()
    _keys: List[str] | None = PrivateAttr(default=None)

    def _chunk_keys(self) -> List[str]:
        """Return the chunk ids in index order (built once, then reused)."""
        if self._keys is None:
            self._keys = list(self.data.keys())
        return self._keys

    def create_and_save_corpus_index(
            self, storage_path: ValidatedStoragePath) -> None:
        """Tokenize every chunk, build the BM25 index and save it.

        Args:
            storage_path: Folder where the index is saved.

        Raises:
            TypeError: If `data` is not a dict (chunk id -> text).
            Exception: If saving fails (the cause is chained).
        """
        if not isinstance(self.data, dict):
            raise TypeError(
                f"Unsupported data type: {type(self.data)}. "
                "Valid type is Dict[str, str].")
        corpus_tokens = bm25s.tokenize(
            list(self.data.values()), stopwords='en')
        self._retriever = BM25()
        self._retriever.index(corpus_tokens)
        self._keys = None
        try:
            self._retriever.save(storage_path)
        except Exception as e:
            raise Exception(
                f"Indexed corpus saving failed. Error: {e}") from e

    def load_corpus_index(self, storage_path: ValidatedStoragePath) -> None:
        """Load a saved BM25 index (read only).

        Args:
            storage_path: Folder the index was saved in.

        Raises:
            Exception: If loading fails, or the index does not match the
                current chunk list.
        """
        try:
            self._retriever = BM25.load(storage_path)
        except Exception as e:
            raise Exception(
                f"Indexed corpus loading failed. Error: {e}") from e
        self._keys = None
        # An index built from other chunks would return wrong sources.
        scores = getattr(self._retriever, "scores", None)
        num_docs = scores.get("num_docs") if isinstance(scores, dict) else None
        if num_docs is not None and num_docs != len(self.data):
            raise Exception(
                f"BM25 index has {num_docs} chunks but the chunk file has "
                f"{len(self.data)}. Run `index` again.")

    @validate_call
    def retrieve_from_raw_data(
        self,
        hash: str,
        question: Annotated[str, Field(min_length=3)],
        k: Annotated[int, Field(gt=0)],
        question_id: str | None = None
            ) -> MinimalSearchResults:
        """Search the index (no cache lookup) and store the result.

        Args:
            hash: Cache key for this question.
            question: Question text.
            k: Number of sources wanted (clamped to the corpus size).
            question_id: Id for the result (a new UUID if None).

        Returns:
            Sources and BM25 scores, best first. Empty if the question
            has no usable words (for example only stop words).
        """
        if question_id is None:
            question_id = str(uuid.uuid4())

        k = min(k, len(self._chunk_keys()))
        query_tokens = bm25s.tokenize(
            question, stopwords='en', show_progress=False)
        if k == 0 or not query_tokens.ids[0]:
            return MinimalSearchResults(
                question=question, question_id=question_id,
                retrieved_sources_scores=[], retrieved_sources=[])

        indexes, scores = self._retriever.retrieve(
            query_tokens, k=k, return_as="tuple", show_progress=False)
        source_score = [float(s) for s in scores[0]]  # numpy -> float
        sources = self.get_all_minimal_source(
            [int(i) for i in indexes[0]])
        self.cache.update_answer_cache(
            hash=hash,
            scores=source_score,
            resources=sources,
            question_id=question_id
            )

        return MinimalSearchResults(
            question=question,
            question_id=question_id,
            retrieved_sources_scores=source_score,
            retrieved_sources=sources
        )

    def get_all_minimal_source(self, ids: List) -> List[MinimalSource]:
        """Convert index positions to sources.

        A chunk id has the form `<path>:<first>:<last>`. The path may
        itself contain ':', so the id is split from the right.

        Args:
            ids: Positions in the BM25 index.

        Returns:
            One `MinimalSource` per position.
        """
        keys = self._chunk_keys()
        sources: List[MinimalSource] = []
        for i in ids:
            path, first, last = str(keys[i]).rsplit(":", 2)
            sources.append(MinimalSource(
                file_path=path,
                first_character_index=int(first),
                last_character_index=int(last)))
        return sources


class BatchSourceRetriever(BaseModel):
    """Run a retriever over many questions.

    Attributes:
        retriever: Any retriever.
        k: Number of sources per question.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    retriever: Retriever
    k: int = Field(gt=0, default=1)

    def process_batch(
        self, questions: List[Dict],
            ) -> StudentSearchResults:
        """Search every question.

        A question that is empty, too short or makes the search fail is
        kept with an empty source list, so the output has one entry per
        input question and one bad question cannot stop the batch.

        Args:
            questions: Dicts with `question` and `question_id`.

        Returns:
            The results for the whole batch.
        """
        all_sources: List[MinimalSearchResults] = []
        for item in tqdm(questions):
            question = item.get('question') or ""
            question_id = item.get('question_id') or str(uuid.uuid4())
            try:
                if len(question.strip()) < 3:
                    raise ValueError("question too short")
                all_sources.append(self.retriever.get_matching_chunk(
                    question=question,
                    k=self.k,
                    question_id=question_id
                ))
            except Exception as e:
                print(f"Question '{question_id}' not searched: {e}")
                all_sources.append(MinimalSearchResults(
                    question=question, question_id=question_id,
                    retrieved_sources_scores=[], retrieved_sources=[]))
        return StudentSearchResults(
            search_results=all_sources,
            k=self.k
        )
