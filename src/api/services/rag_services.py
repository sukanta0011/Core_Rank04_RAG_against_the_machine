"""Service that owns the retriever and the answer generator for the API."""
import threading
from pathlib import Path
from typing import Dict

from src.all_paths import (
    ANSWER_CACHE, DATA_CHUNK_FOLDER, PROCESSED_DATA_FOLDER)
from src.answer_generation.answer import AnswerGenerator
from src.answer_generation.models.qwen3__0_6B import SmallLLM
from src.answer_generation.pre_prompt import InitialPromptGenerator
from src.data_retrieval.abstract_classes import Retriever, Update
from src.data_retrieval.chunk_data import SplitDataByChunks
from src.data_retrieval.hybrid_retriever import HybridRetriever
from src.data_retrieval.lexical_retriever import BM25Retriever
from src.data_retrieval.semantic_retriever import MiniLML6Retriever
from src.speed_optimization.cache import AnswerCache

METHODS = {
    "lexical": BM25Retriever,
    "semantic": MiniLML6Retriever,
    "hybrid": HybridRetriever,
}


class IndexNotFoundError(Exception):
    """Raised when `index` has not been run."""


class RAGService:
    """Load the index once, load the LLM on first use, share both.

    Attributes:
        method: lexical, semantic or hybrid.
        device: Device for the language model.
        retriever: Ready after `initialize_resources`.
        answer_generator: Created lazily by `get_answer_generator`.
        cache: Retrieval cache shared by all requests.
    """

    def __init__(self, method: str = "lexical", device: str = "cpu") -> None:
        """Create an empty service (nothing is loaded yet).

        Args:
            method: Retrieval method to serve.
            device: "cpu" or "cuda" for the language model.

        Raises:
            ValueError: If the method is unknown.
        """
        if method not in METHODS:
            raise ValueError(
                f"unknown method {method!r}, choose from {sorted(METHODS)}")
        self.method = method
        self.device = device
        self.retriever: Retriever | None = None
        self.answer_generator: AnswerGenerator | None = None
        self.all_chunks: Dict[str, str] = {}
        self.cache: AnswerCache | None = None
        self._llm_lock = threading.Lock()

    def initialize_resources(self) -> None:
        """Load the chunks and the saved index (the fast part).

        Raises:
            IndexNotFoundError: If `index` has not been run.
        """
        if self.retriever is not None:
            return
        chunks = Path(DATA_CHUNK_FOLDER)
        if not ((chunks / "data.json").is_file()
                and Path(PROCESSED_DATA_FOLDER).is_dir()):
            raise IndexNotFoundError(
                "index not found. Run `index` first.")

        all_chunks, _ = SplitDataByChunks.load_from_files(
            DATA_CHUNK_FOLDER)
        self.all_chunks = all_chunks
        self.cache = AnswerCache(ANSWER_CACHE)

        retriever = METHODS[self.method](
            data=all_chunks,
            update=Update(),  # serving never modifies the index
            cache=self.cache,
            method=self.method,
        )
        retriever.load_corpus_index(storage_path=PROCESSED_DATA_FOLDER)
        self.retriever = retriever

    def get_answer_generator(self) -> AnswerGenerator:
        """Return the generator, loading the LLM on the first call.

        Returns:
            The shared `AnswerGenerator`.
        """
        with self._llm_lock:
            if self.answer_generator is None:
                print("Initiating answer generator")
                if self.cache is None:
                    raise RuntimeError("call initialize_resources first")
                self.answer_generator = AnswerGenerator(
                    model=SmallLLM(device_type=self.device),
                    prompt_generator=(
                        InitialPromptGenerator.get_type1_prompt),
                    chunked_texts=self.all_chunks,
                    cache=self.cache,
                    method=self.method,
                )
        return self.answer_generator

    def save_cache(self) -> None:
        """Write the retrieval cache to disk."""
        if self.cache is not None:
            self.cache.save_to_answer_cache()
