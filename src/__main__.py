"""Command-line interface for the RAG pipeline (built with Python Fire).

Commands: index, search, search_dataset, answer, answer_dataset, evaluate.
Every command reports errors with a short message and exit code 1,
never with a traceback.
"""
import functools
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, TypeVar, cast

import fire

from src.data_retrieval.helper_classes import (
    FilesInDir,
    DataManager)
from src.data_retrieval.abstract_classes import Update
from src.data_retrieval.chunk_data import SplitDataByChunks
from src.data_retrieval.abstract_classes import Retriever, RetrieveMethod
from src.data_retrieval.lexical_retriever import (
    BM25Retriever, BatchSourceRetriever)
from src.answer_generation.models.qwen3__0_6B import SmallLLM
from src.answer_generation.pre_prompt import InitialPromptGenerator
from src.answer_generation.answer import (
    AnswerGenerator, BatchAnswerGenerator)
from src.validator.resource_validator import (
    RecallN, AnswerValidator, BatchAnswerValidator)
from src.parsing.parse_rag_dataset import RagDatasetParser
from src.base_patterns import StudentSearchResults
from .data_retrieval.semantic_retriever import MiniLML6Retriever
from .data_retrieval.hybrid_retriever import HybridRetriever
from .speed_optimization.cache import AnswerCache, IndexCache
from src.all_paths import (
    INDEX_CACHE, ANSWER_CACHE, PROCESSED_DATA_FOLDER,
    VLLM_FOLDER, DATA_CHUNK_FOLDER, RETRIEVED_SOURCES,
    RETRIEVED_SOURCES_WITH_ANSWER
)


DEVICE_TYPE = "cpu"
MIN_QUERY_LENGTH = 3
VALID_METHODS = {m.value for m in RetrieveMethod}

F = TypeVar("F", bound=Callable[..., Any])


class CLIError(Exception):
    """A user-facing error: bad input or a missing prerequisite."""


def handled(func: F) -> F:
    """Turn any exception raised by a command into a clean exit.

    Args:
        func: A CLI command.

    Returns:
        The wrapped command. On failure it prints `Error: <message>`
        and exits with code 1 instead of showing a traceback.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except CLIError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")
        sys.exit(1)
    return cast(F, wrapper)


def _check_k(k: Any) -> int:
    """Validate `k`.

    Args:
        k: Value received from the command line.

    Returns:
        `k` if it is a positive integer.

    Raises:
        CLIError: If `k` is not a positive integer.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise CLIError(f"k must be a positive integer, got {k!r}")
    return int(k)


def _check_text(value: Any, name: str) -> str:
    """Validate a question text.

    Args:
        value: Value received from the command line.
        name: Option name, used in the error message.

    Returns:
        The text if it has enough non-space characters.

    Raises:
        CLIError: If it is not a string or is too short.
    """
    if not isinstance(value, str) or len(value.strip()) < MIN_QUERY_LENGTH:
        raise CLIError(
            f"{name} must contain at least {MIN_QUERY_LENGTH} "
            "non-space characters")
    return value


def _check_method(method: Any) -> str:
    """Validate the retrieval method name.

    Args:
        method: Value received from the command line.

    Returns:
        The method name if it is known.

    Raises:
        CLIError: If the method is unknown.
    """
    if method not in VALID_METHODS:
        raise CLIError(
            f"unknown method {method!r}. Choose from "
            f"{sorted(VALID_METHODS)}")
    return str(method)


def _require_file(path: Any, what: str) -> str:
    """Check that an input file exists.

    Args:
        path: Path received from the command line.
        what: Option name, used in the error message.

    Returns:
        The path as a string.

    Raises:
        CLIError: If the path is not an existing file.
    """
    if not isinstance(path, str) or not Path(path).is_file():
        raise CLIError(f"{what}: file not found: {path!r}")
    return path


def _index_ready() -> bool:
    """Tell whether every artifact written by `index` exists.

    Returns:
        True if the chunk files and a non-empty index folder exist.
    """
    chunks = Path(DATA_CHUNK_FOLDER)
    processed = Path(PROCESSED_DATA_FOLDER)
    return ((chunks / "data.json").is_file()
            and (chunks / "update.json").is_file()
            and processed.is_dir()
            and any(processed.iterdir()))


def _require_index() -> None:
    """Stop with a clear message if `index` has not been run.

    Raises:
        CLIError: If the index artifacts are missing.
    """
    if not _index_ready():
        raise CLIError("index not found. Run `index` first.")


class CLI:
    """Commands exposed through Fire (`python -m src <command>`)."""

    def __init__(self) -> None:
        """Create the CLI with every heavy resource unloaded."""
        self._retriever: Retriever | None = None
        self._all_chunks: Dict | None = None
        self._all_updates: Update | None = None
        self._answer_generator: AnswerGenerator | None = None
        self._answer_cache: AnswerCache | None = None
        self._index_cache: IndexCache | None = None

    # ------------------------- helpers -------------------------

    def _load_index_cache(
            self, chunk_size: int, overlap: int, method: str
            ) -> IndexCache:
        """Return the index cache, creating it on first use.

        Args:
            chunk_size: Maximum chunk size.
            overlap: Chunk overlap.
            method: Retrieval method name.

        Returns:
            The shared `IndexCache` instance.
        """
        if self._index_cache is None:
            self._index_cache = IndexCache(
                path=INDEX_CACHE,
                chunk_size=chunk_size,
                overlap=overlap,
                method=method)
        return self._index_cache

    def _get_answer_cache(self) -> AnswerCache:
        """Return the answer cache, creating it on first use."""
        if self._answer_cache is None:
            self._answer_cache = AnswerCache(ANSWER_CACHE)
        return self._answer_cache

    def _get_retriever(
            self, method: str, load: bool = True) -> Retriever:
        """Create the retriever for `method` once and reuse it.

        Args:
            method: One of the values of `RetrieveMethod`.
            load: Load the saved index when True. Pass False when the
                index is about to be (re)built.

        Returns:
            The retriever instance.
        """
        if self._retriever is None:
            retriever_class: type[Retriever]
            if method == RetrieveMethod.HYBRID.value:
                print("Using hybrid retriever....")
                retriever_class = HybridRetriever
            elif method == RetrieveMethod.SEMANTIC.value:
                print("Using semantic retriever....")
                retriever_class = MiniLML6Retriever
            else:
                print("Using lexical retriever....")
                retriever_class = BM25Retriever

            all_chunks, all_updates = SplitDataByChunks.load_from_files(
                DATA_CHUNK_FOLDER)
            self._all_chunks = all_chunks
            self._all_updates = all_updates

            self._retriever = retriever_class(
                data=all_chunks,
                update=all_updates,
                cache=self._get_answer_cache(),
                method=method
            )
            if load:
                self._retriever.load_corpus_index(
                    storage_path=PROCESSED_DATA_FOLDER)
        return self._retriever

    def _get_answer_generator(
            self,
            llm_model: type[SmallLLM],
            all_chunks: Dict,
            method: str
            ) -> AnswerGenerator:
        """Create the answer generator once (loads the LLM).

        Args:
            llm_model: Model class to instantiate.
            all_chunks: Chunk id -> text mapping.
            method: Retrieval method name.

        Returns:
            The shared `AnswerGenerator`.
        """
        if self._answer_generator is None:
            print("Initiating answer generator")
            llm = llm_model(device_type=DEVICE_TYPE)
            self._answer_generator = AnswerGenerator(
                model=llm,
                prompt_generator=InitialPromptGenerator.get_type1_prompt,
                chunked_texts=all_chunks,
                cache=self._get_answer_cache(),
                method=method
                )
        return self._answer_generator

    @staticmethod
    def _report_time(start_time: float) -> None:
        """Print the elapsed time since `start_time`."""
        print(f"Time taken: \033[92m{time.time() - start_time:.3f}s\033[0m")

    # ------------------------- commands -------------------------

    @handled
    def index(
            self, max_chunk_size: int = 2000, method: str = "lexical"
            ) -> None:
        """Chunk the corpus and build the search index.

        Skips all work when no file and no setting changed and the
        saved index still exists.

        Args:
            max_chunk_size: Maximum characters per chunk.
            method: lexical, semantic or hybrid.
        """
        start_time = time.time()
        _check_k(max_chunk_size)
        method = _check_method(method)
        if not Path(VLLM_FOLDER).is_dir():
            raise CLIError(f"corpus folder not found: {VLLM_FOLDER}")

        all_valid_paths = FilesInDir.extract_all_file_paths(
            path=VLLM_FOLDER)
        only_paths = IndexCache.get_only_paths(all_valid_paths)

        overlap = 50 if max_chunk_size >= 500 else 0
        index_cache = self._load_index_cache(
            max_chunk_size, overlap, method)
        digest, info_changed = index_cache.has_file_stats_changed(
            only_paths)

        if info_changed or not _index_ready():
            print("Files or settings changed: rebuilding the index")
            splitter = SplitDataByChunks(
                all_paths=all_valid_paths,
                chunk_size=max_chunk_size,
                txt_overlap=overlap,
                code_overlap=overlap,
                use_indexing=True,
                folder=DATA_CHUNK_FOLDER,
                method=method)
            splitter.chunk_all_files()
            splitter.save_chunked_data()

            retriever = self._get_retriever(method, False)
            retriever.create_and_save_corpus_index(
                storage_path=PROCESSED_DATA_FOLDER)
            index_cache.save_stats_digest(digest)  # always last
            print("Ingestion complete! Indices saved under "
                  f"{PROCESSED_DATA_FOLDER}")
        else:
            print("Index is up to date, nothing to do.")
        self._report_time(start_time)

    @handled
    def search(
            self, query: str, k: int = 10, method: str = "lexical"
            ) -> None:
        """Print the top-k sources for one question.

        Args:
            query: Question text (at least 3 non-space characters).
            k: Number of sources to return.
            method: lexical, semantic or hybrid.
        """
        start_time = time.time()
        query = _check_text(query, "query")
        k = _check_k(k)
        method = _check_method(method)
        _require_index()

        retriever = self._get_retriever(method)
        search_result = retriever.get_matching_chunk(question=query, k=k)
        self._get_answer_cache().save_to_answer_cache()

        print(f"Question: {query}")
        print("Matching Resources")
        for source in search_result.retrieved_sources:
            print(f"{source.file_path}["
                  f"{source.first_character_index}:"
                  f"{source.last_character_index}]")
        self._report_time(start_time)

    @handled
    def search_dataset(
            self,
            dataset_path: str,
            k: int = 10,
            method: str = "lexical",
            save_directory: str = RETRIEVED_SOURCES
            ) -> None:
        """Search every question of a dataset and save the results.

        The output file is `<save_directory>/<dataset file name>`.

        Args:
            dataset_path: JSON file with the questions.
            k: Number of sources per question.
            method: lexical, semantic or hybrid.
            save_directory: Folder for the output JSON.
        """
        start_time = time.time()
        dataset_path = _require_file(dataset_path, "dataset_path")
        k = _check_k(k)
        method = _check_method(method)
        _require_index()

        rag_parser = RagDatasetParser(
            answered_question_paths=[],
            unanswered_question_paths=[dataset_path]
        )
        rag_parser.extract_data_from_paths()
        questions = rag_parser.get_unanswered_data()
        all_questions: List[Dict[str, str]] = [
            {"question": q.question, "question_id": q.question_id}
            for q in questions.rag_questions
        ]

        retriever = self._get_retriever(method)
        batch_source_retriever = BatchSourceRetriever(
            retriever=retriever, k=k)
        student_answers = batch_source_retriever.process_batch(
            questions=all_questions)

        output_file = Path(save_directory) / Path(dataset_path).name
        DataManager.save_data(
            storage_path=str(output_file), data=student_answers)
        self._get_answer_cache().save_to_answer_cache()

        print(f"\033[92m{len(all_questions)}\033[0m questions received, "
              f"\033[92m{len(student_answers.search_results)}\033[0m "
              "retrieved")
        print("Saved student_search_results to "
              f"\033[92m{output_file}\033[0m")
        self._report_time(start_time)

    @handled
    def answer(
            self, question: str, k: int = 10, method: str = "lexical"
            ) -> None:
        """Retrieve sources for one question and generate an answer.

        Args:
            question: Question text (at least 3 non-space characters).
            k: Number of sources given to the model.
            method: lexical, semantic or hybrid.
        """
        start_time = time.time()
        question = _check_text(question, "question")
        k = _check_k(k)
        method = _check_method(method)
        _require_index()

        retriever = self._get_retriever(method)
        search_result = retriever.get_matching_chunk(
            question=question, k=k)

        answer_generator = self._get_answer_generator(
            SmallLLM, cast(Dict, self._all_chunks), method)
        generated_answer = answer_generator.generate_answer(
            search_result=search_result, tokens_limit=100, refiner=None)
        self._get_answer_cache().save_to_answer_cache()

        print(f"Question: {question}")
        print(f"Answer:\n\033[92m{generated_answer.answer}\033[0m")
        self._report_time(start_time)

    @handled
    def answer_dataset(
            self,
            student_search_results_path: str,
            save_directory: str = RETRIEVED_SOURCES_WITH_ANSWER,
            method: str = "lexical",
            ) -> None:
        """Generate an answer for every question of a results file.

        The output file is
        `<save_directory>/<results file name>`.

        Args:
            student_search_results_path: JSON written by search_dataset.
            save_directory: Folder for the output JSON.
            method: lexical, semantic or hybrid.
        """
        start_time = time.time()
        student_search_results_path = _require_file(
            student_search_results_path, "student_search_results_path")
        method = _check_method(method)
        _require_index()

        data = DataManager.load_data(student_search_results_path)
        if not isinstance(data, dict):
            raise CLIError("results file must contain a JSON object")
        student_search_results = StudentSearchResults(**data)

        if self._all_chunks is None:
            self._get_retriever(method)
        answer_generator = self._get_answer_generator(
            SmallLLM, cast(Dict, self._all_chunks), method)

        batch_answer_generator = BatchAnswerGenerator(
            generator=answer_generator, tokens_limit=100, refiner=None)
        answers = batch_answer_generator.process_batch(
            search_results=student_search_results)

        output_file = (
            Path(save_directory) / Path(student_search_results_path).name)
        DataManager.save_data(storage_path=str(output_file), data=answers)
        self._get_answer_cache().save_to_answer_cache()

        print("Saved student_search_results_and_answer to "
              f"{output_file}")
        self._report_time(start_time)

    @handled
    def evaluate(
            self,
            student_search_results_path: str,
            dataset_path: str,
            k: int = 10) -> None:
        """Report recall@k of a results file against ground truth.

        For your own testing. The official score comes from the
        moulinette.

        Args:
            student_search_results_path: JSON written by search_dataset.
            dataset_path: Answered-questions (ground truth) JSON.
            k: Number of top results to score.
        """
        start_time = time.time()
        student_search_results_path = _require_file(
            student_search_results_path, "student_search_results_path")
        dataset_path = _require_file(dataset_path, "dataset_path")
        k = _check_k(k)

        rag_parser = RagDatasetParser(
            answered_question_paths=[dataset_path],
            unanswered_question_paths=[]
        )
        rag_parser.extract_data_from_paths()
        ground_truth = rag_parser.get_ground_truth()

        batch_validator = BatchAnswerValidator(
            validator=AnswerValidator(
                ground_truth_map=ground_truth, recall=RecallN, n=k))

        data = DataManager.load_data(student_search_results_path)
        if not isinstance(data, dict):
            raise CLIError("results file must contain a JSON object")
        scores = batch_validator.process_batch(
            answers=StudentSearchResults(**data))
        if not scores:
            raise CLIError("no questions to evaluate")

        recall = 100 * sum(scores) / len(scores)
        print(f"Recall@{k}: \033[92m{recall:.1f}%\033[0m "
              f"({len(scores)} questions)")
        self._report_time(start_time)


if __name__ == "__main__":
    fire.Fire(CLI)
