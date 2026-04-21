import time
import fire
# import torch
from typing import List
from src.data_retrieval.helper_classes import (
    FilesInDir,
    DataManager)
from src.data_retrieval.chunk_data import SplitDataByChunks
from src.data_retrieval.abstract_classes import Retriever
from src.data_retrieval.lexical_retriever import (
    BM25Retriever, BatchSourceRetriever)
from src.answer_generation.models.qwen3__0_6B import SmallLLM
from src.answer_generation.pre_prompt import InitialPromptGenerator
from src.answer_generation.answer import (
    AnswerGenerator, BatchAnswerGenerator)
from src.validator.resource_validator import (
    RecallN, AnswerValidator,
    BatchAnswerValidator, RecallOverlap)
from src.parsing.parse_rag_dataset import RagDatasetParser
from src.validator.resource_validator import Validator
from src.base_patterns import StudentSearchResults
from src.data_retrieval.resource_refiner import ResourceRefiner
from src.data_retrieval.chunk_data import (
    TextChunk, CodeChunk)
# from .data_retrieval.semantic_retriever import MiniLML6Retriever
# from .data_retrieval.hybrid_retriever import HybridRetriever


RETRIEVER = BM25Retriever
LLM = SmallLLM
DEVICE_TYPE = "cpu"


class CLI:
    def __init__(self):
        self._retriever = None
        self._all_chunks = None
        self._answer_generator = None

    def _get_retriever(self) -> Retriever:
        """Helper to initialize resources once."""
        if self._retriever is None:
            # 1. Load Chunks
            all_sources, all_chunks =\
                SplitDataByChunks.load_from_files("data/chunks")
            self._all_chunks = all_chunks

            # 2. Initialize Retriever
            self._retriever = RETRIEVER(
                data=all_chunks,
                all_minimal_resource=all_sources
            )
            # 3. Load Index
            self._retriever.load_corpus_index(storage_path="data/processed/")

        return self._retriever

    def _get_answer_generator(
            self,
            all_chunks: List[str]
            ) -> AnswerGenerator:
        if self._answer_generator is None:
            print("Initiating answer generator")
            # 1. Load llm
            llm = LLM(
                device_type=DEVICE_TYPE)

            # 2. Initialize Answer Generator
            self._answer_generator = AnswerGenerator(
                model=llm,
                prompt_generator=InitialPromptGenerator.get_type1_prompt,
                chunked_texts=all_chunks
                )
        return self._answer_generator

    def _get_refiner(
        self, chunk_size: int, overlap: int, k: int = 5
            ) -> ResourceRefiner:
        return ResourceRefiner(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            k=k,
            text_chunk=TextChunk,
            code_chunk=CodeChunk,
            retriever=RETRIEVER
        )

    def index(self, max_chunk_size: int = 2000) -> None:
        start_time = time.time()

        # Extracting all valid path from the folder
        all_valid_paths = FilesInDir.extract_all_file_paths(
            path='data/raw/vllm-0.10.1')

        # Split the data into chunks
        overlap = 50 if max_chunk_size >= 500 else 0
        splitter = SplitDataByChunks(
            all_paths=all_valid_paths,
            chunk_size=max_chunk_size,
            txt_overlap=overlap,
            code_overlap=overlap)
        splitter.chunk_all_files()
        splitter.save_chunked_data("data/chunks")

        # splitter.save_chunked_data("data/chunks")
        # all_minimal_sources, all_data_chunks = splitter.load_from_files(
        #     "data/chunks")

        all_minimal_sources, all_data_chunks = splitter.get_all_data()

        retriever = RETRIEVER(
            data=all_data_chunks,
            all_minimal_resource=all_minimal_sources)

        retriever.create_and_save_corpus_index(storage_path="data/processed/")

        print("Ingestion complete! Indices saved under data/processed")
        print(f"Time taken: \033[92m{(time.time() - start_time):.3f}s\033[0m")

    def search(self, query: str, k: int = 10):
        start_time = time.time()

        retriever = self._get_retriever()

        search_result = retriever.get_matching_chunk(
            question=query, k=k)

        print(f"Question: {query}")
        print("Matching Resources")
        for source in search_result.retrieved_sources:
            print(f"{source.file_path}")
        print(f"Time taken: \033[92m{(time.time() - start_time):.3f}s\033[0m")

    def search_dataset(
            self,
            dataset_path: str,
            k: int = 10,
            save_directory: str = "data/output/sources.json"
            ) -> None:
        start_time = time.time()

        retriever = self._get_retriever()

        batch_source_retriever = BatchSourceRetriever(
            retriever=retriever,
            k=k
        )

        rag_parser = RagDatasetParser(
            answered_question_paths=[],
            unanswered_question_paths=[dataset_path]
        )
        rag_parser.extract_data_from_paths()
        questions = rag_parser.get_unanswered_data()

        all_questions = []
        # print(questions)
        for question in questions.rag_questions:
            all_questions.append({
                "question": question.question,
                "question_id": question.question_id
            })
        student_answers = batch_source_retriever.process_batch(
            questions=all_questions)

        DataManager.save_data(
            storage_path=save_directory,
            data=student_answers
        )

        print(f"\033[92m{len(all_questions)}\033[0m questions are retrieved")
        print("Saved student_search_results to "
              f"\033[92m{save_directory}\033[0m")
        print(f"Time taken: \033[92m{(time.time() - start_time):.3f}s\033[0m")

    def answer(self, question: str, k: int = 10) -> None:
        start_time = time.time()

        retriever = self._get_retriever()

        search_result = retriever.get_matching_chunk(
            question=question, k=k)

        # Refining the retrieved source

        refiner = self._get_refiner(200, 50, 5)

        # Generate Answer
        answer_generator = self._get_answer_generator(self._all_chunks)

        generated_answer = answer_generator.generate_answer(
            search_result=search_result, tokens_limit=100,
            refiner=refiner
            )

        print(f"Question: {question}")
        print(f"Answer:\n\033[92m{generated_answer.answer}\033[0m")
        print(f"Time taken: \033[92m{(time.time() - start_time):.3f}s\033[0m")

    def answer_dataset(
            self,
            student_search_results_path: str,
            save_directory: str
            ) -> None:
        start_time = time.time()

        data = DataManager.load_data(student_search_results_path)
        student_search_results = StudentSearchResults(**data)

        if self._all_chunks is None:
            self._get_retriever()

        refiner = self._get_refiner(500, 50, 4)

        answer_generator = self._get_answer_generator(self._all_chunks)

        batch_answer_generator = BatchAnswerGenerator(
            generator=answer_generator,
            tokens_limit=500,
            refiner=refiner
        )
        answers = batch_answer_generator.process_batch(
            search_results=student_search_results
        )

        DataManager.save_data(
            storage_path="data/answers/sources_with_answer.json",
            data=answers
        )

        # print(f"\033[92m{len(all_questions)}\033[0m questions are retrieved")
        print(f"Saved student_search_results_and_answer to {save_directory}")
        print(f"Time taken: \033[92m{(time.time() - start_time):.3f}s\033[0m")

    def evaluate(
            self,
            student_answer_path: str,
            dataset_path: str,
            k: int = 10,
            validation_type: Validator = RecallN) -> None:

        start_time = time.time()

        rag_parser = RagDatasetParser(
            answered_question_paths=[dataset_path],
            unanswered_question_paths=[]
        )
        rag_parser.extract_data_from_paths()
        ground_truth = rag_parser.get_ground_truth()

        answer_validator_recall = AnswerValidator(
            ground_truth_map=ground_truth,
            recall=validation_type,
            n=k
        )
        batch_validator = BatchAnswerValidator(
            validator=answer_validator_recall
        )

        data = DataManager.load_data(student_answer_path)
        validation_list = batch_validator.process_batch(
            answers=StudentSearchResults(**data)
        )
        # print(validation_list)
        print(f"Recall@{k}: "
              f"\033[92m{100 * sum(
                validation_list)/len(validation_list)}%\033[0m")
        print(f"Time taken: \033[92m{(time.time() - start_time):.3f}s\033[0m")


def main():
    start_time = time.time()

    all_valid_paths = FilesInDir.extract_all_file_paths(
        path='data/raw/vllm-0.10.1')
    splitter = SplitDataByChunks(
        all_paths=all_valid_paths,
        chunk_size=2000, txt_overlap=50,
        code_overlap=50)
    splitter.chunk_all_files()
    # splitter.save_chunked_data("data/chunks")

    # all_minimal_sources, all_data_chunks = splitter.load_from_files(
    #     "data/chunks")
    # print(len(all_minimal_sources))
    all_minimal_sources, all_data_chunks = splitter.get_all_data()
    print("Data chunking is completed with "
          f"\033[92m{len(all_data_chunks)}\033[0m chunks")
    print(f"Time taken: \033[92m{(time.time() - start_time):.3f}s\033[0m")

    bm25_retriever = BM25Retriever(
        storage_path="data/processed/",
        data=all_data_chunks,
        all_minimal_resource=all_minimal_sources)

    bm25_retriever.create_and_save_corpus_index(storage_path="data/processed/")
    # bm25_retriever.load_corpus_index()

    answered_path = [
        "datasets_public/public/AnsweredQuestions/dataset_code_public.json"
        ]
    unanswered_path = [
        "datasets_public/public/UnansweredQuestions/dataset_docs_public.json"
        ]

    rag_parser = RagDatasetParser(
        answered_question_paths=answered_path,
        unanswered_question_paths=unanswered_path
    )
    rag_parser.extract_data_from_paths()
    ground_truth = rag_parser.get_ground_truth()

    all_questions = []
    for _, data in ground_truth.items():
        all_questions.append({
            "question": data.question,
            "question_id": data.question_id
        })

    # Retrieving the resources for the questions
    batch_source_retriever = BatchSourceRetriever(
        retriever=bm25_retriever,
        k=1
    )
    student_answers = batch_source_retriever.process_batch(
        questions=all_questions)

    DataManager.save_data(
        storage_path="data/answers/sources.json",
        data=student_answers
    )
    print("Answer retrieval is completed "
          f"\033[92m{len(student_answers.search_results)}\033[0m chunks")
    print(f"Time taken: \033[92m{(time.time() - start_time):.3f}s\033[0m")

    # Validating resources Recall@N
    for validator_type in [RecallN, RecallOverlap]:
        answer_validator_recall = AnswerValidator(
            ground_truth_map=ground_truth,
            recall=validator_type,
            n=1
        )
        batch_validator = BatchAnswerValidator(
            validator=answer_validator_recall
        )
        validation_list = batch_validator.process_batch(
            answers=student_answers
        )
        # print(validation_list)
        print("Validation: "
              f"\033[92m{100 * sum(
                validation_list)/len(validation_list)}%\033[0m")
        print(f"Time taken: \033[92m{(time.time() - start_time):.3f}s\033[0m")

    # Using model to get the answer
    llm = SmallLLM(device_type='cpu')
    answer_generator = AnswerGenerator(
        model=llm,
        prompt_generator=InitialPromptGenerator.get_type1_prompt,
        chunked_texts=all_data_chunks
    )
    batch_answer_generator = BatchAnswerGenerator(
        generator=answer_generator,
        tokens_limit=500
    )
    answers = batch_answer_generator.process_batch(
        search_results=student_answers
    )

    DataManager.save_data(
        storage_path="data/answers/sources_with_answer.json",
        data=answers
    )
    print("\033[92mAnswer Generation Completed\033[0m")
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time):.3f}s")


if __name__ == "__main__":
    # main()
    fire.Fire(CLI)

    # try:
    #     fire.Fire(CLI)
    # except Exception as e:
    #     print(e)
