import time
from src.data_retrieval.helper_classes import FilesInDir, prepare_storage_folder
from src.data_retrieval.chunk_data import SplitDataByChunks
from src.data_retrieval.BM25retriever import BM25Retriever
from src.answer_generation.models.qwen3__0_6B import SmallLLM
from src.answer_generation.pre_prompt import InitialPromptGenerator
from src.answer_generation.answer import AnswerGenerator
from src.validator.resource_validator import RecallN, SingleAnswerValidator
from src.parsing.parse_rag_dataset import RagDatasetParser


def main():
    start_time = time.time()

    all_valid_paths = FilesInDir.extract_all_file_paths(path='data/raw/vllm-0.10.1')
    splitter = SplitDataByChunks(
        all_paths=all_valid_paths,
        chunk_size=2000, txt_overlap=0)
    splitter.chunk_all_files()
    # splitter.save_chunked_data("data/chunks")

    # all_minimal_sources, all_data_chunks = splitter.load_from_files("data/chunks")
    # print(len(all_minimal_sources))
    all_minimal_sources, all_data_chunks = splitter.get_all_data()
    print("Data chunking is completed with "
          f"\033[92m{len(all_data_chunks)}\033[0m chunks")
    print(f"Time taken: \033[92m{(time.time() - start_time):.3f}s\033[0m")

    bm25_retriever = BM25Retriever(
        storage_path="data/processed/",
        data=all_data_chunks,
        all_minimal_resource=all_minimal_sources)

    bm25_retriever.create_corpus_index()
    # prepare_storage_folder("Data indexing is completed")

    # bm25_retriever.load_corpus_index()

    question_id = "cc83c230-099f-4c11-aeab-8c09715c5942"
    question = "What command can be used to evaluate the accuracy of a quantized model using lm_eval with vLLM?"

    minimal_search_results = bm25_retriever.get_matching_chunk(question, k=5, question_id=question_id)
    for data in minimal_search_results.retrieved_sources:
        print(data.file_path)


    # # Validating resources
    # answered_path = "datasets_public/public/AnsweredQuestions/dataset_docs_public_test.json"
    # unanswered_path = "datasets_public/public/UnansweredQuestions/dataset_docs_public.json"

    # rag_parser = RagDatasetParser(
    #     answered_question_paths=[answered_path],
    #     unanswered_question_paths=[unanswered_path]
    # )
    # rag_parser.extract_data_from_paths()
    # ground_truth = rag_parser.get_ground_truth()

    # validator = SingleAnswerValidator(
    #     ground_truth_map=ground_truth,
    #     recall=RecallN,
    #     n=5
    # )
    # is_valid = validator.validate_answer(
    #     answer=minimal_search_results
    # )
    # print(is_valid)

    # Using model to get the answer
    llm = SmallLLM(device_type='cuda')
    answer_generator = AnswerGenerator(
        model=llm,
        prompt_generator=InitialPromptGenerator.get_type1_prompt,
        chunked_texts=all_data_chunks
    )
    
    print("\033[92m ---------Answer------------- \033[0m")
    answer = answer_generator.generate_answer(minimal_search_results, tokens_limit=1000)
    print(answer.answer)

    end_time = time.time()
    print(f"Time taken: {(end_time - start_time):.3f}s")


if __name__ == "__main__":
    main()
