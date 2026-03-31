import time
from src.data_indexing.helper_classes import FilesInDir, prepare_storage_folder
from src.data_indexing.chunk_data import SplitDataByChunks
from src.data_indexing.indexer import BM25Indexer
from src.answer_generation.models.quen3__0_6B import SmallLLM
from src.answer_generation.pre_prompt import InitialPromptGenerator


def main():
    all_valid_paths = FilesInDir.extract_all_file_paths(path='vllm-0.10.1')
    splitter = SplitDataByChunks(
        all_paths=all_valid_paths,
        chunk_size=2000, txt_overlap=200)
    splitter.chunk_all_files()
    all_minimal_sources = splitter.get_all_minimal_sources()
    # print(len(all_minimal_sources))
    all_data_chunks = splitter.get_all_data_chunks()
    print("Data chunking is completed with "
          f"\033[92m{len(all_data_chunks)}\033[0m chunks")

    bm25_indexer = BM25Indexer(storage_path="data/processed/")
    # bm25_indexer.create_corpus_index(all_data_chunks)
    # prepare_storage_folder("Data indexing is completed")
    bm25_indexer.load_corpus_index()
    question = "What activation formats does the fused batched MoE layer return in vLLM?"
    matching_chunks, scores = bm25_indexer.get_matching_chunk(question, k=5)
    # print(matching_chunks)
    matching_chunks_txt = []

    print("\033[92m ---------Resources------------- \033[0m")
    for chunk, score in zip(matching_chunks, scores):
        print("\033[92m ---------Info Source------------- \033[0m")
        source = all_minimal_sources[chunk["id"]]
        print(source.file_path, source.first_character_index, source.last_character_index, score)
        text = all_data_chunks[chunk["id"]]
        # print("\033[92m ---------Info------------- \033[0m")
        # print(text)
        matching_chunks_txt.append(text)
        # print(chunk['text'])

    # Using model to get the answer
    llm = SmallLLM()
    pre_prompt = InitialPromptGenerator.get_type1_prompt(
        question=question, context=matching_chunks_txt)
    print("\033[92m ---------Answer------------- \033[0m")
    answer = llm.generate_answer(pre_prompt, tokens_limit=500)
    print(answer)


if __name__ == "__main__":
    start_time = time.time()

    main()

    end_time = time.time()
    print(f"Time taken: {(end_time - start_time):.3f}s")
