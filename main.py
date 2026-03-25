import time
from src.data_indexing.helper_classes import FilesInDir, prepare_storage_folder
from src.data_indexing.chunk_data import SplitDataByChunks
from src.data_indexing.indexer import BM25Indexer


def main():
    all_valid_paths = FilesInDir.extract_all_file_paths(path='vllm-0.10.1')
    splitter = SplitDataByChunks(
        all_paths=all_valid_paths,
        chunk_size=2000)
    splitter.chunk_all_files()
    # all_minimal_sources = splitter.get_all_minimal_sources()
    # print(len(all_minimal_sources))
    all_data_chunks = splitter.get_all_data_chunks()
    print("Data chunking is completed with "
          f"\033[92m{len(all_data_chunks)}\033[0m chunks")

    bm25_indexer = BM25Indexer(storage_path="data/processed/")
    bm25_indexer.create_corpus_index(all_data_chunks)
    prepare_storage_folder("Data indexing is completed")


if __name__ == "__main__":
    start_time = time.time()

    main()

    end_time = time.time()
    print(f"Time taken: {(end_time - start_time):.3f}s")
