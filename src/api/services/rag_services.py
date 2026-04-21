from typing import List
from src.data_retrieval.chunk_data import SplitDataByChunks
from src.data_retrieval.lexical_retriever import (
    BM25Retriever)
from src.answer_generation.models.qwen3__0_6B import SmallLLM
from src.answer_generation.pre_prompt import InitialPromptGenerator
from src.answer_generation.answer import (
    AnswerGenerator)
from src.data_retrieval.resource_refiner import ResourceRefiner
from src.data_retrieval.chunk_data import (
    TextChunk, CodeChunk)
from src.data_retrieval.hybrid_retriever import HybridRetriever


class RAGService:
    def __init__(self):
        self.retriever = None
        self.answer_generator = None
        self.all_chunks = None
        self.refiner = None

    def initialize_resources(self):
        self.retriever = self._get_retriever()
        self.answer_generator = self._get_answer_generator(self.all_chunks)
        self.refiner = self._get_refiner(chunk_size=2000, overlap=50)

    def _get_retriever(self) -> BM25Retriever:
        """Helper to initialize resources once."""
        if self.retriever is None:
            # 1. Load Chunks
            all_sources, all_chunks =\
                SplitDataByChunks.load_from_files("data/chunks")
            self.all_chunks = all_chunks

            # 2. Initialize Retriever
            self.retriever = HybridRetriever(
                data=all_chunks,
                all_minimal_resource=all_sources
            )
            # 3. Load Index
            self.retriever.load_corpus_index(storage_path="data/processed/")

        return self.retriever
    
    def _get_answer_generator(
            self,
            all_chunks: List[str]
            ) -> AnswerGenerator:
        if self.answer_generator is None:
            print("Initiating answer generator")    
            # 1. Load llm
            llm = SmallLLM(device_type='cuda')

            # 2. Initialize Answer Generator
            self.answer_generator = AnswerGenerator(
                model=llm,
                prompt_generator=InitialPromptGenerator.get_type1_prompt,
                chunked_texts=all_chunks
                )
        return self.answer_generator
    
    def _get_refiner(self, chunk_size:int, overlap: int, k:int=5) -> ResourceRefiner:
        return ResourceRefiner(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            k=k,
            text_chunk=TextChunk,
            code_chunk=CodeChunk,
            retriever=BM25Retriever
        )
