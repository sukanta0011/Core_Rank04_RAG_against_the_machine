import chromadb
import torch
import uuid
from tqdm import tqdm
from chromadb.utils import embedding_functions
from pydantic import PrivateAttr, Field
from typing import Annotated, List
# from sentence_transformers import SentenceTransformer
from ..data_retrieval.abstract_classes import Retriever
from ..base_patterns import MinimalSearchResults, MinimalSource


CHROMADB_BATCH_SIZE = 128


class MiniLML6Retriever(Retriever):
    _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device="cuda" if torch.cuda.is_available() else "cpu"
        )
    _client = PrivateAttr()
    _collection = PrivateAttr()

    def create_and_save_corpus_index(self, storage_path):
        if isinstance(self.data, dict):
            self._client = chromadb.PersistentClient(path=storage_path)

            self._collection = self._client.get_or_create_collection(
                name="vLLM",
                embedding_function=self._embed_fn
            )

            remove_data = list(self.update.remove.values())
            remove_data_ids = list(self.update.remove.keys())
            add_data = list(self.update.add.values())
            add_data_ids = list(self.update.add.keys())

            for i in tqdm(
                range(0, len(remove_data), CHROMADB_BATCH_SIZE),
                    desc="Removing Embeddings"):
                batch_data = remove_data[i: i + CHROMADB_BATCH_SIZE]
                batch_ids = remove_data_ids[i: i + CHROMADB_BATCH_SIZE]
                self._collection.upsert(
                        ids=batch_ids,
                        documents=batch_data
                    )

            for i in tqdm(
                range(0, len(add_data), CHROMADB_BATCH_SIZE),
                      desc="Adding Embeddings"):
                batch_data = add_data[i: i + CHROMADB_BATCH_SIZE]
                batch_ids = add_data_ids[i: i + CHROMADB_BATCH_SIZE]

                self._collection.upsert(
                    ids=batch_ids,
                    documents=batch_data
                )

    def load_corpus_index(self, storage_path):
        try:
            self.create_and_save_corpus_index(storage_path)
        except Exception as e:
            raise Exception(f"Indexed corpus loading failed. Error: {e}")

    def get_matching_chunk(
        self,
        question: Annotated[str, Field(min_length=3)],
        k: Annotated[int, Field(gt=0)],
        question_id: str | None = None
            ) -> MinimalSearchResults:

        results = self._collection.query(
            query_texts=[question],
            n_results=k
        )

        if question_id is None:
            question_id = str(uuid.uuid4())

        keys = results['ids'][0]

        return MinimalSearchResults(
            question=question,
            question_id=question_id,
            retrieved_sources_scores=results['distances'][0],
            retrieved_sources=self.get_all_minimal_source(keys)
        )

    def get_all_minimal_source(
            self, ids: List) -> List[MinimalSource]:
        all_minimal_sources: List[MinimalSource] = []
        for i in ids:
            parts = str(i).split(":")
            if len(parts) > 3:
                path = ":".join(parts[:-2])
            else:
                path = parts[0]
            all_minimal_sources.append(
                MinimalSource(
                    file_path=path,
                    first_character_index=int(parts[-2]),
                    last_character_index=int(parts[-1])
                )
            )
        return all_minimal_sources


def chromadb_test():
    chroma_client = chromadb.PersistentClient(path="data/vector_data")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = chroma_client.get_or_create_collection(
        name="test_collection",
        embedding_function=embed_fn
        )

    collection.upsert(
        ids=["0", "1", "2", "3"],
        documents=[
            "This is a document about pineapple",
            "This is a document about oranges",
            "This is a document about apples",
            "This is a document about Fish"
        ]
    )

    results = collection.query(
        query_texts=["i want orange"],
        n_results=3
    )

    print(results['ids'][0], results['distances'][0])
    print(results)


if __name__ == "__main__":
    print("testing ChromaDB")
    chromadb_test()
