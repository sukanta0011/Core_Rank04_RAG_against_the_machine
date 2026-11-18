"""Semantic retrieval: MiniLM embeddings stored in a Chroma collection."""
import uuid
from functools import cache
from typing import Annotated, Any, List

import chromadb
from pydantic import Field, PrivateAttr, validate_call
from tqdm import tqdm

from ..base_patterns import MinimalSearchResults, MinimalSource
from ..data_retrieval.abstract_classes import Retriever
from src.all_paths import DATA_CHUNK_FOLDER
from .helper_classes import DataManager


CHROMADB_BATCH_SIZE = 128
COLLECTION_NAME = "vLLM"
MODEL_NAME = "all-MiniLM-L6-v2"


@cache
def get_embedding_function() -> Any:
    """Create the embedding function once, on first use.

    Lazy, so commands that never embed (index with lexical, CLI errors)
    do not pay for importing torch or loading the model.
    """
    import torch  # lazy import: heavy, only needed here
    from chromadb.utils import embedding_functions
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME,
        device="cuda" if torch.cuda.is_available() else "cpu")


class MiniLML6Retriever(Retriever):
    """Nearest-neighbour search over chunk embeddings.

    Chunk ids (`path:first:last`) are the Chroma ids, so a result maps
    back to a source without any list position.
    """

    _client: Any = PrivateAttr(default=None)
    _collection: Any = PrivateAttr(default=None)

    def create_and_save_corpus_index(self, storage_path: str) -> None:
        """Apply `self.update` to the Chroma collection.

        Order: reset (if flagged), delete removed ids, add new chunks.
        Afterwards the collection size must equal the chunk count; a
        mismatch means the store and the chunk files disagree.

        Args:
            storage_path: Folder of the persistent Chroma store.

        Raises:
            Exception: If the collection ends up with a different
                number of chunks than `self.data`.
        """
        self._client = chromadb.PersistentClient(path=storage_path)

        if self.update.reset_flag:
            print("Removing old semantic tokens...")
            try:
                self._client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass  # nothing to delete on a first run

        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=get_embedding_function())

        remove_ids = list(self.update.remove.keys())
        add_ids = list(self.update.add.keys())
        add_texts = list(self.update.add.values())

        for i in tqdm(range(0, len(remove_ids), CHROMADB_BATCH_SIZE),
                      desc="Removing Embeddings"):
            self._collection.delete(
                ids=remove_ids[i: i + CHROMADB_BATCH_SIZE])

        for i in tqdm(range(0, len(add_texts), CHROMADB_BATCH_SIZE),
                      desc="Adding Embeddings"):
            self._collection.upsert(
                ids=add_ids[i: i + CHROMADB_BATCH_SIZE],
                documents=add_texts[i: i + CHROMADB_BATCH_SIZE])

        count = self._collection.count()
        print(f"{count} chunks are saved in Chroma DB")
        if count != len(self.data):
            raise Exception(
                f"Chroma holds {count} chunks but there are "
                f"{len(self.data)}. Delete data/processed and "
                "data/chunks, then run `index` again.")
        self._reset_updates_and_save()

    def _reset_updates_and_save(self) -> None:
        all_update_source = "/".join([DATA_CHUNK_FOLDER, 'update.json'])
        self.update.reset_flag = False
        self.update.add = {}
        self.update.remove = {}
        DataManager.save_data(all_update_source, self.update)

    def load_corpus_index(self, storage_path: str) -> None:
        """Open the saved collection WITHOUT modifying it.

        Args:
            storage_path: Folder of the persistent Chroma store.

        Raises:
            Exception: If the collection is missing or does not match
                the chunk file.
        """
        try:
            self._client = chromadb.PersistentClient(path=storage_path)
            self._collection = self._client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=get_embedding_function())
        except Exception as e:
            raise Exception(
                f"Indexed corpus loading failed. Run `index` first. "
                f"Error: {e}") from e
        count = self._collection.count()
        if count != len(self.data):
            raise Exception(
                f"Chroma holds {count} chunks but the chunk file has "
                f"{len(self.data)}. Run `index` again.")

    @validate_call
    def retrieve_from_raw_data(
        self,
        hash: str,
        question: Annotated[str, Field(min_length=3)],
        k: Annotated[int, Field(gt=0)],
        question_id: str | None = None
            ) -> MinimalSearchResults:
        """Query the collection (no cache lookup) and store the result.

        Args:
            hash: Cache key for this question.
            question: Question text.
            k: Number of sources (clamped to the collection size).
            question_id: Id for the result (a new UUID if None).

        Returns:
            Sources and distances (lower is closer), best first.
        """
        if question_id is None:
            question_id = str(uuid.uuid4())

        n_results = min(k, self._collection.count())
        if n_results == 0:
            return MinimalSearchResults(
                question=question, question_id=question_id,
                retrieved_sources_scores=[], retrieved_sources=[])

        results = self._collection.query(
            query_texts=[question], n_results=n_results)
        scores = [float(d) for d in results['distances'][0]]
        sources = self.get_all_minimal_source(results['ids'][0])

        self.cache.update_answer_cache(
            hash=hash, scores=scores, resources=sources,
            question_id=question_id)

        return MinimalSearchResults(
            question=question,
            question_id=question_id,
            retrieved_sources_scores=scores,
            retrieved_sources=sources
        )

    def get_all_minimal_source(
            self, ids: List) -> List[MinimalSource]:
        """Convert chunk ids (`path:first:last`) to sources.

        The path may contain ':', so the id is split from the right.

        Args:
            ids: Chroma ids.

        Returns:
            One `MinimalSource` per id.
        """
        sources: List[MinimalSource] = []
        for chunk_id in ids:
            path, first, last = str(chunk_id).rsplit(":", 2)
            sources.append(MinimalSource(
                file_path=path,
                first_character_index=int(first),
                last_character_index=int(last)))
        return sources
