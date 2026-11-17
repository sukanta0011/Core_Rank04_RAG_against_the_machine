"""Answer generation from retrieved sources, with answer caching."""
from typing import Callable, Dict, List

from pydantic import (
    BaseModel, ConfigDict, Field, InstanceOf, validate_call)
from tqdm import tqdm

from src.answer_generation.models.abstract_model import Model
from src.base_patterns import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer)
from src.data_retrieval.resource_refiner import ResourceRefiner
from ..speed_optimization.cache import AnswerCache

NO_CONTEXT_ANSWER = "I cannot find this in the provided context."


class AnswerGenerator(BaseModel):
    """Build a prompt from retrieved chunks and generate an answer.

    Attributes:
        model: Language model wrapper.
        prompt_generator: Function (question, context) -> messages.
        chunked_texts: chunk id (`path:first:last`) -> chunk text.
        cache: Cache of earlier results and answers.
        method: Retrieval method name, part of the cache key.
        corpus_digest: Digest of the indexed corpus, part of the cache
            key. Must equal the one used by the retriever.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Model
    prompt_generator: Callable
    chunked_texts: Dict
    cache: InstanceOf[AnswerCache]
    method: str
    corpus_digest: str = ""

    @validate_call
    def generate_answer(
            self,
            search_result: MinimalSearchResults,
            tokens_limit: int = Field(gt=0, default=500),
            refiner: ResourceRefiner | None = None
            ) -> MinimalAnswer:
        """Return an answer, from the cache when one is stored.

        Args:
            search_result: Retrieved sources for one question.
            tokens_limit: Maximum number of new tokens.
            refiner: Optional refiner that shrinks the context.

        Returns:
            The question, sources and answer. Always carries the
            CURRENT question id, never a cached one.
        """
        key, cached = self.cache.get_sources_from_cache(
            question=search_result.question,
            resource_numbers=len(search_result.retrieved_sources),
            method=self.method,
            corpus_digest=self.corpus_digest)
        if cached is None or cached.answer is None:
            return self.generate_new_answer(
                key, search_result, tokens_limit, refiner)
        return MinimalAnswer(
            question_id=search_result.question_id,
            question=search_result.question,
            retrieved_sources_scores=search_result.retrieved_sources_scores,
            retrieved_sources=search_result.retrieved_sources,
            answer=cached.answer
        )

    @validate_call
    def generate_new_answer(
        self,
        hash: str,
        search_result: MinimalSearchResults,
        tokens_limit: int = Field(gt=0, default=500),
        refiner: ResourceRefiner | None = None
            ) -> MinimalAnswer:
        """Generate an answer with the model and cache it.

        Args:
            hash: Cache key of this question.
            search_result: Retrieved sources for one question.
            tokens_limit: Maximum number of new tokens.
            refiner: Optional refiner that shrinks the context.

        Returns:
            The question, sources and answer. With no usable context,
            the fixed "cannot find" answer is returned and not cached.
        """
        if not search_result.retrieved_sources:
            return MinimalAnswer(
                question_id=search_result.question_id,
                question=search_result.question,
                retrieved_sources_scores=[],
                retrieved_sources=[],
                answer=NO_CONTEXT_ANSWER)

        if refiner:
            new_search_result, new_chunks = refiner.get_refined_sources(
                question=search_result.question,
                data=self._extract_resource(
                    search_result.retrieved_sources, self.chunked_texts),
                minimal_resource=search_result.retrieved_sources)
            context = self._extract_resource(
                new_search_result.retrieved_sources, new_chunks)
        else:
            context = self._extract_resource(
                search_result.retrieved_sources, self.chunked_texts)

        pre_prompt = self.prompt_generator(
            question=search_result.question, context=context)
        answer = self.model.generate_answer(pre_prompt, tokens_limit)
        self.cache.update_answer_cache(
            hash=hash,
            scores=search_result.retrieved_sources_scores,
            resources=search_result.retrieved_sources,
            question_id=search_result.question_id,
            answer=answer
            )
        return MinimalAnswer(
            question_id=search_result.question_id,
            question=search_result.question,
            retrieved_sources_scores=search_result.retrieved_sources_scores,
            retrieved_sources=search_result.retrieved_sources,
            answer=answer
        )

    def _extract_resource(
            self, resources: List[MinimalSource],
            chunked_texts: Dict) -> List[str]:
        """Look up the text of each source.

        Args:
            resources: Sources to look up.
            chunked_texts: chunk id -> text.

        Returns:
            Texts in the same order. A source whose chunk is missing
            (stale cache or index) is skipped with a warning, not a
            crash.
        """
        texts: List[str] = []
        for resource in resources:
            chunk_id = (f"{resource.file_path}:"
                        f"{resource.first_character_index}:"
                        f"{resource.last_character_index}")
            text = chunked_texts.get(chunk_id)
            if text is None:
                print(f"Warning: chunk not found: {chunk_id}")
                continue
            texts.append(text)
        return texts


class BatchAnswerGenerator(BaseModel):
    """Generate answers for a whole results file.

    Attributes:
        generator: The shared answer generator.
        tokens_limit: Maximum number of new tokens per answer.
        refiner: Optional refiner.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    generator: AnswerGenerator
    tokens_limit: int = Field(gt=0, default=500)
    refiner: ResourceRefiner | None = None

    @validate_call
    def process_batch(
            self,
            search_results: StudentSearchResults,
            ) -> StudentSearchResultsAndAnswer:
        """Answer every question; one failure cannot stop the batch.

        Args:
            search_results: Output of `search_dataset`.

        Returns:
            One answer per question, in order. A question whose
            generation fails gets an empty-context answer.
        """
        all_answers: List[MinimalAnswer] = []
        for single in tqdm(search_results.search_results,
                           desc="Generating Answers"):
            try:
                all_answers.append(self.generator.generate_answer(
                    search_result=single,
                    tokens_limit=self.tokens_limit,
                    refiner=self.refiner))
            except Exception as e:
                print(f"Answer failed for '{single.question_id}': {e}")
                all_answers.append(MinimalAnswer(
                    question_id=single.question_id,
                    question=single.question,
                    retrieved_sources_scores=(
                        single.retrieved_sources_scores),
                    retrieved_sources=single.retrieved_sources,
                    answer=NO_CONTEXT_ANSWER))
        return StudentSearchResultsAndAnswer(
            search_results=all_answers,
            k=search_results.k
        )
