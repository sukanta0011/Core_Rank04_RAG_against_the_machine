from typing import Callable, Dict, List
# import time
from tqdm import tqdm
from pydantic import BaseModel, validate_call, Field, ConfigDict, InstanceOf
from src.base_patterns import (
    MinimalSource,
    MinimalSearchResults,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer)
from src.answer_generation.models.abstract_model import Model
from src.data_retrieval.resource_refiner import ResourceRefiner
from ..speed_optimization.cache import (
    AnswerCacheStructure, AnswerCache)

class AnswerGenerator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Model
    prompt_generator: Callable
    chunked_texts: Dict
    cache: InstanceOf[AnswerCache]
    method: str

    @validate_call
    def generate_answer(
            self,
            search_result: MinimalSearchResults,
            tokens_limit: int = Field(gt=0, default=500),
            refiner: ResourceRefiner | None = None
            ) -> MinimalAnswer:

        hash, resources = self.cache.get_sources_from_cache(
            question=search_result.question,
            resource_numbers=len(search_result.retrieved_sources),
            method=self.method)
        if resources is None:
            return self.generate_new_answer(hash, search_result, tokens_limit, refiner)

        if resources.answer is None:
            return self.generate_new_answer(hash, search_result, tokens_limit, refiner)
        else:
            return MinimalAnswer(
                question_id=search_result.question_id,
                question=search_result.question,
                retrieved_sources_scores=resources.retrieved_sources_scores,
                retrieved_sources=resources.retrieved_sources,
                answer=resources.answer
            )

    @validate_call
    def generate_new_answer(
        self,
        hash: str,
        search_result: MinimalSearchResults,
        tokens_limit: int = Field(gt=0, default=500),
        refiner: ResourceRefiner | None = None
            ) -> MinimalAnswer:
        if refiner:
            # print("token refiner is running")
            new_search_result, new_chunks = refiner.get_refined_sources(
                question=search_result.question,
                data=self._extract_resource(
                    search_result.retrieved_sources,
                    self.chunked_texts),
                minimal_resource=search_result.retrieved_sources
            )
            pre_prompt = self.prompt_generator(
                question=search_result.question,
                context=self._extract_resource(
                    new_search_result.retrieved_sources,
                    new_chunks)
                )
            # print(new_chunks)
        else:
            pre_prompt = self.prompt_generator(
                question=search_result.question,
                context=self._extract_resource(
                    search_result.retrieved_sources,
                    self.chunked_texts)
            )
        # print(pre_prompt)

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
        text = []
        for resource in resources:
            text.append(
                chunked_texts[
                    f"{resource.file_path}:{resource.first_character_index}:{resource.last_character_index}"])
        return text


class BatchAnswerGenerator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    generator: AnswerGenerator
    tokens_limit: int = Field(gt=0, default=500)
    refiner: ResourceRefiner | None = None

    @validate_call
    def process_batch(
            self,
            search_results: StudentSearchResults,
            ) -> StudentSearchResultsAndAnswer:

        all_answers = []
        total_questions = len(search_results.search_results)
        for i, single_search_result in tqdm(enumerate(
                search_results.search_results),
                total=total_questions,
                desc="Generating Answers"):

            all_answers.append(self.generator.generate_answer(
                search_result=single_search_result,
                tokens_limit=self.tokens_limit,
                refiner=self.refiner
            ))
        return StudentSearchResultsAndAnswer(
            search_results=all_answers,
            k=search_results.k
        )
