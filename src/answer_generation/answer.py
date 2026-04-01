from typing import Callable, List
from pydantic import BaseModel, validate_call, Field, ConfigDict
from src.base_patterns import MinimalSearchResults, MinimalAnswer
from src.answer_generation.models.abstract_model import Model
# from src.answer_generation import pre_prompt


class AnswerGenerator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Model
    prompt_generator: Callable
    chunked_texts: List[str]

    @validate_call
    def generate_answer(
            self,
            search_result: MinimalSearchResults,
            tokens_limit: int = Field(gt=50, default=500)
            ) -> MinimalAnswer:

        pre_prompt = self.prompt_generator(
            question=search_result.question,
            context=[self.chunked_texts[idx]\
                     for idx in search_result.retrieved_sources_indexes])

        answer = self.model.generate_answer(pre_prompt, tokens_limit)
        return MinimalAnswer(
            question_id=search_result.question_id,
            question=search_result.question,
            retrieved_sources_indexes=search_result.retrieved_sources_indexes,
            retrieved_sources_scores=search_result.retrieved_sources_scores,
            retrieved_sources=search_result.retrieved_sources,
            answer=answer
        )


class BatchAnswerGenerator(AnswerGenerator):
    @validate_call
    def generate_answers(
            self,
            search_results: List[MinimalSearchResults],
            tokens_limit: int = Field(gt=50, default=500)
            ) -> List[MinimalAnswer]:

        all_answers = []
        for single_search_result in search_results:
            all_answers.append(self.generate_answer(
                search_result=single_search_result,
                tokens_limit=tokens_limit,
            ))
        return all_answers
