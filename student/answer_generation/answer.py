from typing import Callable, List
import time
from tqdm import tqdm
from pydantic import BaseModel, validate_call, Field, ConfigDict
from student.base_patterns import (
    MinimalSearchResults,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer)
from student.answer_generation.models.abstract_model import Model


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


class BatchAnswerGenerator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    generator: AnswerGenerator
    tokens_limit: int = Field(gt=0, default=500)

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

            # print(f"Question: {i}, ", end="")
            start = time.time()
            all_answers.append(self.generator.generate_answer(
                search_result=single_search_result,
                tokens_limit=self.tokens_limit,
            ))
            # time_taken = round((time.time() - start), 3)
            # print(f"Time: \033[92m{time_taken}\033[0ms")
        return StudentSearchResultsAndAnswer(
            search_results=all_answers,
            k=search_results.k
        )
