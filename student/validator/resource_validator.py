from abc import ABC, abstractmethod
from typing import List, Self, Dict, Type
from pydantic import (
    BaseModel, Field, model_validator,
    ValidationError, ConfigDict)
from student.base_patterns import (
    AnsweredQuestion, MinimalSearchResults, MinimalSource,
    StudentSearchResults
)


class ValidatorGuards:
    @staticmethod
    def get_min_len(total_retrieved_paths: int, n: int) -> int:
        if total_retrieved_paths < n:
            print(
                f"Recall@{n} required minimum {n} source paths,"
                f" You have provided {total_retrieved_paths} paths"
                )
            n = total_retrieved_paths
        return n

    @staticmethod
    def question_comparison(true_question: str, asked_question: str) -> None:
        if true_question != asked_question:
            raise ValueError(
                f"Ground truth question: {true_question} is "
                f" different from the answered question: {asked_question}."
            )


class Validator(BaseModel, ABC):
    n: int = Field(gt=0)
    ground_truth_path: MinimalSource
    indexer_retrieved_paths: List[MinimalSource]

    @model_validator(mode='after')
    def validate_paths(self) -> Self:
        if len(self.ground_truth_path.file_path.strip()) == 0:
            raise ValidationError("Absolute_truth path is empty")

        for i, path in enumerate(self.indexer_retrieved_paths):
            if len(path.file_path.strip()) == 0:
                raise ValidationError(
                    f"Index retrieved path {i} is empty")
        return self

    @abstractmethod
    def validate_source(self) -> bool:
        pass


class RecallN(Validator):
    def validate_source(self) -> bool:
        total_retrieved_paths = len(self.indexer_retrieved_paths)
        self.n = ValidatorGuards.get_min_len(total_retrieved_paths, self.n)

        for i in range(0, self.n):
            if (self.ground_truth_path.file_path ==
                    self.indexer_retrieved_paths[i].file_path):
                return True
        return False


class RecallOverlap(Validator):
    def validate_source(self) -> bool:
        total_retrieved_paths = len(self.indexer_retrieved_paths)
        self.n = ValidatorGuards.get_min_len(total_retrieved_paths, self.n)
        truth_start = self.ground_truth_path.first_character_index
        truth_end = self.ground_truth_path.last_character_index
        if truth_end == truth_start:
            print(f"{self.ground_truth_path.file_path} "
                  "has not same start and stop")
            return True

        total_overlap = 0

        for i in range(0, self.n):
            answer = self.indexer_retrieved_paths[i]
            if (self.ground_truth_path.file_path !=
                    answer.file_path):
                continue
            total_overlap += max(
                0, (min(truth_end, answer.last_character_index) -
                    max(truth_start, answer.first_character_index)))

        if (total_overlap / (truth_end - truth_start)) >= 0.05:
            return True
        return False


class AnswerValidator(BaseModel):
    ground_truth_map: Dict[str, AnsweredQuestion]
    recall: Type[Validator]
    n: int = Field(gt=0)

    def validate_answer(self, answer: MinimalSearchResults) -> bool:
        question_id = answer.question_id
        ground_truth = self.ground_truth_map.get(question_id)

        if ground_truth is None:
            print(f"'{question_id}' do not have Ground truth to validate")
            return False

        ValidatorGuards.question_comparison(
            ground_truth.question, answer.question)

        recall_instance = self.recall(
            n=self.n,
            ground_truth_path=ground_truth.sources[0],
            indexer_retrieved_paths=answer.retrieved_sources
            )
        return recall_instance.validate_source()


class BatchAnswerValidator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    validator: AnswerValidator

    def process_batch(self, answers: StudentSearchResults) -> List[bool]:
        results: List[bool] = []
        for answer in answers.search_results:
            try:
                results.append(self.validator.validate_answer(answer))
            except Exception as e:
                print(f"Error: {e}")
        return results
