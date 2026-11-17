from typing import Dict, List, Type

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from src.base_patterns import (
    AnsweredQuestion, MinimalSearchResults, MinimalSource,
    StudentSearchResults
)

IOU_THRESHOLD = 0.05


def span_iou(a: MinimalSource, b: MinimalSource) -> float:
    """Intersection over union of two character spans.

    Args:
        a: First source (only the character range is used).
        b: Second source.

    Returns:
        IoU in [0, 1]. Zero if the spans do not overlap.
    """
    inter = max(0, min(a.last_character_index, b.last_character_index)
                - max(a.first_character_index, b.first_character_index))
    union = ((a.last_character_index - a.first_character_index)
             + (b.last_character_index - b.first_character_index) - inter)
    return inter / union if union > 0 else 0.0


class ValidatorGuards:
    """Small helper checks shared by the validators."""

    @staticmethod
    def effective_n(total_retrieved: int, n: int) -> int:
        """Return how many results can really be inspected.

        Args:
            total_retrieved: Number of sources actually returned.
            n: Requested k.

        Returns:
            min(total_retrieved, n), without side effects.
        """
        return min(total_retrieved, n)

    @staticmethod
    def same_question(true_question: str, asked_question: str) -> bool:
        """Compare question texts ignoring surrounding whitespace.

        Args:
            true_question: Question text from the ground truth.
            asked_question: Question text from the student output.

        Returns:
            True if they match after stripping whitespace.
        """
        return true_question.strip() == asked_question.strip()


class Validator(BaseModel):
    """Recall of ONE question against ALL its true sources.

    A true source counts as found only if a single retrieved chunk in
    the top n is in the same file AND overlaps its range with
    IoU >= iou_threshold. Otherwise that source contributes zero.
    """

    n: int = Field(gt=0)
    ground_truth_paths: List[MinimalSource]
    indexer_retrieved_paths: List[MinimalSource]
    iou_threshold: float = IOU_THRESHOLD

    @model_validator(mode='after')
    def validate_paths(self) -> Self:
        """Reject empty paths early.

        Raises:
            ValueError: If any ground-truth or retrieved path is empty.
        """
        for i, truth in enumerate(self.ground_truth_paths):
            if not truth.file_path.strip():
                raise ValueError(f"Ground truth path {i} is empty")
        for i, path in enumerate(self.indexer_retrieved_paths):
            if not path.file_path.strip():
                raise ValueError(f"Retrieved path {i} is empty")
        return self

    def _top(self) -> List[MinimalSource]:
        """Return the first n retrieved sources (no mutation of self.n)."""
        count = ValidatorGuards.effective_n(
            len(self.indexer_retrieved_paths), self.n)
        return self.indexer_retrieved_paths[:count]

    def _matches(self, truth: MinimalSource, result: MinimalSource) -> bool:
        """Check one retrieved chunk against one true source.

        Args:
            truth: Ground-truth source.
            result: One retrieved source.

        Returns:
            True only if same file and IoU >= iou_threshold. An empty
            truth span matches if the chunk contains that position.
        """
        if result.file_path != truth.file_path:
            return False
        if truth.first_character_index == truth.last_character_index:
            return (result.first_character_index
                    <= truth.first_character_index
                    <= result.last_character_index)
        return span_iou(truth, result) >= self.iou_threshold

    def source_found(self, truth: MinimalSource) -> bool:
        """True if any top-n result matches this true source.

        Args:
            truth: One ground-truth source.

        Returns:
            Whether the source counts as found.
        """
        return any(self._matches(truth, r) for r in self._top())

    def validate_source(self) -> float:
        """Share of the true sources found in the top n.

        Returns:
            Recall in [0, 1]. 0.0 if there are no ground-truth sources.
        """
        if not self.ground_truth_paths:
            return 0.0
        found = sum(self.source_found(t) for t in self.ground_truth_paths)
        return found / len(self.ground_truth_paths)


class RecallN(Validator):
    """Recall@n with the overlap criterion applied."""


class RecallOverlap(Validator):
    """Same metric, kept so existing imports and calls still work."""


class AnswerValidator(BaseModel):
    """Scores one student result against the ground-truth map."""

    ground_truth_map: Dict[str, AnsweredQuestion]
    recall: Type[Validator]
    n: int = Field(gt=0)

    def validate_answer(self, answer: MinimalSearchResults) -> float:
        """Compute recall@n for one question.

        Args:
            answer: Student search result for one question.

        Returns:
            Recall in [0, 1].

        Raises:
            KeyError: If the question id has no ground truth.
        """
        truth = self.ground_truth_map.get(answer.question_id)
        if truth is None:
            raise KeyError(f"No ground truth for '{answer.question_id}'")
        if not ValidatorGuards.same_question(
                truth.question, answer.question):
            print(f"Warning: question text differs for "
                  f"{answer.question_id}")
        return self.recall(
            n=self.n,
            ground_truth_paths=truth.sources,
            indexer_retrieved_paths=answer.retrieved_sources,
        ).validate_source()


class BatchAnswerValidator(BaseModel):
    """Scores a whole student output; failures count as 0, not skipped."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    validator: AnswerValidator
    errors: Dict[str, str] = Field(default_factory=dict)

    def process_batch(self, answers: StudentSearchResults) -> List[float]:
        """Score every question.

        Args:
            answers: The student's search results.

        Returns:
            One recall per question, in order. A question that raises
            is recorded in `self.errors` and scored 0.0.
        """
        scores: List[float] = []
        for answer in answers.search_results:
            try:
                scores.append(self.validator.validate_answer(answer))
            except Exception as e:
                self.errors[answer.question_id] = str(e)
                scores.append(0.0)
        if self.errors:
            print(f"{len(self.errors)} question(s) failed and scored 0.")
        return scores
