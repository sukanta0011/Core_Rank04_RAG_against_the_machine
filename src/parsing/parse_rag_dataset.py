"""Parsing of RAG question datasets (answered and unanswered)."""
import json
from typing import Any, Dict, List

from pydantic import BaseModel, PrivateAttr, ValidationError

from src.data_retrieval.helper_classes import ExistingPath
from src.base_patterns import (
    UnansweredQuestion,
    AnsweredQuestion, RagDataset
)


class ParsingError(Exception):
    """Raised when a dataset file cannot be read or has a wrong shape."""


class RagDatasetParser(BaseModel):
    """Load question datasets and expose them as pydantic models.

    Attributes:
        answered_question_paths: JSON files with ground-truth questions.
        unanswered_question_paths: JSON files with questions only.
    """

    answered_question_paths: List[ExistingPath]
    unanswered_question_paths: List[ExistingPath]

    _answered_data_set: RagDataset = PrivateAttr(
        default_factory=lambda: RagDataset(rag_questions=[]))
    _ground_truth_map: Dict[str, AnsweredQuestion] = PrivateAttr(
        default_factory=dict)
    _unanswered_data_set: RagDataset = PrivateAttr(
        default_factory=lambda: RagDataset(rag_questions=[]))

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        """Read a JSON file whose top level must be an object.

        Args:
            path: File to read.

        Returns:
            The parsed top-level dict.

        Raises:
            ParsingError: If the file cannot be read, is not valid
                JSON/UTF-8, or its top level is not an object.
        """
        try:
            with open(path, 'r', encoding="utf-8") as fl:
                data = json.load(fl)
        except OSError as e:
            raise ParsingError(f"Unable to open {path}, Error: {e}") from e
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ParsingError(
                f"{path} is not in valid Json format: {e}") from e
        if not isinstance(data, dict):
            raise ParsingError(
                f"{path}: top level must be a JSON object with a "
                f"'rag_questions' list, got {type(data).__name__}")
        return data

    @staticmethod
    def _load_questions(path: str) -> List[Dict[str, Any]]:
        """Return the `rag_questions` list of a dataset file.

        Args:
            path: Dataset file.

        Returns:
            The list of raw question dicts.

        Raises:
            ParsingError: If `rag_questions` is missing, is not a list,
                or contains something other than objects.
        """
        data = RagDatasetParser._load_json(path)
        if "rag_questions" not in data:
            raise ParsingError(f"{path}: missing 'rag_questions' key")
        questions = data["rag_questions"]
        if not isinstance(questions, list):
            raise ParsingError(
                f"{path}: 'rag_questions' must be a list, got "
                f"{type(questions).__name__}")
        for i, item in enumerate(questions):
            if not isinstance(item, dict):
                raise ParsingError(
                    f"{path}: question {i} must be an object")
        return questions

    def extract_data_from_paths(self) -> None:
        """Load every configured file into the internal datasets.

        Can be called again: previous results are replaced, not added to.

        Raises:
            ParsingError: On unreadable files, wrong structure, or
                questions that fail validation (the message names the
                file and question index).
        """
        self._ground_truth_map = {}
        all_answered: List[AnsweredQuestion] = []
        for path in self.answered_question_paths:
            for i, raw in enumerate(self._load_questions(path)):
                try:
                    question = AnsweredQuestion(**raw)
                except (ValidationError, TypeError) as e:
                    raise ParsingError(
                        f"{path}: invalid answered question {i}: {e}"
                    ) from e
                if question.question_id in self._ground_truth_map:
                    print(f"Warning: duplicate question_id "
                          f"'{question.question_id}' in {path}")
                self._ground_truth_map[question.question_id] = question
                all_answered.append(question)
        self._answered_data_set = RagDataset(rag_questions=all_answered)

        all_unanswered: List[UnansweredQuestion] = []
        for path in self.unanswered_question_paths:
            for i, raw in enumerate(self._load_questions(path)):
                try:
                    all_unanswered.append(UnansweredQuestion(**raw))
                except (ValidationError, TypeError) as e:
                    raise ParsingError(
                        f"{path}: invalid question {i}: {e}") from e
        self._unanswered_data_set = RagDataset(rag_questions=all_unanswered)

    def get_answered_data(self) -> RagDataset:
        """Return the answered questions (empty before extraction)."""
        return self._answered_data_set

    def get_unanswered_data(self) -> RagDataset:
        """Return the unanswered questions (empty before extraction)."""
        return self._unanswered_data_set

    def get_ground_truth(self) -> Dict[str, AnsweredQuestion]:
        """Return a question_id -> answered question map."""
        return self._ground_truth_map
