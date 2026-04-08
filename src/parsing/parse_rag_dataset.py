from typing import List, Dict
import json
from pydantic import BaseModel, Field, PrivateAttr
from src.data_retrieval.helper_classes import ExistingPath
from src.base_patterns import(
    UnansweredQuestion,
    AnsweredQuestion, RagDataset
)

class ParsingError(Exception):
    pass


class RagDatasetParser(BaseModel):
    answered_question_paths: List[ExistingPath]
    unanswered_question_paths: List[ExistingPath]

    _answered_data_set: RagDataset = PrivateAttr() 
    _ground_truth_map: Dict[str, AnsweredQuestion] = PrivateAttr(default_factory=dict)
    _unanswered_data_set: RagDataset = PrivateAttr()

    @staticmethod
    def _load_json(path) -> Dict:
        try:
            with open(path, 'r') as fl:
                data = json.load(fl)
        except OSError as e:
            raise ParsingError(f"Unable to open {path}, Error: {e}")
        except json.JSONDecodeError as e:
            raise ParsingError(
                f"{path} is not in valid Json format: {e}"
            )
        return data

    def extract_data_from_paths(self):
        all_answered: List[AnsweredQuestion] = []
        for path in self.answered_question_paths:
            data = RagDatasetParser._load_json(path)
            data = data.get('rag_questions', [])
            if len(data) > 0:
                for answer in data:
                    answered_questions = AnsweredQuestion(**answer)
                    self._ground_truth_map[answered_questions.question_id] = answered_questions
                    all_answered.append(answered_questions)
        self._answered_data_set = RagDataset(rag_questions=all_answered)

        all_unanswered: List[UnansweredQuestion] = []
        for path in self.unanswered_question_paths:
            data = RagDatasetParser._load_json(path)
            data = data.get('rag_questions', [])
            if len(data) > 0:
                for answer in data:
                    unanswered_questions = UnansweredQuestion(**answer)
                    all_unanswered.append(unanswered_questions)
        self._unanswered_data_set = RagDataset(rag_questions=all_unanswered)

    def get_answered_data(self) -> RagDataset:
        return self._answered_data_set
    
    def get_unanswered_data(self) -> RagDataset:
        return self._unanswered_data_set

    def get_ground_truth(self) -> Dict[str, AnsweredQuestion]:
        return self._ground_truth_map


def test_rag_data_parser() -> None:
    answered_path = "datasets_public/public/AnsweredQuestions/dataset_docs_public_test.json"
    unanswered_path = "datasets_public/public/UnansweredQuestions/dataset_docs_public.json"

    rag_parser = RagDatasetParser(
        answered_question_paths=[answered_path],
        unanswered_question_paths=[unanswered_path]
    )
    rag_parser.extract_data_from_paths()
    # answered_data = rag_parser.get_answered_data()
    # print(answered_data.rag_questions)
    # for data in answered_data.rag_questions:
    #     print(f"Id: {data.question_id}")
    #     source = data.sources
    #     print(f"data: {source[0].file_path}")

    answered_data = rag_parser.get_ground_truth()
    for data in answered_data:
        print(data)


if __name__ == "__main__":
    try:
        test_rag_data_parser()
    except Exception as e:
        print(e)