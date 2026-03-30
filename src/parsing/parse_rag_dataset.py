from typing import List, Dict
import json
from pydantic import BaseModel, Field, PrivateAttr
from src.data_indexing.helper_classes import ExistingPath
from src.base_patterns import(
    MinimalSource, UnansweredQuestion,
    AnsweredQuestion, RagDataset
)

class ParsingError(Exception):
    pass


class RagDatasetParser(BaseModel):
    answered_question_paths: List[ExistingPath]
    unanswered_question_paths: List[ExistingPath]

    _answered_data_set: RagDataset = PrivateAttr()
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
        answered_data: Dict
        unanswered_data: Dict

        all_answered: List[AnsweredQuestion] = []
        for path in self.answered_question_paths:
            data = RagDatasetParser._load_json(path)
            data = data.get('rag_questions', [])
            if len(data) > 0:
                # print(data)
                for answer in data:
                    answered_questions = AnsweredQuestion(
                        question_id=answer['question_id'],
                        question=answer['question'],
                        sources=answer['sources'],
                        answer=answer['answer']
                    )
                    all_answered.append(answered_questions)
        self._answered_data_set = RagDataset(rag_questions=all_answered)
        
        all_unanswered: List[UnansweredQuestion] = []
        for path in self.unanswered_question_paths:
            data = RagDatasetParser._load_json(path)
            data = data.get('rag_questions', [])
            if len(data) > 0:
                for answer in unanswered_data:
                    unanswered_questions = UnansweredQuestion(
                        question_id=answer['question_id'],
                        question=answer['question'],
                    )
                    all_unanswered.append(unanswered_questions)
        self._unanswered_data_set = RagDataset(rag_questions=unanswered_questions)

    def get_answered_data(self) -> RagDataset:
        return self._answered_data_set


def test_rag_data_parser() -> None:
    answered_path = "datasets_public/public/AnsweredQuestions/dataset_docs_public.json"
    unanswered_path = "datasets_public/public/UnansweredQuestions/dataset_docs_public.json"

    rag_parser = RagDatasetParser(
        answered_question_paths=[answered_path],
        unanswered_question_paths=[unanswered_path]
    )
    rag_parser.extract_data_from_paths()
    answered_data = rag_parser.get_answered_data()
    # print(answered_data.rag_questions)
    for data in answered_data.rag_questions:
        print(data.question_id)


if __name__ == "__main__":
    test_rag_data_parser()