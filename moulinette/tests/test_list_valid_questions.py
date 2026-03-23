# ABOUTME: Tests for the list_valid_questions CLI command and build_eval_objects helper.
# ABOUTME: Covers unit tests for evaluation logic and integration tests for CLI output formatting.

import json
import pytest
from moulinette.evaluate_retrieval import (
    build_eval_objects,
    calculate_recall_at_k_for_one_question,
)
from moulinette.models import (
    MinimalSource,
    MinimalSearchResults,
    AnsweredQuestion,
    RagDataset,
    StudentSearchResults,
)
from moulinette.__main__ import Moulinette
from tests.conftest import make_source


# ---------------------------------------------------------------------------- #
#                            Unit tests (logic)                                 #
# ---------------------------------------------------------------------------- #


class TestBuildEvalObjects:
    def test_correct_mapping(self, sample_rag_dataset, sample_student_results_all_found):
        """EvalObjects are keyed by question_id with matching sources."""
        eval_objects = build_eval_objects(sample_student_results_all_found, sample_rag_dataset)

        assert set(eval_objects.keys()) == {"q1", "q2", "q3"}
        # q1 has 2 true sources and 3 pred sources (including extra)
        assert len(eval_objects["q1"].true_sources) == 2
        assert len(eval_objects["q1"].pred_sources) == 3
        # q2 has 1 true and 1 pred
        assert len(eval_objects["q2"].true_sources) == 1
        assert len(eval_objects["q2"].pred_sources) == 1

    def test_missing_student_answer_gets_empty_pred(self, sample_rag_dataset):
        """Questions without student results get empty pred_sources."""
        empty_student = StudentSearchResults(k=10, search_results=[])
        eval_objects = build_eval_objects(empty_student, sample_rag_dataset)

        for eval_obj in eval_objects.values():
            assert eval_obj.pred_sources == []


class TestRecallForOneQuestion:
    def test_all_sources_found(self):
        """recall == 1.0 when all true sources are found."""
        true = [make_source("a.py", 0, 100), make_source("b.py", 0, 50)]
        pred = [make_source("a.py", 0, 100), make_source("b.py", 0, 50)]
        assert calculate_recall_at_k_for_one_question(pred, true) == 1.0

    def test_no_sources_found(self):
        """recall == 0.0 when no true sources are found."""
        true = [make_source("a.py", 0, 100)]
        pred = [make_source("z.py", 0, 100)]
        assert calculate_recall_at_k_for_one_question(pred, true) == 0.0

    def test_partial_sources(self):
        """recall == 0.5 when half the sources are found."""
        true = [make_source("a.py", 0, 100), make_source("b.py", 0, 50)]
        pred = [make_source("a.py", 0, 100), make_source("z.py", 0, 999)]
        assert calculate_recall_at_k_for_one_question(pred, true) == 0.5

    def test_k_truncation(self):
        """Correct source at index 5 is excluded when k=3 (caller truncates)."""
        true = [make_source("target.py", 0, 100)]
        pred = [
            make_source("x1.py", 0, 10),
            make_source("x2.py", 0, 10),
            make_source("x3.py", 0, 10),
            make_source("x4.py", 0, 10),
            make_source("x5.py", 0, 10),
            make_source("target.py", 0, 100),  # index 5, beyond k=3
        ]
        # With all 6 -> found
        assert calculate_recall_at_k_for_one_question(pred, true) == 1.0
        # With only first 3 -> not found
        assert calculate_recall_at_k_for_one_question(pred[:3], true) == 0.0

    def test_empty_true_sources(self):
        """Returns 1.0 when there are no true sources (existing behavior)."""
        pred = [make_source("a.py", 0, 100)]
        assert calculate_recall_at_k_for_one_question(pred, []) == 1.0


# ---------------------------------------------------------------------------- #
#                     Integration tests (CLI output formatting)                 #
# ---------------------------------------------------------------------------- #


def _write_fixtures_to_disk(tmp_path, student_results, rag_dataset):
    """Helper: write student results and dataset to JSON files on disk."""
    student_path = tmp_path / "student_results.json"
    dataset_path = tmp_path / "dataset.json"
    student_path.write_text(student_results.model_dump_json(indent=2))
    dataset_path.write_text(rag_dataset.model_dump_json(indent=2))
    return str(student_path), str(dataset_path)


class TestListValidQuestionsOutput:
    def test_output_contains_valid_tags_all_found(
        self, capsys, tmp_path, sample_rag_dataset, sample_student_results_all_found
    ):
        """When all sources found, all questions show [VALID] and summary shows 3/3."""
        student_path, dataset_path = _write_fixtures_to_disk(
            tmp_path, sample_student_results_all_found, sample_rag_dataset
        )
        m = Moulinette()
        m.list_valid_questions(student_path, dataset_path, k=10)

        output = capsys.readouterr().out
        assert "[VALID]" in output
        assert "[INVALID]" not in output
        assert "3/3" in output

    def test_output_contains_invalid_tags_partial(
        self, capsys, tmp_path, sample_rag_dataset, sample_student_results_partial
    ):
        """Partial results produce a mix of VALID and INVALID tags."""
        student_path, dataset_path = _write_fixtures_to_disk(
            tmp_path, sample_student_results_partial, sample_rag_dataset
        )
        m = Moulinette()
        m.list_valid_questions(student_path, dataset_path, k=10)

        output = capsys.readouterr().out
        assert "[VALID]" in output
        assert "[INVALID]" in output
        # q2 is the only fully valid one (recall=1.0)
        assert "1/3" in output

    def test_partial_mode_more_valid(
        self, capsys, tmp_path, sample_rag_dataset, sample_student_results_partial
    ):
        """require_all_sources=False makes partial matches (recall>0) count as valid."""
        student_path, dataset_path = _write_fixtures_to_disk(
            tmp_path, sample_student_results_partial, sample_rag_dataset
        )
        m = Moulinette()
        m.list_valid_questions(
            student_path, dataset_path, k=10, require_all_sources=False
        )

        output = capsys.readouterr().out
        # q1 recall=0.5 (valid), q2 recall=1.0 (valid), q3 recall=0.0 (invalid)
        assert "2/3" in output
        assert "mode=any_source" in output

    def test_output_contains_question_str_in_quotes(
        self, capsys, tmp_path, sample_rag_dataset, sample_student_results_all_found
    ):
        """Question text appears in double quotes in the output."""
        student_path, dataset_path = _write_fixtures_to_disk(
            tmp_path, sample_student_results_all_found, sample_rag_dataset
        )
        m = Moulinette()
        m.list_valid_questions(student_path, dataset_path, k=10)

        output = capsys.readouterr().out
        assert '"How does vLLM handle batching?"' in output
        assert '"What is PagedAttention?"' in output
        assert '"How does prefix caching work?"' in output
