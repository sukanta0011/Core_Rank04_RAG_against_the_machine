"""CLI robustness tests: no input may produce an unhandled traceback."""
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parent.parent
OK_CODES = (0, 1)  # 1 is what the CLI returns after a handled error
HAS_INDEX = (ROOT / "data" / "processed").exists()

needs_index = pytest.mark.skipif(
    not HAS_INDEX, reason="run `index` first: data/processed missing")


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run `python -m src <args>` from the repo root.

    Args:
        *args: CLI arguments.

    Returns:
        The finished process with captured text output.
    """
    return subprocess.run(
        [sys.executable, "-m", "src", *args],
        capture_output=True, text=True, timeout=600, cwd=ROOT)


def assert_handled(result: subprocess.CompletedProcess) -> None:
    """Assert the CLI ended cleanly, with no traceback.

    Args:
        result: Process returned by `run_cli`.
    """
    combined = result.stdout + result.stderr
    assert "Traceback" not in combined, combined[-1500:]
    assert result.returncode in OK_CODES, (
        f"exit {result.returncode}\n{combined[-1500:]}")


def assert_error_reported(result: subprocess.CompletedProcess) -> None:
    """Assert a bad input was refused with a readable message."""
    assert_handled(result)
    assert result.returncode == 1, "bad input must exit with code 1"
    assert "Error:" in result.stdout + result.stderr


def write_json(path: Path, data: object) -> str:
    """Write `data` as JSON and return the path as a string."""
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


# ---------------- search (single query) ----------------

@pytest.mark.parametrize("query", ["", " ", "a", "ab", "   \t  "])
def test_search_empty_or_tiny_query(query: str) -> None:
    assert_error_reported(
        run_cli("search", "--query", query, "--k", "5"))


@pytest.mark.parametrize("k", ["0", "-1", "abc", "1.5", ""])
def test_search_bad_k(k: str) -> None:
    assert_error_reported(
        run_cli("search", "--query", "How to load a LoRA?", "--k", k))


def test_search_unknown_method() -> None:
    assert_error_reported(run_cli(
        "search", "--query", "How to load a LoRA?", "--method", "nope"))


@needs_index
def test_search_gibberish_query() -> None:
    assert_handled(
        run_cli("search", "--query", "asdkjh qwe zzzx", "--k", "5"))


@needs_index
def test_search_k_larger_than_corpus() -> None:
    assert_handled(
        run_cli("search", "--query", "LoRA adapter", "--k", "100000"))


# ---------------- answer (validation happens before the model loads) ----

@pytest.mark.parametrize("question", ["", "  ", "ab"])
def test_answer_empty_question(question: str) -> None:
    assert_error_reported(
        run_cli("answer", "--question", question, "--k", "3"))


def test_answer_k_zero() -> None:
    assert_error_reported(
        run_cli("answer", "--question", "How to load a LoRA?", "--k", "0"))


# ---------------- search_dataset ----------------

def test_search_dataset_missing_file(tmp_path: Path) -> None:
    assert_error_reported(run_cli(
        "search_dataset",
        "--dataset_path", str(tmp_path / "nope.json"),
        "--k", "5", "--save_directory", str(tmp_path / "out")))


@needs_index
def test_search_dataset_malformed_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    assert_error_reported(run_cli(
        "search_dataset", "--dataset_path", str(bad),
        "--k", "5", "--save_directory", str(tmp_path / "out")))


@needs_index
@pytest.mark.parametrize("content", [
    [], {}, {"rag_questions": "oops"}, {"rag_questions": [{"x": 1}]}, 42])
def test_search_dataset_wrong_structure(
        tmp_path: Path, content: object) -> None:
    path = write_json(tmp_path / "wrong.json", content)
    assert_handled(run_cli(
        "search_dataset", "--dataset_path", path,
        "--k", "5", "--save_directory", str(tmp_path / "out")))


@needs_index
def test_search_dataset_k_zero(tmp_path: Path) -> None:
    path = write_json(tmp_path / "ds.json", {"rag_questions": [
        {"question_id": "q1", "question": "How to load a LoRA adapter?"}]})
    assert_error_reported(run_cli(
        "search_dataset", "--dataset_path", path,
        "--k", "0", "--save_directory", str(tmp_path / "out")))


@needs_index
def test_search_dataset_output_file_and_batch_survives(
        tmp_path: Path) -> None:
    questions: List[dict] = [
        {"question_id": "q1", "question": "How to load a LoRA adapter?"},
        {"question_id": "q2", "question": ""},
        {"question_id": "q3", "question": "What is the OpenAI server?"},
    ]
    path = write_json(tmp_path / "ds.json", {"rag_questions": questions})
    out = tmp_path / "out"
    result = run_cli(
        "search_dataset", "--dataset_path", path,
        "--k", "3", "--save_directory", str(out))
    # print(result.returncode, result.stdout[-800:], result.stderr[-800:])
    assert_handled(result)
    # The spec: output is <save_directory>/<same file name as the dataset>
    saved = out / "ds.json"
    assert saved.exists(), "output must be <save_directory>/ds.json"
    results = json.loads(saved.read_text())["search_results"]
    ids = [r["question_id"] for r in results]
    assert "q1" in ids and "q3" in ids
    for r in results:
        for s in r["retrieved_sources"]:
            assert Path(ROOT / s["file_path"]).is_file(), s["file_path"]
            span = s["last_character_index"] - s["first_character_index"]
            assert 0 < span <= 2000


# ---------------- answer_dataset ----------------

def test_answer_dataset_missing_file(tmp_path: Path) -> None:
    assert_error_reported(run_cli(
        "answer_dataset",
        "--student_search_results_path", str(tmp_path / "nope.json"),
        "--save_directory", str(tmp_path / "out")))


# ---------------- evaluate ----------------

def test_evaluate_missing_files(tmp_path: Path) -> None:
    assert_error_reported(run_cli(
        "evaluate",
        "--student_search_results_path", str(tmp_path / "a.json"),
        "--dataset_path", str(tmp_path / "b.json")))


def test_evaluate_bad_k(tmp_path: Path) -> None:
    a = write_json(tmp_path / "a.json", {})
    b = write_json(tmp_path / "b.json", {})
    assert_error_reported(run_cli(
        "evaluate", "--student_search_results_path", a,
        "--dataset_path", b, "--k", "0"))


def test_evaluate_malformed_json(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    a.write_text("[[[", encoding="utf-8")
    b = write_json(tmp_path / "b.json", {"rag_questions": []})
    assert_handled(run_cli(
        "evaluate", "--student_search_results_path", str(a),
        "--dataset_path", b))


# ---------------- unknown command ----------------

def test_unknown_command() -> None:
    combined = run_cli("does_not_exist")
    assert "Traceback" not in combined.stdout + combined.stderr
