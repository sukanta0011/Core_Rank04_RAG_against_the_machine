"""Retrieval tests: every method must return valid, useful sources."""
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest

from src.base_patterns import StudentSearchResults

ROOT = Path(__file__).resolve().parent.parent
METHODS = ["lexical", "semantic", "hybrid"]
K = 5
SAMPLE = 30            # questions per method (keeps the suite fast)
MAX_SPAN = 2000
IOU_BAR = 0.05
# Margins below your real scores, so a regression fails the test.
MIN_RECALL = {"docs": 0.70, "code": 0.40}
DATASETS = {
    "docs": "dataset_docs_public.json",
    "code": "dataset_code_public.json",
}
CANDIDATES = [
    ROOT / "data" / "datasets" / "AnsweredQuestions",
    ROOT / "datasets_public" / "public" / "AnsweredQuestions",
]
GT_DIR = next((p for p in CANDIDATES if p.is_dir()), CANDIDATES[0])

pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "processed").exists(),
    reason="run `index` first")


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run `python -m src <args>` from the repo root."""
    return subprocess.run(
        [sys.executable, "-m", "src", *args],
        capture_output=True, text=True, timeout=1800, cwd=ROOT)


def read_raw(path: Path) -> str:
    """Read a file exactly as the grader would (no newline change)."""
    with open(path, "r", encoding="utf-8", newline="") as fl:
        return fl.read()


def iou(a: Dict, b: Dict) -> float:
    """IoU of two character spans given as source dicts."""
    inter = max(0, min(a["last_character_index"], b["last_character_index"])
                - max(a["first_character_index"],
                      b["first_character_index"]))
    union = ((a["last_character_index"] - a["first_character_index"])
             + (b["last_character_index"] - b["first_character_index"])
             - inter)
    return inter / union if union > 0 else 0.0


def recall_at_k(results: List[Dict], truth: Dict[str, Dict]) -> float:
    """Mean per-question recall: same file and IoU >= IOU_BAR."""
    scores = []
    for res in results:
        sources = truth[res["question_id"]]["sources"]
        found = sum(
            any(r["file_path"] == t["file_path"] and iou(t, r) >= IOU_BAR
                for r in res["retrieved_sources"][:K])
            for t in sources)
        scores.append(found / len(sources))
    return sum(scores) / len(scores)


@pytest.fixture(scope="module", params=list(DATASETS))
def kind(request: pytest.FixtureRequest) -> str:
    """Run every test for the docs and the code dataset."""
    return str(request.param)


@pytest.fixture(scope="module")
def truth(kind: str) -> Dict[str, Dict]:
    """Ground truth for the first SAMPLE questions: id -> question dict."""
    gt_file = GT_DIR / DATASETS[kind]
    if not gt_file.is_file():
        pytest.fail(f"ground truth not found: {gt_file}")
    questions = json.loads(gt_file.read_text(encoding="utf-8"))[
        "rag_questions"][:SAMPLE]
    return {q["question_id"]: q for q in questions}


@pytest.fixture(scope="module")
def outputs(tmp_path_factory: pytest.TempPathFactory, kind: str,
            truth: Dict[str, Dict]) -> Dict[str, Dict]:
    """Search once per (method, dataset); return the parsed outputs."""
    base = tmp_path_factory.mktemp(kind)
    ds = base / DATASETS[kind]
    ds.write_text(json.dumps({"rag_questions": [
        {"question_id": q["question_id"], "question": q["question"]}
        for q in truth.values()]}), encoding="utf-8")
    result: Dict[str, Dict] = {}
    for method in METHODS:
        out = base / method
        proc = run_cli(
            "search_dataset", "--dataset_path", str(ds), "--k", str(K),
            "--method", method, "--save_directory", str(out))
        saved = out / DATASETS[kind]
        if not saved.exists():
            result[method] = {"error": proc.stdout[-600:] + proc.stderr[-600:]}
        else:
            result[method] = json.loads(saved.read_text(encoding="utf-8"))
    return result


def get(outputs: Dict[str, Dict], method: str) -> Dict:
    """Return a method's output, failing clearly if the run failed."""
    out = outputs[method]
    assert "error" not in out, f"{method} run failed:\n{out['error']}"
    return out


@pytest.mark.parametrize("method", METHODS)
def test_output_matches_spec_model(outputs: Dict, method: str) -> None:
    """The file validates against the spec's StudentSearchResults."""
    parsed = StudentSearchResults(**get(outputs, method))
    assert parsed.k == K


@pytest.mark.parametrize("method", METHODS)
def test_one_result_per_question(
        outputs: Dict, truth: Dict, method: str) -> None:
    """Ids and question text pass through unchanged, none are lost."""
    results = get(outputs, method)["search_results"]
    assert {r["question_id"] for r in results} == set(truth)
    for r in results:
        assert r["question"] == truth[r["question_id"]]["question"]


@pytest.mark.parametrize("method", METHODS)
def test_sources_are_valid(outputs: Dict, method: str) -> None:
    """Exactly K unique sources, real files, sane spans, real text."""
    for res in get(outputs, method)["search_results"]:
        srcs = res["retrieved_sources"]
        assert len(srcs) == K, res["question"]
        keys = [(s["file_path"], s["first_character_index"],
                 s["last_character_index"]) for s in srcs]
        assert len(set(keys)) == len(keys), "duplicate source"
        for s in srcs:
            path = ROOT / s["file_path"]
            assert not s["file_path"].startswith(("/", "./")), s
            assert "\\" not in s["file_path"], s
            assert path.is_file(), s["file_path"]
            first, last = (s["first_character_index"],
                           s["last_character_index"])
            assert 0 <= first < last <= len(read_raw(path)), s
            assert last - first <= MAX_SPAN, s


@pytest.mark.parametrize("method", METHODS)
def test_scores_are_json_numbers(outputs: Dict, method: str) -> None:
    """Scores (if present) are plain numbers, one per source."""
    for res in get(outputs, method)["search_results"]:
        scores = res.get("retrieved_sources_scores")
        if scores is not None:
            assert len(scores) == len(res["retrieved_sources"])
            assert all(isinstance(x, (int, float)) for x in scores)


@pytest.mark.parametrize("method", METHODS)
def test_recall_above_floor(
        outputs: Dict, truth: Dict, kind: str, method: str) -> None:
    """Recall@K (spec rule) stays above a floor; catches regressions."""
    value = recall_at_k(get(outputs, method)["search_results"], truth)
    print(f"{kind} {method} recall@{K} = {value:.3f}")
    assert value >= MIN_RECALL[kind], f"{kind}/{method}: {value:.3f}"


def test_hybrid_not_worse_than_best_single(
        outputs: Dict, truth: Dict, kind: str) -> None:
    """Hybrid should not lose clearly to its own components."""
    scores = {m: recall_at_k(get(outputs, m)["search_results"], truth)
              for m in METHODS}
    print(kind, scores)
    assert scores["hybrid"] >= max(scores["lexical"],
                                   scores["semantic"]) - 0.10


def test_deterministic(tmp_path: Path) -> None:
    """The same question twice gives the same sources (cache safe)."""
    ds = tmp_path / "ds.json"
    ds.write_text(json.dumps({"rag_questions": [
        {"question_id": "q1", "question": "How to load a LoRA adapter?"}
    ]}), encoding="utf-8")
    runs = []
    for i in range(2):
        out = tmp_path / f"out{i}"
        run_cli("search_dataset", "--dataset_path", str(ds), "--k", "5",
                "--save_directory", str(out))
        runs.append(json.loads((out / "ds.json").read_text())
                    ["search_results"][0]["retrieved_sources"])
    assert runs[0] == runs[1]
