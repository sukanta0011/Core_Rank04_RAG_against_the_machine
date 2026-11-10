import subprocess
import sys
import time

# Every module you import, including the ones commented out in __main__.py
MODULES = [
    "fire",
    "torch",
    "sentence_transformers",
    "chromadb",
    "transformers",
    "bm25s",
    "langchain_text_splitters",
    "pydantic",
    "src.data_retrieval.helper_classes",
    "src.data_retrieval.chunk_data",
    "src.data_retrieval.abstract_classes",
    "src.data_retrieval.lexical_retriever",
    "src.data_retrieval.semantic_retriever",
    "src.data_retrieval.hybrid_retriever",
    "src.data_retrieval.resource_refiner",
    "src.validator.resource_validator",
    "src.parsing.parse_rag_dataset",
    "src.base_patterns",
    "src.answer_generation.models.qwen3__0_6B",
    "src.answer_generation.pre_prompt",
    "src.answer_generation.answer",
    "src.__main__",
]


def time_import(module: str, repeats: int = 2) -> float | None:
    """Best of `repeats` wall-clock times for importing `module` alone."""
    best = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            capture_output=True,
        )
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            return None
        best = elapsed if best is None else min(best, elapsed)
    return best


def main() -> None:
    baseline = time_import("sys") or 0.0
    print(f"{'module':50s}{'seconds':>9s}{'net':>9s}")
    print(f"{'(empty python startup)':50s}{baseline:9.2f}{'':>9s}")
    rows = []
    for mod in MODULES:
        t = time_import(mod)
        if t is None:
            print(f"{mod:50s}{'FAILED':>9s}")
            continue
        rows.append((mod, t))
        print(f"{mod:50s}{t:9.2f}{t - baseline:9.2f}")
    print("\nSlowest 5 (net of startup):")
    for mod, t in sorted(rows, key=lambda r: r[1], reverse=True)[:5]:
        print(f"  {t - baseline:6.2f}s  {mod}")


if __name__ == "__main__":
    main()