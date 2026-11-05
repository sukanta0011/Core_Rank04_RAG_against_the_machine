import json
import sys
from collections import Counter
from pathlib import Path


def read_raw(path: str) -> str:
    """Strict UTF-8 read, no newline translation."""
    with open(path, "r", encoding="utf-8", newline="") as fl:
        return fl.read()


def first_diff(a: str, b: str) -> int:
    """Index of the first differing character (or shorter length)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def run(folder: str) -> None:
    with open(Path(folder) / "text.json", "r") as fl:
        texts = json.load(fl)
    with open(Path(folder) / "minimal_source.json", "r") as fl:
        infos = json.load(fl)

    raw_cache: dict[str, str] = {}
    bad_per_file: Counter[str] = Counter()
    example_per_ext: dict[str, str] = {}

    for text, info in zip(texts, infos):
        path = info["file_path"]
        first = info["first_character_index"]
        last = info["last_character_index"]

        if path not in raw_cache:
            raw_cache[path] = read_raw(path)
        sliced = raw_cache[path][first:last]

        if sliced != text:
            bad_per_file[path] += 1
            ext = Path(path).suffix or "(none)"
            if ext not in example_per_ext:
                d = first_diff(text, sliced)
                lo = max(0, d - 20)
                example_per_ext[ext] = (
                    f"{path} [{first}:{last}] diff at {d}\n"
                    f"      stored: {text[lo:d + 20]!r}\n"
                    f"      sliced: {sliced[lo:d + 20]!r}"
                )

    # Per-file report with CRLF check
    with_crlf = 0
    ext_files: Counter[str] = Counter()
    ext_chunks: Counter[str] = Counter()
    for path, n in bad_per_file.items():
        with open(path, "rb") as fl:
            has_crlf = b"\r\n" in fl.read()
        with_crlf += has_crlf
        ext = Path(path).suffix or "(none)"
        ext_files[ext] += 1
        ext_chunks[ext] += n

    print(f"distinct files with mismatches: {len(bad_per_file)}")
    print(f"...of which contain \\r\\n:       {with_crlf}")
    print()
    print(f"{'ext':8s}{'files':>6s}{'bad chunks':>12s}")
    for ext, nf in ext_files.most_common():
        print(f"{ext:8s}{nf:6d}{ext_chunks[ext]:12d}")
    print()
    print("One example per extension:")
    for ext, ex in example_per_ext.items():
        print(f"  [{ext}] {ex}")


if __name__ == "__main__":
    run("data/chunks")
