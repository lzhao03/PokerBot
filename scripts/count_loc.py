#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

SOURCE_EXTS = {
    ".cc",
    ".cpp",
    ".cxx",
    ".c",
    ".h",
    ".hpp",
    ".hh",
}

SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "generated",
    "dist",
    "build",
    ".next",
    ".cache",
}


def is_test_file(path: Path) -> bool:
    return "tests" in path.parts or path.stem.endswith("_test")


def count_lines(file_path: Path) -> tuple[int, int]:
    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return len(lines), sum(bool(line.strip()) for line in lines)


def source_files(root: Path, include_generated: bool) -> list[Path]:
    skipped = SKIP_DIRS - ({"generated"} if include_generated else set())
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SOURCE_EXTS
        and not any(part in skipped or part.startswith(".") for part in path.parts)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rudimentary LOC summary by file extension."
    )
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument(
        "--exclude-tests",
        action="store_true",
        help="Skip files ending with '_test' before the extension (e.g., _test.cc).",
    )
    parser.add_argument(
        "--include-generated",
        action="store_true",
        help="Include files under generated directories.",
    )
    parser.add_argument(
        "--split-tests",
        action="store_true",
        help="Report source and test file line counts separately.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    total_lines: dict[str, int] = defaultdict(int)
    code_lines: dict[str, int] = defaultdict(int)
    split_totals: dict[str, int] = defaultdict(int)
    split_code: dict[str, int] = defaultdict(int)
    split_files: dict[str, int] = defaultdict(int)

    file_count = 0
    for path in source_files(root, args.include_generated):
        if args.exclude_tests and is_test_file(path):
            continue
        try:
            total, code = count_lines(path)
        except OSError:
            continue

        ext = path.suffix.lower()
        total_lines[ext] += total
        code_lines[ext] += code
        file_count += 1
        if args.split_tests:
            bucket = "test" if is_test_file(path) else "source"
            split_totals[bucket] += total
            split_code[bucket] += code
            split_files[bucket] += 1

    grand_total = sum(total_lines.values())
    grand_code = sum(code_lines.values())

    print(f"Scanned {file_count} source-like files")
    print(f"Total lines: {grand_total}")
    print(f"Non-empty lines: {grand_code}")
    if args.split_tests:
        print("By source/test:")
        for bucket in ("source", "test"):
            print(
                f"  {bucket:6}  files={split_files[bucket]:>6}  "
                f"total={split_totals[bucket]:>10}  code={split_code[bucket]:>10}"
            )
    print("By extension:")
    for ext in sorted(total_lines):
        print(f"  {ext:6}  total={total_lines[ext]:>10}  code={code_lines[ext]:>10}")


if __name__ == "__main__":
    main()
