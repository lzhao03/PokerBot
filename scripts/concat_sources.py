#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def source_files(directory: Path, output: Path) -> list[Path]:
    output = output.resolve()
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.resolve() != output
        and path.suffix.lower() in SOURCE_EXTS
        and "generated" not in path.parts
    )


def write_concat(root: Path, output: Path, files: list[Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out:
        for path in files:
            rel = path.relative_to(root)
            out.write(f"// ===== BEGIN {rel} =====\n")
            text = path.read_text(encoding="utf-8", errors="ignore")
            out.write(text)
            if not text.endswith("\n"):
                out.write("\n")
            out.write(f"// ===== END {rel} =====\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate actual source files into one annotated file."
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="source_bundle.txt",
        help="Output path, default: source_bundle.txt",
    )
    parser.add_argument(
        "--test-output",
        default="test_bundle.txt",
        help="Test output path, default: test_bundle.txt",
    )
    parser.add_argument(
        "--root", default=".", help="Repo root, default: current directory."
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    test_output = Path(args.test_output)
    if not test_output.is_absolute():
        test_output = root / test_output

    files = source_files(root / "src", output)
    test_files = source_files(root / "tests", test_output)
    write_concat(root, output, files)
    write_concat(root, test_output, test_files)
    display_output = output.relative_to(root) if output.is_relative_to(root) else output
    display_test_output = (
        test_output.relative_to(root)
        if test_output.is_relative_to(root)
        else test_output
    )
    print(f"Wrote {len(files)} files to {display_output}")
    print(f"Wrote {len(test_files)} files to {display_test_output}")


if __name__ == "__main__":
    main()
