"""
Gather excerpts from passim page-level results.
"""

import argparse
import csv
import sys
from collections.abc import Generator
from pathlib import Path

import orjsonl

from corppa.poetry_detection.core import LabeledExcerpt


def get_passim_excerpts(input_file: Path) -> Generator[LabeledExcerpt]:
    """
    Extracts and yields the passim-identified passage-level excerpts
    """
    for page in orjsonl.stream(input_file):
        if not page["n_spans"]:
            # Skipe pages without matches
            continue
        page_id = page["page_id"]
        for poem_span in page["poem_spans"]:
            excerpt = LabeledExcerpt(
                page_id=page_id,
                ppa_span_start=poem_span["page_start"],
                ppa_span_end=poem_span["page_end"],
                ppa_span_text=poem_span["page_excerpt"],
                detection_methods={"passim"},
                poem_id=poem_span["ref_id"],
                ref_corpus=poem_span["ref_corpus"],
                ref_span_start=poem_span["ref_start"],
                ref_span_end=poem_span["ref_end"],
                ref_span_text=poem_span["ref_excerpt"],
                identification_methods={"passim"},
            )
            yield excerpt


def save_passim_excerpts(input_file: Path, output_file: Path) -> None:
    with open(output_file, mode="w", newline="") as csvfile:
        fieldnames = [
            "page_id",
            "excerpt_id",
            "ppa_span_start",
            "ppa_span_end",
            "ppa_span_text",
            "poem_id",
            "ref_corpus",
            "ref_span_start",
            "ref_span_end",
            "ref_span_text",
            "detection_methods",
            "identification_methods",
            "notes",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for excerpt in get_passim_excerpts(input_file):
            writer.writerow(excerpt.to_csv())


def main():
    """
    Command-line access to build CSV file of passage-level passim matches.
    """
    parser = argparse.ArgumentParser(
        description="Extract passage-level passim results (CSV)"
    )

    # Required arguments
    parser.add_argument(
        "input",
        help="Page-level passim results file (JSONL)",
        type=Path,
    )
    parser.add_argument(
        "output",
        help="Filename for passage-level passim output (CSV)",
        type=Path,
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.input.is_file():
        print(f"Error: input {args.input} does not exist", file=sys.stderr)
        sys.exit(1)
    if args.output.is_file():
        print(f"Error: output file {args.output} exist", file=sys.stderr)
        sys.exit(1)

    save_passim_excerpts(args.input, args.output)


if __name__ == "__main__":
    main()
