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


def get_page_texts(page_ids: set[str], text_corpus: Path) -> dict[str, str]:
    """
    Gathers the texts from the PPA text corpus file for the specified pages (by id), returns
    a dictionary mapping page ids to page texts.
    """
    page_texts = {}
    for page in orjsonl.stream(text_corpus):
        page_id = page["id"]
        if page_id in page_ids:
            page_texts[page_id] = page["text"]
    return page_texts


def get_passim_excerpts(
    input_file: Path, ppa_text_corpus: None | Path = None
) -> Generator[LabeledExcerpt]:
    """
    Extracts and yields the passim-identified passage-level excerpts from passim page-level results file
    (as produced by `get_passim_page_results.py`). Optionally, can provide a path to the original PPA text
    corpus to "undo" the text transformations done during the passim pipeline.
    """
    # Optionally, gather relevant PPA page texts
    if ppa_text_corpus:
        relevant_ids = {
            page["page_id"] for page in orjsonl.stream(input_file) if page["n_spans"]
        }
        ppa_page_texts = get_page_texts(relevant_ids, ppa_text_corpus)

    # Gather excerpts
    for page in orjsonl.stream(input_file):
        print("foo")
        if not page["n_spans"]:
            # Skip pages without matches
            continue
        page_id = page["page_id"]
        # Get ppa page text
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
            if ppa_text_corpus and page_id in ppa_page_texts:
                # Correct excerpt if we have the original page text
                excerpt = excerpt.correct_page_excerpt(ppa_page_texts[page_id])
            yield excerpt


def save_passim_excerpts(
    input_file: Path, output_file: Path, ppa_text_corpus: None | Path = None
) -> None:
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
        for excerpt in get_passim_excerpts(input_file, ppa_text_corpus=ppa_text_corpus):
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
    # Optional arguments
    parser.add_argument(
        "--ppa-text-corpus",
        help="PPA text corpus file (JSONL) for correcting identified excerpts",
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
    if args.ppa_text_corpus and not args.ppa_text_corpus.is_file():
        print(
            f"Error: ppa text corpus {args.ppa_text_corpus} does not exist",
            file=sys.stderr,
        )
        sys.exit(1)

    save_passim_excerpts(args.input, args.output, ppa_text_corpus=args.ppa_text_corpus)


if __name__ == "__main__":
    main()
