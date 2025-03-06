"""
This script processes the adjudication data produced by Prodigy for our
poetry detection task into two outputs:

    1. A JSONL file that compiles the annotation data into page-level records.
       So, each record contains some page-level metdata and the compiled list
       of poetry excerpts (if any) determined in the adjudication process.

    2. A CSV file containing excerpt-level data per line.

Note that the first file explicitly include information on the pages where
no poetry was identified, while the second will only implicitly through
absence and requires external knowledge of what pages were covered in
the annotation rounds. So, the former is particularly useful for the evaluation
process while the latter is better suited for building a final excerpt dataset.

Example command line usage:
```
python process_adjudication_data.py prodigy_data.jsonl adj_pages.jsonl adj_excerpts.csv
```
"""

import argparse
import csv
import pathlib
import sys
from collections.abc import Iterable
from typing import Any

import orjsonl
from tqdm import tqdm
from xopen import xopen

from corppa.poetry_detection.core import Excerpt


def get_excerpts(page_annotation: dict[str, Any]) -> list[Excerpt]:
    """
    Extract excerpts from page-level annotation. Excerpts have the following
        * start: character-level starting index
        * end: character-level end index (Pythonic, exclusive)
        * text: text of page excerpt

    Note: This ignores span labels since there's only one for the
          poetry detection task.
    """
    excerpts = []
    # Blank pages may not have a text field, so in these cases set to empty string
    page_text = page_annotation.get("text", "")
    if "spans" not in page_annotation:
        raise ValueError("Page annotation missing 'spans' field")
    for span in page_annotation["spans"]:
        excerpt = Excerpt(
            page_id=page_annotation["id"],
            ppa_span_start=span["start"],
            ppa_span_end=span["end"],
            ppa_span_text=page_text[span["start"] : span["end"]],
            detection_methods={"adjudication"},
        )
        excerpts.append(excerpt.strip_whitespace())
    return excerpts


def process_page_annotation(page_annotation) -> dict[str, Any]:
    """
    Extracts desired content from page-level annotation. The returned data has
    the following fields"
        * page_id: Page's PPA page identifier
        * work_id: PPA work identifier
        * work_title: Title of PPA work
        * work_author: Author of PPA work
        * work_year: Publication of PPA work
        * n_excerpts: Number of poetry excerpts contained in page
        * excerpts: List of poetry excerpts identified within page
    """
    page_data = {}
    page_data["page_id"] = page_annotation["id"]
    page_data["work_id"] = page_annotation["work_id"]
    page_data["work_title"] = page_annotation["meta"]["title"]
    page_data["work_author"] = page_annotation["meta"]["author"]
    page_data["work_year"] = page_annotation["meta"]["year"]
    page_data["excerpts"] = get_excerpts(page_annotation)
    page_data["n_excerpts"] = len(page_data["excerpts"])
    return page_data


def simplify_excerpts(excerpts: Iterable[Excerpt]) -> list[dict[str, Any]]:
    """
    Converts excerpts into a simplified form with the following fields:
        * start = starting index of PPA span
        * end = ending index of PPA span
        * text = text of PPA span
    """
    simplified_excerpts = []
    for excerpt in excerpts:
        simple_excerpt = {
            "start": excerpt.ppa_span_start,
            "end": excerpt.ppa_span_end,
            "text": excerpt.ppa_span_text,
        }
        simplified_excerpts.append(simple_excerpt)
    return simplified_excerpts


def process_adjudication_data(
    input_jsonl: pathlib.Path,
    output_pages: pathlib.Path,
    output_excerpts: pathlib.Path,
    disable_progress: bool = False,
) -> None:
    """
    Process adjudication annotation data and write output files containing page-level
    and excerpt-level information that are JSONL and CSV files respectively.
    """
    n_lines = sum(1 for line in xopen(input_jsonl, mode="rb"))
    progress_annos = tqdm(
        orjsonl.stream(input_jsonl),
        total=n_lines,
        disable=disable_progress,
    )
    csv_fieldnames = [
        "page_id",
        "excerpt_id",
        "ppa_span_start",
        "ppa_span_end",
        "ppa_span_text",
        "detection_methods",
        "notes",
    ]
    with open(output_excerpts, mode="w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        csv_writer.writeheader()
        for page_anno in progress_annos:
            page_data = process_page_annotation(page_anno)

            # Write excerpt-level data
            for excerpt in page_data["excerpts"]:
                csv_writer.writerow(excerpt.to_csv())

            # Simplify excerpts
            # TODO: This is needed for compatiblity with existing evaluation code
            page_data["excerpts"] = simplify_excerpts(page_data["excerpts"])
            # Write page-level data
            orjsonl.append(output_pages, page_data)


def main():
    """
    Extracts page- and excerpt-level data from a Prodigy data file (JSONL)
    and writes the page-level excerpt data to a JSONL (`output_pages`) and the
    excerpt-level data to a CSV (`output_excerpts`).
    """
    parser = argparse.ArgumentParser(
        description="Extracts & saves page- and excerpt-level data from Prodigy data file",
    )
    parser.add_argument(
        "input",
        help="Path to Prodigy annotation data export (JSONL file)",
        type=pathlib.Path,
    )
    parser.add_argument(
        "output_pages",
        help="Filename where extracted page-level data (JSONL file) should be written",
        type=pathlib.Path,
    )
    parser.add_argument(
        "output_excerpts",
        help="Filename where extracted excerpt-level data (CSV file) should be written",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--progress",
        help="Show progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()
    disable_progress = not args.progress

    # Check that input file exists
    if not args.input.is_file():
        print(
            f"Error: input file {args.input.is_file()} does not exist", file=sys.stderr
        )
        sys.exit(1)

    # Check that output files does not exist
    for output_file in [args.output_pages, args.output_excerpts]:
        if output_file.exists():
            print(
                f"Error: output file {output_file} already exists, not overwriting",
                file=sys.stderr,
            )
            sys.exit(1)

    process_adjudication_data(
        args.input,
        args.output_pages,
        args.output_excerpts,
        disable_progress=disable_progress,
    )


if __name__ == "__main__":
    main()
