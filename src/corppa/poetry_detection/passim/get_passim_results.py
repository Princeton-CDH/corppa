"""
Build passim page-level and span-level matches from passim output files.

Examples:
    get_passim_results.py --ppa-corpus ppa.jsonl --ref-corpus ref.jsonl \
        --passim-dir passim_output --page-results passim_page_results.jsonl \
        --span-results passim_spans.tsv
    get_passim_results.py --ppa-corpus ppa.jsonl --ref-corpus ref_a.jsonl \
        --ref-corpus ref_b.jsonl --passim-dir passim_output \
        --page-results passim_page_results.jsonl --span-results passim_spans.tsv
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict

import orjsonl
from tqdm import tqdm


def get_passim_span(alignment_record):
    """
    Extract span record from and rename fields from passim alignment record into a new
    page-level record.
    """
    span_record = {
        "ref_id": alignment_record["id"],
        "ref_corpus": alignment_record["corpus"],
        "ref_start": alignment_record["begin"],
        "ref_end": alignment_record["end"],
        "page_id": alignment_record["id2"],
        "page_start": alignment_record["begin2"],
        "page_end": alignment_record["end2"],
        "matches": alignment_record["matches"],
        # Note: "aligned" excerpts use "-" to indicate insertions
        "aligned_ref_excerpt": re.sub(r"\s", " ", alignment_record["s1"]),
        "aligned_ppa_excerpt": re.sub(r"\s", " ", alignment_record["s2"]),
    }
    return span_record


def extract_passim_spans(
    passim_dir: Path,
    include_excerpts: bool = False,
    disable_progress: bool = False,
):
    """
    Exctracts all span-level matches identified by passim returned as a generator
    """
    align_dir = passim_dir.joinpath("align.json")
    if not align_dir.is_dir():
        raise ValueError("Error: Alignment directory '{align.json}' does not exist.")
    for filepath in align_dir.glob("*.json"):
        record_progress = tqdm(
            orjsonl.stream(filepath),
            desc=f"Extracting matches from {filepath.name}",
            disable=disable_progress,
        )
        for record in record_progress:
            yield get_passim_span(record)


def add_excerpts(
    page_results,
    ppa_corpus: Path,
    ref_corpora: Iterable[Path],
    disable_progress: bool = False,
) -> None:
    """
    Add original PPA and reference excerpts to the span within page results
    """
    # For tracking page reuse by reference text
    refs_to_pages = defaultdict(lambda: defaultdict(set))

    # Add PPA excerpts
    ppa_progress = tqdm(
        orjsonl.stream(ppa_corpus),
        total=len(page_results),
        desc="Adding PPA excerpts",
        disable=disable_progress,
    )
    for ppa_record in ppa_progress:
        page_id = ppa_record["id"]
        poem_spans = page_results[page_id]["poem_spans"]
        # Add PPA excerpt to each poem span
        for span in poem_spans:
            start, end = span["page_start"], span["page_end"]
            span["ppa_excerpt"] = ppa_record["text"][start:end]
            # Add the page_id to the referenced text
            corpus_id = span["ref_corpus"]
            ref_id = span["ref_id"]
            refs_to_pages[corpus_id][ref_id].add(page_id)

    # Add reference excerpts
    for ref_corpus in ref_corpora:
        ref_progress = tqdm(
            orjsonl.stream(ref_corpus),
            desc=f"Adding {ref_corpus} excerpts",
            disable=disable_progress,
        )
        for ref_record in ref_progress:
            ref_corpus = ref_record["corpus"]
            # Skip unreference corpora
            if ref_corpus not in refs_to_pages:
                print(f"Warning: {ref_corpus} is never reused", file=sys.stderr)
            # Skip unreferenced texts
            ref_id = ref_record["id"]
            if ref_id not in refs_to_pages[ref_corpus]:
                continue
            for page_id in refs_to_pages[ref_corpus][ref_id]:
                # Add reference excerpts to corresponding spans
                for span in page_results[page_id]["poem_spans"]:
                    if span["ref_corpus"] == ref_corpus and span["ref_id"] == ref_id:
                        start, end = span["ref_start"], span["ref_end"]
                        span["ref_excerpt"] = ref_record["text"][start:end]


def build_passim_page_results(
    ppa_corpus: Path,
    ref_corpora: Iterable[Path],
    passim_dir: Path,
    disable_progress: bool = False,
):
    # Initialize page-level results
    page_results = {}
    page_progress = tqdm(
        orjsonl.stream(ppa_corpus),
        desc="Initializing PPA page-level results",
        disable=disable_progress,
    )
    for record in page_progress:
        page_id = record["id"]
        page_results[page_id] = {"page_id": page_id, "n_spans": 0, "poem_spans": []}

    # Add passage-level matches to page-level records
    for match in extract_passim_spans(passim_dir, disable_progress=disable_progress):
        page_id = match.pop("page_id")
        page_results[page_id]["poem_spans"].append(match)
        page_results[page_id]["n_spans"] += 1

    # Add excerpts to span records
    add_excerpts(
        page_results, ppa_corpus, ref_corpora, disable_progress=disable_progress
    )

    return page_results


def write_passim_results(
    ppa_corpus: Path,
    ref_corpora: Iterable[Path],
    passim_dir: Path,
    out_page_results: Path,
    out_span_results: Path,
    disable_progress: bool = False,
) -> None:
    page_results = build_passim_page_results(
        ppa_corpus, ref_corpora, passim_dir, disable_progress=disable_progress
    )

    # Write page-level & span-level output by page
    page_progress = tqdm(
        page_results.items(),
        desc="Writing results",
        disable=disable_progress,
    )

    with open(out_span_results, mode="w", newline="") as csvfile:
        fieldnames = [
            "ref_id",
            "ref_corpus",
            "ref_start",
            "ref_end",
            "page_id",
            "page_start",
            "page_end",
            "matches",
            "ref_excerpt",
            "ppa_excerpt",
            "aligned_ref_excerpt",
            "aligned_ppa_excerpt",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, dialect="excel-tab")
        writer.writeheader()

        for page_id, record in page_progress:
            # Write page-level results to file
            orjsonl.append(out_page_results, record)

            # Write span-level results to file
            for span in record["poem_spans"]:
                match = span.copy()
                match["page_id"] = page_id
                writer.writerow(match)


def main():
    """
    Command-line access to build page-level and span-level passim results.
    """
    parser = argparse.ArgumentParser(description="Build passim results.")
    # Required arguments
    parser.add_argument(
        "--ppa-corpus",
        help="Path to PPA passim-friendly corpus file (JSONL)",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--ref-corpus",
        help="Path to reference passim-friendly corpus file (JSONL). Can specify multiple",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--passim-dir",
        help="The top-level directory containing the ouput of the passim run",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--page-results",
        help="Filename for the page-level passim results (JSONL)",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--span-results",
        help="Filename for the span-level passim results (TSV)",
        type=Path,
        required=True,
    )

    # Optional arguments
    parser.add_argument(
        "--progress",
        help="Show progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()

    # Validate input/output paths
    if not args.ppa_corpus.is_file():
        print(
            f"Error: PPA corpus {args.ppa_corpus} does not exist",
            file=sys.stderr,
        )
        sys.exit(1)
    for ref in args.ref_corpus:
        if not ref.is_file():
            print(
                f"Error: reference corpus {ref} does not exist",
                file=sys.stderr,
            )
            sys.exit(1)
    if not args.passim_dir.is_dir():
        print(
            f"Error: Passim directory {args.passim_dir} does not exist",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.page_results.is_file():
        print(
            f"Error: Output page-level results file {args.page_results} exists",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.span_results.is_file():
        print(
            f"Error: Output span-level results file {args.span_results} exists",
            file=sys.stderr,
        )
        sys.exit(1)

    write_passim_results(
        args.ppa_corpus,
        args.ref_corpus,
        args.passim_dir,
        args.page_results,
        args.span_results,
        not args.progress,
    )


if __name__ == "__main__":
    main()
