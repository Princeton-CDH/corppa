"""
Evaluate the poetry spans detectede and identified by some *system*
against a provided *reference* set of span annotations.

Examples:
```
python evaluate_poetry_spans.py ref_spans.jsonl system_spans.jsonl \
        eval_results.csv
python evaluate_poetry_spans.py adjudicated_spans.jsonl passim_spans.jsonl \
        eval_results.csv
```
"""

import argparse
import csv
import sys
import typing
from collections.abc import Generator
from pathlib import Path

import orjsonl
from tqdm import tqdm
from xopen import xopen

from corppa.poetry_detection.core import Span


class PageReferenceSpans:
    """
    Page-level reference spans object
    """

    def __init__(self, page_json):
        self.page_id = page_json["page_id"]
        self.spans = self._get_spans(page_json)

    @staticmethod
    def _get_spans(page_json):
        """
        Get span list from page-level json
        """
        spans = []
        if page_json["n_excerpts"] > 0:
            for excerpt in page_json["excerpts"]:
                spans.append(Span(excerpt["start"], excerpt["end"], excerpt["poem_id"]))

        # Sort spans by primarily by start index and secondarily by end index
        spans.sort(key=lambda x: (x.start, x.end))
        return spans


class PageSystemSpans:
    """
    Page-level system spans object.

    Note: Unlike a page's reference spans, the spans produced by a system might
    overlap. While these overlapping spans are worth penalizing when span labels
    are taken in to account, this seems less true when labels are ignored.
    """

    def __init__(self, page_json):
        self.page_id = page_json["page_id"]
        self.labeled_spans = self._get_labeled_spans(page_json)
        self.unlabeled_spans = self._get_unlabeled_spans(self.labeled_spans)

    @staticmethod
    def _get_labeled_spans(page_json):
        """
        Get (labeled) spans from page-level json
        """
        spans = []
        if page_json["n_spans"] > 0:
            # TODO: Revisit the format for the system results jsonl
            for span in page_json["poem_spans"]:
                spans.append(Span(span["page_start"], span["page_end"], span["ref_id"]))
        # Sort spans by primarily by start index and secondarily by end index
        spans.sort(key=lambda x: (x.start, x.end))
        return spans

    @staticmethod
    def _get_unlabeled_spans(labeled_spans):
        """
        Get "label-free" spans derived from object's labeled spans. All overlapping
        (but not adjacent) spans are merged into a single span.
        """
        spans = []
        for i, labeled_span in enumerate(labeled_spans):
            unlabeled_span = Span(labeled_span.start, labeled_span.end, "")
            if not i:
                # Edge case: Add the unlabeled version of the first span
                spans.append(unlabeled_span)
            else:
                # If there is overlap with the previous span, merge.
                # Note: This relies on the ordering of labeled_spans
                prev_span = spans[-1]
                if prev_span.has_overlap(unlabeled_span):
                    prev_span.end = max(prev_span.end, unlabeled_span.end)
                else:
                    spans.append(unlabeled_span)
        return spans


class PageEvaluation:
    """
    Page-level span evaluation.
    """

    page_id: str
    ignore_label: bool  # Flag for ignoring span labels in evaluation
    ref_spans: list[Span]  # Reference spans
    sys_spans: list[Span]  # System spans
    ref_to_sys: list[int | None]  # Mapping from reference to system spans
    sys_to_refs: list[list[int]]  # Mapping from system to references spans
    span_pairs: list[tuple[Span, Span]]  # List of reference-system span pairs.

    def __init__(
        self,
        page_ref_spans: PageReferenceSpans,
        page_sys_spans: PageSystemSpans,
        ignore_label: bool = False,
    ) -> None:
        if page_ref_spans.page_id != page_sys_spans.page_id:
            raise ValueError(
                "Reference and system spans must correspond to the same page"
            )
        # Save working input
        self.page_id = page_ref_spans.page_id
        self.ignore_label = ignore_label
        self.ref_spans = page_ref_spans.spans
        if self.ignore_label:
            self.sys_spans = page_sys_spans.unlabeled_spans
        else:
            self.sys_spans = page_sys_spans.labeled_spans
        # Determine mappings between reference and system spans
        self.ref_to_sys, self.sys_to_refs = self._get_span_mappings(
            self.ref_spans, self.sys_spans, self.ignore_label
        )
        # Determine the reference-system span pairs
        self.span_pairs = self._get_span_pairs(
            self.ref_spans, self.sys_spans, self.sys_to_refs
        )

    @staticmethod
    def _get_span_mappings(
        ref_spans: list[Span],
        sys_spans: list[Span],
        ignore_label: bool,
    ) -> tuple[list[int | None], list[list[int]]]:
        """
        Determines the mappings between the reference and system spans. Each
        reference span is mapped to at most one system span. Namely, the system
        span with the highest degree of overlap. Conversely, each system span
        is mapped to any number of overlapping (but disjoint) reference spans.
        """
        ref_to_sys: list[int | None] = [None for _ in ref_spans]
        sys_to_refs: list[list[int]] = [[] for _ in sys_spans]

        # Assign each reference span to at most one system span
        for i, ref_span in enumerate(ref_spans):
            current_match_idx = None
            current_match_overlap: float = 0
            # Note: given these have an ordering this could be optimized
            for j, sys_span in enumerate(sys_spans):
                overlap = ref_span.overlap_factor(sys_span, ignore_label=ignore_label)
                if overlap > current_match_overlap:
                    current_match_idx = j
                    current_match_overlap = overlap
            # Update mappings if a match is found
            if current_match_idx is not None:
                ref_to_sys[i] = current_match_idx
                sys_to_refs[current_match_idx].append(i)
        return ref_to_sys, sys_to_refs

    @staticmethod
    def _get_span_pairs(
        ref_spans: list[Span],
        sys_spans: list[Span],
        sys_to_refs: list[list[int]],
    ) -> list[tuple[Span, Span]]:
        """
        Determines the reference-system span pairs used in evaluation.
        A span pair corresponds to a tuple of
            1. A reference span r mapped to system span s
            2. The system span s if s solely maps to r. Otherwise, a subspan of s.

        When a system span s maps to k > 1 reference spans r_i. It's split into
        subspans with the following ranges:
            (s start, r_1 start), (r_1 start, r_2 start), ..., (r_k start, s end)
        """
        span_pairs = []
        for sys_id, ref_ids in enumerate(sys_to_refs):
            sys_span = sys_spans[sys_id]
            if len(ref_ids) == 1:
                ref_span = ref_spans[ref_ids[0]]
                span_pairs.append((ref_span, sys_span))
            else:
                # Effectively, split system span into k pieces (one for each reference span)
                for i, ref_id in enumerate(ref_ids):
                    ref_span = ref_spans[ref_id]
                    start = ref_span.start if i else sys_span.start
                    if i == len(ref_ids) - 1:
                        end = sys_span.end
                    else:
                        next_ref_span = ref_spans[ref_ids[i + 1]]
                        end = next_ref_span.start
                    sub_sys_span = Span(start, end, sys_span.label)
                    span_pairs.append((ref_span, sub_sys_span))
        return span_pairs

    @staticmethod
    def _relevance_score(
        span_pairs: list[tuple[Span, Span]],
        ignore_label: bool,
        partial_match_weight: float,
    ) -> float:
        """
        Computes the relevance score for a set of reference-system span pairs.
        This score is used to calculate precision and recall. It corresponds to
        the effective number of relevant spans retrieved. Partial matches are
        awarded a fractional score corresponding to the span pairs overlap
        factor (`overlap_factor`). Optionally, partial matches can be further
        downweighted via `partial_match_weight`.
        """
        score: float = 0
        for ref_span, sys_span in span_pairs:
            if ref_span.is_exact_match(sys_span, ignore_label=ignore_label):
                score += 1
            else:
                overlap = ref_span.overlap_factor(sys_span, ignore_label=ignore_label)
                score += partial_match_weight * overlap
        return score

    @staticmethod
    def _retrieved_count(sys_to_refs: list[list[int]]) -> int:
        """
        Returns the number of (possibly split) spans retrieved by the system based
        on the input system to reference span mapping.
        """
        n_retrieved = 0
        for ref_ids in sys_to_refs:
            if ref_ids:
                n_retrieved += len(ref_ids)
            else:
                # Include spurious/incorrect spans
                n_retrieved += 1
        return n_retrieved

    def precision(self, partial_match_weight: float = 1) -> float:
        """
        Calculate page-level precision. Edge case: If there are no system spans,
        return 1 if there are also no reference spans and 0 otherwise.
        """
        if not self.sys_spans:
            # Edge case to avoid divide by zero error
            return 1 if not self.ref_spans else 0

        n_retrieved = self._retrieved_count(self.sys_to_refs)
        relevance_score = self._relevance_score(
            self.span_pairs, self.ignore_label, partial_match_weight
        )
        return relevance_score / n_retrieved

    def recall(self, partial_match_weight: float = 1) -> float:
        """
        Calculate page-level recall. Edge case: If there are no reference spans,
        return 1 if there are also no system spans and 0 otherwise.
        """
        if not self.ref_spans:
            # Edge case to avoid divide by zero error
            return 1 if not self.sys_spans else 0

        n_relevant = len(self.ref_spans)
        relevance_score = self._relevance_score(
            self.span_pairs, self.ignore_label, partial_match_weight
        )
        return relevance_score / n_relevant

    @staticmethod
    def _get_match_counts(
        ref_spans: list[Span], ref_to_sys: list[int | None]
    ) -> dict[str, int]:
        """
        Calculates partial match and miss counts at the span and poem level
        w.r.t reference spans. Returns results as a struct with the following
        fields:
            * n_span_matches: Number of reference spans with (partial) matches
            * n_span_misses: Number of reference spans with not match
            * n_poem_matches: Number of poems with at least one (partial) span match
            * n_poem_misses: Number of poems with no span match
        """
        n_span_matches = 0
        n_span_misses = 0
        poem_matches = set()
        poem_misses = set()
        for ref_idx, sys_idx in enumerate(ref_to_sys):
            if sys_idx is None:
                n_span_misses += 1
                poem_misses.add(ref_spans[ref_idx].label)
            else:
                n_span_matches += 1
                poem_matches.add(ref_spans[ref_idx].label)
        result = {
            "n_span_matches": n_span_matches,
            "n_span_misses": n_span_misses,
            "n_poem_matches": len(poem_matches),
            "n_poem_misses": len(poem_misses),
        }
        return result

    @staticmethod
    def _get_spurious_counts(
        ref_spans: list[Span], sys_spans: list[Span], sys_to_refs: list[list[int]]
    ) -> dict[str, int]:
        """
        Determines the number of spurious spans and poems identified by the system.
        Returns results as a struct with the following fields:
            * n_span_spurious: Number of spurious systems spans
            * n_poem_spurious: Number of spuriously identified poems
        """
        # Get set of reference poems
        ref_poems = {ref.label for ref in ref_spans}
        n_span_spurious = 0
        poem_spurious = set()
        for sys_idx, refs in enumerate(sys_to_refs):
            # Spurious spans are the ones mapped to no reference spans
            if not refs:
                n_span_spurious += 1
                poem_id = sys_spans[sys_idx].label
                # Spurious poems are one's not in the reference set
                if poem_id not in ref_poems:
                    poem_spurious.add(poem_id)
        result = {
            "n_span_spurious": n_span_spurious,
            "n_poem_spurious": len(poem_spurious),
        }
        return result

    def evaluate(self, partial_match_weight: float = 1) -> dict[str, str | float | int]:
        """
        Perform page-level evaluation and return results as a struct with the
        following fields:
            * page_id: page id
            * precision: precision score
            * recall: recalls score
            * n_span_matches: number of (partial) span matches
            * n_span_misses: number of span misses
            * n_span_spurious: number of spurious system spans
            * n_poem_matches: number of correctly identified poems
            * n_poem_misses: number of missed poems
            * n_poem_spurious: number of spuriously identified poems
        """
        # Determine span-level and poem-level (partial) match & miss counts
        match_results = self._get_match_counts(self.ref_spans, self.ref_to_sys)

        # Determine number of spurious spans and poems
        spurious_results = self._get_spurious_counts(
            self.ref_spans, self.sys_spans, self.sys_to_refs
        )

        # Calculate precision and recall
        precision = self.precision(partial_match_weight=partial_match_weight)
        recall = self.recall(partial_match_weight=partial_match_weight)

        result: dict[str, str | int | float] = {
            "page_id": self.page_id,
            "precision": precision,
            "recall": recall,
        }
        result.update(match_results)
        result.update(spurious_results)
        return result


def get_page_eval(ref_json, sys_json, ignore_label: bool = False) -> PageEvaluation:
    """
    Returns the PageEvaluation object for a given page's reference
    and system annotation json objects.
    """
    page_ref = PageReferenceSpans(ref_json)
    page_sys = PageSystemSpans(sys_json)
    return PageEvaluation(page_ref, page_sys, ignore_label=ignore_label)


@typing.no_type_check
def get_page_evals(
    ref_file: Path,
    sys_file: Path,
    ignore_label: bool = False,
    disable_progress: bool = False,
) -> Generator[PageEvaluation]:
    """
    Yields page-level evaluation objects for each page in reference annotation file.
    """
    # Read in system page jsons
    system_pages = {}
    for sys_page in orjsonl.stream(sys_file):
        page_id = sys_page["page_id"]
        system_pages[page_id] = sys_page

    # Then for each page in reference, get page-level evaluations
    n_lines = sum(1 for line in xopen(ref_file, mode="rb"))
    progress_pages = tqdm(
        orjsonl.stream(ref_file),
        total=n_lines,
        disable=disable_progress,
    )
    for ref_page in progress_pages:
        page_id = ref_page["page_id"]
        sys_page = system_pages[page_id]
        yield get_page_eval(ref_page, sys_page, ignore_label=ignore_label)


def write_page_evals(
    ref_file: Path,
    sys_file: Path,
    out_csv: Path,
    ignore_label: bool = False,
    partial_match_weight: float = 1,
    disable_progress: bool = False,
) -> None:
    """
    Writes the page-level span evaluations to a CSV file.
    """
    # For reporting average results
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    page_count = 0

    field_names = [
        "page_id",
        "precision",
        "recall",
        "n_span_matches",
        "n_span_misses",
        "n_span_spurious",
        "n_poem_matches",
        "n_poem_misses",
        "n_poem_spurious",
    ]
    with open(out_csv, mode="w", newline="") as file_handler:
        writer = csv.DictWriter(file_handler, fieldnames=field_names)
        writer.writeheader()
        for page_eval in get_page_evals(
            ref_file,
            sys_file,
            ignore_label=ignore_label,
            disable_progress=disable_progress,
        ):
            page_results = page_eval.evaluate(partial_match_weight=partial_match_weight)
            writer.writerow(page_results)
            # Update reporting variables
            cumulative_precision += page_results["precision"]
            cumulative_recall += page_results["recall"]
            page_count += 1

    avg_precision = cumulative_precision / page_count
    avg_recall = cumulative_recall / page_count
    print(
        f"Overall: {page_count} Pages | Precision = {avg_precision:.4g} | Recall = {avg_recall:.4g}"
    )


def main():
    """
    Calculates page-level span evaluations given some reference (i.e, adjudicated
    annotations) and system annotations (e.g. passim results) JSONL files. These
    results are written to a CSV file. Optionally, the evaluation can ignore span
    labels (i.e. poem ids) and downweight (i.e. penalize) partial span matches.
    """
    parser = argparse.ArgumentParser(
        description="Calculates page-level span evaluations"
    )
    parser.add_argument(
        "reference_jsonl",
        help="Path to reference poetry span annotations (JSONL file)",
        type=Path,
    )
    parser.add_argument(
        "system_jsonl",
        help="Path to system span annotations to be evaluated (JSONL file)",
        type=Path,
    )
    parser.add_argument(
        "output_file",
        help="Filename where results should be written (CSV file)",
        type=Path,
    )
    parser.add_argument(
        "--ignore-label",
        help="Ignore span labels for span evaluations",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--partial-match-weight",
        help="Downweight for partial matches for span evaluations (default: 1.0)",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--progress",
        help="Show progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()
    disable_progress = not args.progress

    # Check that input files exist
    if not args.reference_jsonl.is_file():
        print(
            f"Error: reference JSONL file {args.reference_jsonl} does not exist",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.system_jsonl.is_file():
        print(
            f"Error: system JSONL file {args.system_jsonl} does not exist",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check that output file does not exist
    if args.output_file.exists():
        print(
            f"Error: output file {args.output_file} already exists, not overwriting",
            file=sys.stderr,
        )
        sys.exit(1)

    write_page_evals(
        args.reference_jsonl,
        args.system_jsonl,
        args.output_file,
        ignore_label=args.ignore_label,
        partial_match_weight=args.partial_match_weight,
        disable_progress=disable_progress,
    )


if __name__ == "__main__":
    main()
