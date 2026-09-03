# Copyright (c) 2026, Center for Digital Humanities, Princeton University
# SPDX-License-Identifier: Apache-2.0

"""Identify and remove running headers from page-level text.

Running headers are particularly troublesome when comparing volumes: OCR sees
them as page content, and a header that occurs on every page can dominate
text-reuse results.  This module deliberately only considers the first two
substantial lines of a page.  That keeps repeated phrases in the body text
from being mistaken for headers.

The detection approach is adapted from the running-header cleanup code
originally contributed by Wouter Haverals in ``ppa-nlp-archive``, which in
turn was inspired by Ted Underwood's ``HeaderFinder``.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any

import polars as pl


def _substantial_lines(text: str, limit: int = 2) -> list[tuple[int, str]]:
    """Return the first ``limit`` non-empty, non-numeric lines."""
    lines = text.splitlines()
    result = []
    # Keep the original line number so removal can preserve all other lines
    # and their line endings; a regex would obscure that two-step operation.
    for number, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) < 5 or stripped.isdigit():
            continue
        result.append((number, line))
        if len(result) == limit:
            break
    return result


def _comparison_text(line: str) -> str:
    """Normalize a line for comparison, ignoring page numbers and punctuation."""
    return re.sub(r"[^a-zA-Z]+", "", line).casefold()


def _text_value(value: Any) -> str:
    """Convert a nullable dataframe/page value to text for processing."""
    return "" if value is None else str(value)


def identify_headers(
    pages: Sequence[Mapping[str, Any]],
    *,
    text_key: str = "page_text",
    window: int = 2,
    similarity_threshold: float = 0.8,
) -> dict[int, set[str]]:
    """Identify likely running headers in a sequence of page dictionaries.

    Pages are compared only to the preceding and following ``window`` pages.
    The return value maps each page index to the exact line(s) identified on
    that page, making it possible to remove a header without removing the
    same phrase from another page's body text.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1")

    candidates = []
    for page in pages:
        lines = _substantial_lines(_text_value(page.get(text_key, "")), limit=3)
        # On short pages, do not classify the only body line as a header just
        # because two pages happen to contain the same short text.
        candidates.append(lines[:2] if len(lines) == 3 else lines[:1])
    headers: dict[int, set[str]] = {}
    for index, lines in enumerate(candidates):
        start = max(0, index - window)
        stop = min(len(candidates), index + window + 1)
        for _, line in lines:
            normalized = _comparison_text(line)
            if not normalized:
                continue
            for other_index in range(start, stop):
                if other_index == index:
                    continue
                if any(
                    SequenceMatcher(
                        None, normalized, _comparison_text(other_line)
                    ).ratio()
                    >= similarity_threshold
                    for _, other_line in candidates[other_index]
                ):
                    headers.setdefault(index, set()).add(line)
                    break
    return headers


def remove_headers(
    pages: Sequence[Mapping[str, Any]],
    headers: dict[int, set[str]] | None = None,
    *,
    text_key: str = "page_text",
    **identify_kwargs: Any,
) -> list[dict[str, Any]]:
    """Return copies of ``pages`` with identified leading headers removed."""
    if headers is None:
        headers = identify_headers(pages, text_key=text_key, **identify_kwargs)

    cleaned = []
    for index, page in enumerate(pages):
        result = dict(page)
        remove = headers.get(index, set())
        if remove:
            lines = _text_value(result.get(text_key, "")).splitlines(keepends=True)
            substantial_seen = 0
            output = []
            for line in lines:
                if len(line.strip()) >= 5 and not line.strip().isdigit():
                    substantial_seen += 1
                if substantial_seen <= 2 and line.rstrip("\r\n") in remove:
                    continue
                output.append(line)
            result[text_key] = "".join(output)
        cleaned.append(result)
    return cleaned


def cleanup_pages(
    pages: Iterable[Mapping[str, Any]],
    *,
    text_key: str = "text",
    **identify_kwargs: Any,
) -> list[dict[str, Any]]:
    """Identify and remove running headers in an iterable of page dictionaries."""
    page_list = list(pages)
    return remove_headers(page_list, text_key=text_key, **identify_kwargs)


def cleanup_dataframe(
    dataframe: pl.DataFrame,
    *,
    text_column: str = "text",
    group_column: str | None = "work_id",
    **identify_kwargs: Any,
) -> Any:
    """Return a Polars DataFrame with running headers removed.

    Header detection is performed independently for each work, so the last
    page of one work cannot cause a false match on the first page of the next.
    The input row order and all columns are preserved.
    """
    if text_column not in dataframe.columns:
        raise ValueError(f"DataFrame has no {text_column!r} column")

    if group_column is None:
        groups = [(None, list(range(dataframe.height)))]
    else:
        if group_column not in dataframe.columns:
            raise ValueError(f"DataFrame has no {group_column!r} column")
        groups = []
        rows_with_index = dataframe.with_row_index("_row")
        for value in dataframe.get_column(group_column).unique(maintain_order=True):
            groups.append(
                (
                    value,
                    rows_with_index.filter(pl.col(group_column) == value)["_row"].to_list(),
                )
            )

    texts = dataframe.get_column(text_column).to_list()
    for _, row_indices in groups:
        pages = [{text_column: texts[index]} for index in row_indices]
        cleaned = cleanup_pages(pages, text_key=text_column, **identify_kwargs)
        for index, page in zip(row_indices, cleaned):
            texts[index] = page[text_column]
    return dataframe.with_columns(pl.Series(text_column, texts))
