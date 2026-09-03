# Copyright (c) 2026, Center for Digital Humanities, Princeton University
# SPDX-License-Identifier: Apache-2.0

import polars as pl

from corppa.utils.running_headers import (
    cleanup_dataframe,
    cleanup_pages,
    identify_headers,
    remove_headers,
)


def test_identify_headers_compares_nearby_pages():
    pages = [
        {"page_text": "THE POEMS OF BLAIR\n1\nA first page."},
        {"page_text": "THE POEMS OF BLAIR\n2\nA second page."},
        {"page_text": "THE POEMS OF BLAIR\n3\nA third page."},
    ]

    assert identify_headers(pages) == {
        0: {"THE POEMS OF BLAIR"},
        1: {"THE POEMS OF BLAIR"},
        2: {"THE POEMS OF BLAIR"},
    }


def test_remove_headers_preserves_metadata_and_body():
    pages = [
        {
            "work_id": "work",
            "page_text": "THE POEMS OF BLAIR\n1\nA repeated phrase in the body.",
        },
        {"page_text": "THE POEMS OF BLAIR\n2\nMore text."},
    ]

    result = remove_headers(pages)

    assert result[0]["work_id"] == "work"
    assert result[0]["page_text"] == "1\nA repeated phrase in the body."
    assert result[1]["page_text"] == "2\nMore text."


def test_cleanup_pages_accepts_iterators():
    pages = ({"text": f"THE HEADER\n{n}\nText."} for n in range(2))
    assert [page["text"] for page in cleanup_pages(pages)] == ["0\nText.", "1\nText."]


def test_cleanup_dataframe_groups_by_work_and_preserves_rows():
    dataframe = pl.DataFrame(
        {
            "work_id": ["a", "b", "a", "b"],
            "text": [
                "HEADER A\n1\nA text.",
                "HEADER B\n1\nB text.",
                "HEADER A\n2\nA more text.",
                "HEADER B\n2\nB more text.",
            ],
        }
    )

    result = cleanup_dataframe(dataframe)

    assert result["text"].to_list() == [
        "1\nA text.",
        "1\nB text.",
        "2\nA more text.",
        "2\nB more text.",
    ]
