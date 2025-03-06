import sys
from inspect import isgenerator
from unittest.mock import MagicMock, call, patch

import pytest

from corppa.poetry_detection.annotation.process_adjudication_data import (
    get_excerpts,
    process_adjudication_data,
    process_page_annotation,
    simplify_excerpts,
)
from corppa.poetry_detection.core import Excerpt


def test_get_excerpts():
    page_annotation = {"id": "page_id", "text": "some page text"}

    # Missing spans field
    with pytest.raises(ValueError, match="Page annotation missing 'spans' field"):
        get_excerpts(page_annotation)

    # Empty spans field
    page_annotation["spans"] = []
    assert get_excerpts(page_annotation) == []

    # Missing text field
    blank_page = {"spans": []}
    assert get_excerpts(blank_page) == []

    # Typical case
    page_annotation["spans"].append({"start": 0, "end": 4})
    page_annotation["spans"].append({"start": 9, "end": 14})
    results = get_excerpts(page_annotation)
    ## No whitespace stripping
    assert results[0] == Excerpt(
        page_id="page_id",
        ppa_span_start=0,
        ppa_span_end=4,
        ppa_span_text="some",
        detection_methods={"adjudication"},
    )
    # Whitespace stripping occurs
    assert results[1] == Excerpt(
        page_id="page_id",
        ppa_span_start=10,
        ppa_span_end=14,
        ppa_span_text="text",
        detection_methods={"adjudication"},
    )


@patch("corppa.poetry_detection.annotation.process_adjudication_data.get_excerpts")
def test_process_page_annotation(mock_get_excerpts):
    mock_get_excerpts.return_value = ["some", "poetry", "excerpts"]
    page_annotation = {
        "id": "some-page-id",
        "work_id": "some-work-id",
        "meta": {"title": "some-title", "author": "some-author", "year": "some-year"},
        "spans": "some-spans",
    }
    result = process_page_annotation(page_annotation)
    assert result == {
        "page_id": "some-page-id",
        "work_id": "some-work-id",
        "work_title": "some-title",
        "work_author": "some-author",
        "work_year": "some-year",
        "excerpts": ["some", "poetry", "excerpts"],
        "n_excerpts": 3,
    }
    mock_get_excerpts.assert_called_once_with(page_annotation)


def test_simplify_excerpts():
    excerpts = [
        Excerpt(
            page_id="0",
            ppa_span_start=0,
            ppa_span_end=1,
            ppa_span_text="a",
            detection_methods={"adjudication"},
        ),
        Excerpt(
            page_id="1",
            ppa_span_start=1,
            ppa_span_end=2,
            ppa_span_text="b",
            detection_methods={"adjudication"},
        ),
    ]
    expected_results = [
        {"start": 0, "end": 1, "text": "a"},
        {"start": 1, "end": 2, "text": "b"},
    ]
    assert simplify_excerpts(excerpts) == expected_results


@patch("corppa.poetry_detection.annotation.process_adjudication_data.simplify_excerpts")
@patch(
    "corppa.poetry_detection.annotation.process_adjudication_data.process_page_annotation"
)
@patch("corppa.poetry_detection.annotation.process_adjudication_data.orjsonl")
@patch("corppa.poetry_detection.annotation.process_adjudication_data.tqdm")
def test_process_adjudication_data(
    mock_tqdm,
    mock_orjsonl,
    mock_process_page_annotation,
    mock_simplify_excerpts,
    tmpdir,
):
    input_jsonl = tmpdir / "input.jsonl"
    input_jsonl.write_text("some\ntext\n", encoding="utf-8")
    out_csv = tmpdir / "output.csv"

    excerpts = [
        Excerpt(
            page_id="a",
            ppa_span_start=0,
            ppa_span_end=1,
            ppa_span_text="a",
            detection_methods={"adjudication"},
        ),
        Excerpt(
            page_id="b",
            ppa_span_start=1,
            ppa_span_end=2,
            ppa_span_text="b",
            detection_methods={"adjudication"},
        ),
    ]

    # Default case
    ## Mock setup
    mock_orjsonl.stream.return_value = "jsonl stream"
    mock_tqdm.return_value = ["a", "b"]
    mock_process_page_annotation.side_effect = [
        {"page_id": "a", "excerpts": [excerpts[0]]},
        {"page_id": "b", "excerpts": [excerpts[1]]},
    ]
    mock_simplify_excerpts.side_effect = ["excerpts a", "excerpts b"]

    process_adjudication_data(input_jsonl, "out.jsonl", out_csv)
    ## Verify mock calls
    mock_orjsonl.stream.assert_called_once_with(input_jsonl)
    mock_tqdm.assert_called_once_with("jsonl stream", total=2, disable=False)
    assert mock_process_page_annotation.call_count == 2
    mock_process_page_annotation.assert_has_calls([call("a"), call("b")])
    assert mock_simplify_excerpts.call_count == 2
    mock_simplify_excerpts.assert_has_calls([call(excerpts[0:1]), call(excerpts[-1:])])
    assert mock_orjsonl.append.call_count == 2
    mock_orjsonl.append.assert_has_calls(
        [
            call("out.jsonl", {"page_id": "a", "excerpts": "excerpts a"}),
            call("out.jsonl", {"page_id": "b", "excerpts": "excerpts b"}),
        ]
    )

    ## Verify excerpt-level CSV output
    csv_fields = [
        "page_id",
        "excerpt_id",
        "ppa_span_start",
        "ppa_span_end",
        "ppa_span_text",
        "detection_methods",
        "notes",
    ]
    excerpts_csv_form = [e.to_csv() for e in excerpts]
    csv_text = ",".join(csv_fields) + "\n"
    # Note: notes field is uninitialized
    csv_text += (
        ",".join([f"{excerpts_csv_form[0][f]}" for f in csv_fields[:-1]]) + ",\n"
    )
    csv_text += (
        ",".join([f"{excerpts_csv_form[1][f]}" for f in csv_fields[:-1]]) + ",\n"
    )
    assert out_csv.read_text(encoding="utf-8") == csv_text

    # Disable progress
    ## Reset mocks
    mock_orjsonl.reset_mock()
    mock_orjsonl.stream.return_value = "jsonl stream"
    mock_tqdm.reset_mock()
    mock_tqdm.return_value = ["a", "b"]
    mock_process_page_annotation.side_effect = [
        {"page_id": "a", "excerpts": [excerpts[0]]},
        {"page_id": "b", "excerpts": [excerpts[1]]},
    ]
    mock_simplify_excerpts.side_effect = ["excerpts a", "excerpts b"]
    ## Limit testing to disabling progress
    process_adjudication_data(input_jsonl, "out.jsonl", out_csv, disable_progress=True)
    mock_orjsonl.stream.assert_called_once_with(input_jsonl)
    mock_tqdm.assert_called_once_with("jsonl stream", total=2, disable=True)
