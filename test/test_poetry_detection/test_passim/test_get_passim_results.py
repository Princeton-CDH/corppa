from unittest.mock import call, patch

from corppa.poetry_detection.core import LabeledExcerpt
from corppa.poetry_detection.passim.get_passim_results import (
    build_passim_excerpt,
    get_page_texts,
    get_passim_span,
)


@patch("corppa.poetry_detection.passim.get_passim_results.orjsonl")
def test_get_page_texts(mock_orjsonl):
    # Setup corpus data
    jsonl_data = [
        {"id": "a", "text": "1"},
        {"id": "b", "text": "2"},
        {"id": "c", "text": "3"},
    ]
    mock_orjsonl.stream.return_value = jsonl_data

    expected_results = {"a": "1", "c": "3"}
    assert get_page_texts({"a", "c", "z"}, "input jsonl") == expected_results
    mock_orjsonl.stream.assert_called_once_with("input jsonl")


@patch.object(LabeledExcerpt, "correct_page_excerpt")
def test_build_passim_excerpt(mock_correct_excerpt):
    span_record = {
        "ref_id": "poem_a",
        "ref_corpus": "ref",
        "ref_start": 11,
        "ref_end": 12,
        "ref_excerpt": "B",
        "page_start": 1,
        "page_end": 2,
        "ppa_excerpt": "b",
        "matches": 1,
    }
    # Basic case (without correction)
    expected_result = LabeledExcerpt(
        page_id="page_id",
        ppa_span_start=1,
        ppa_span_end=2,
        ppa_span_text="b",
        poem_id="poem_a",
        ref_corpus="ref",
        ref_span_start=11,
        ref_span_end=12,
        ref_span_text="B",
        detection_methods={"passim"},
        identification_methods={"passim"},
        notes="passim: 1 char matches",
    )
    assert build_passim_excerpt("page_id", span_record) == expected_result
    mock_correct_excerpt.assert_not_called()
    # With correction
    mock_correct_excerpt.side_effect = "a"
    assert build_passim_excerpt("page_id", span_record, ppa_page_text="ppa_page") == "a"
    mock_correct_excerpt.assert_called_once_with("ppa_page")


def test_get_passim_span():
    passim_record = {
        "id": "poem_id",
        "corpus": "ref",
        "begin": "ref_start_idx",
        "end": "ref_end_idx",
        "id2": "ppa_page_id",
        "begin2": "page_start_idx",
        "end2": "page_end_idx",
        "matches": "# char matches",
        "s1": "Aligned text\nwith dashes",
        "s2": "Aligned text with\t------",
    }
    assert get_passim_span(passim_record) == {
        "page_id": "ppa_page_id",
        "page_start": "page_start_idx",
        "page_end": "page_end_idx",
        "ref_id": "poem_id",
        "ref_corpus": "ref",
        "ref_start": "ref_start_idx",
        "ref_end": "ref_end_idx",
        "matches": "# char matches",
        "aligned_ref_excerpt": "Aligned text with dashes",
        "aligned_ppa_excerpt": "Aligned text with ------",
    }
