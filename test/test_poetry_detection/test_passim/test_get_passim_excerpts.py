from unittest.mock import patch

from corppa.poetry_detection.core import LabeledExcerpt
from corppa.poetry_detection.passim.get_passim_excerpts import (
    get_passim_excerpts,
    save_passim_excerpts,
)


@patch("corppa.poetry_detection.passim.get_passim_excerpts.orjsonl")
def test_get_passim_exceprts(mock_orjsonl):
    test_page_data = [
        # A page with one poem excerpt
        {
            "page_id": "page_a",
            "n_spans": 1,
            "poem_spans": [
                {
                    "ref_id": "poem_a",
                    "ref_corpus": "r1",
                    "ref_start": 10,
                    "ref_end": 11,
                    "ref_excerpt": "A",
                    "page_start": 0,
                    "page_end": 1,
                    "page_excerpt": "a",
                },
            ],
        },
        # A page with two poem excerpts
        {
            "page_id": "page_b",
            "n_spans": 2,
            "poem_spans": [
                {
                    "ref_id": "poem_z",
                    "ref_corpus": "r2",
                    "ref_start": 11,
                    "ref_end": 12,
                    "ref_excerpt": "B",
                    "page_start": 1,
                    "page_end": 2,
                    "page_excerpt": "b",
                },
                {
                    "ref_id": "poem_y",
                    "ref_corpus": "r2",
                    "ref_start": 5,
                    "ref_end": 7,
                    "ref_excerpt": "bb",
                    "page_start": 9,
                    "page_end": 11,
                    "page_excerpt": "BB",
                },
            ],
        },
        # A page with no identified poem excerpts
        {"page_id": "page_c", "n_spans": 0, "poem_spans": []},
    ]
    expected_results = [
        LabeledExcerpt(
            page_id="page_a",
            ppa_span_start=0,
            ppa_span_end=1,
            ppa_span_text="a",
            poem_id="poem_a",
            ref_corpus="r1",
            ref_span_start=10,
            ref_span_end=11,
            ref_span_text="A",
            detection_methods={"passim"},
            identification_methods={"passim"},
        ),
        LabeledExcerpt(
            page_id="page_b",
            ppa_span_start=1,
            ppa_span_end=2,
            ppa_span_text="b",
            poem_id="poem_z",
            ref_corpus="r2",
            ref_span_start=11,
            ref_span_end=12,
            ref_span_text="B",
            detection_methods={"passim"},
            identification_methods={"passim"},
        ),
        LabeledExcerpt(
            page_id="page_b",
            ppa_span_start=9,
            ppa_span_end=11,
            ppa_span_text="BB",
            poem_id="poem_y",
            ref_corpus="r2",
            ref_span_start=5,
            ref_span_end=7,
            ref_span_text="bb",
            detection_methods={"passim"},
            identification_methods={"passim"},
        ),
    ]

    mock_orjsonl.stream.return_value = iter(test_page_data)
    results = list(get_passim_excerpts("input"))
    mock_orjsonl.stream.assert_called_once_with("input")

    # Verify output
    assert results == expected_results


@patch("corppa.poetry_detection.passim.get_passim_excerpts.get_passim_excerpts")
def test_save_passim_excerpts(mock_get_excerpts, tmpdir):
    out_csv = tmpdir / "test.csv"

    test_excerpts = [
        LabeledExcerpt(
            page_id="a",
            ppa_span_start=0,
            ppa_span_end=3,
            ppa_span_text="abc",
            detection_methods={"passim"},
            poem_id="poem",
            ref_corpus="ref",
            ref_span_start=0,
            ref_span_end=3,
            ref_span_text="ABC",
            identification_methods={"passim"},
        ),
        LabeledExcerpt(
            page_id="b",
            ppa_span_start=0,
            ppa_span_end=3,
            ppa_span_text="abc",
            detection_methods={"passim"},
            poem_id="poem",
            ref_corpus="ref",
            ref_span_start=10,
            ref_span_end=13,
            ref_span_text="abc",
            identification_methods={"passim"},
        ),
    ]

    mock_get_excerpts.return_value = iter(test_excerpts)
    save_passim_excerpts("input", out_csv)
    mock_get_excerpts.assert_called_once_with("input")

    # Verify CSV output
    csv_fields = [
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
    excerpts_csv_form = [e.to_csv() for e in test_excerpts]
    csv_text = ",".join(csv_fields) + "\n"
    # Note: notes field is uninitialized
    csv_text += (
        ",".join([f"{excerpts_csv_form[0][f]}" for f in csv_fields[:-1]]) + ",\n"
    )
    csv_text += (
        ",".join([f"{excerpts_csv_form[1][f]}" for f in csv_fields[:-1]]) + ",\n"
    )
    assert out_csv.read_text(encoding="utf-8") == csv_text
