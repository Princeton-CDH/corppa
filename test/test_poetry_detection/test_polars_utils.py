import csv
from unittest.mock import patch

import polars as pl
import pytest

from corppa.poetry_detection.core import MULTIVAL_DELIMITER, Excerpt, LabeledExcerpt
from corppa.poetry_detection.polars_utils import (
    POEM_FIELDS,
    PPA_FIELDS,
    add_ppa_works_meta,
    add_ref_poems_meta,
    extract_page_meta,
    fix_data_types,
    has_poem_ids,
    load_excerpts_df,
    load_meta_df,
)

excerpt1 = Excerpt(
    page_id="p.1",
    ppa_span_start=10,
    ppa_span_end=20,
    ppa_span_text="some text",
    detection_methods={"manual"},
)
excerpt2 = Excerpt(
    page_id="p.23",
    ppa_span_start=5,
    ppa_span_end=22,
    ppa_span_text="other text",
    detection_methods={"xml"},
)

excerpt1_label1 = LabeledExcerpt.from_excerpt(
    excerpt1,
    poem_id="poem-01",
    ref_corpus="test",
    ref_span_start=22,
    ref_span_end=35,
    ref_span_text="similar text",
    notes="extra info",
    identification_methods={"manual"},
)

excerpt1_label2 = LabeledExcerpt.from_excerpt(
    excerpt1,
    poem_id="poem-02",
    ref_corpus="test",
    ref_span_start=22,
    ref_span_end=35,
    ref_span_text="similar text",
    notes="id info",
    identification_methods={"refmatcha"},
)

excerpt2_label1 = LabeledExcerpt.from_excerpt(
    excerpt2,
    poem_id="poem-32",
    ref_corpus="test",
    ref_span_start=32,
    ref_span_end=54,
    ref_span_text="more text",
    identification_methods={"test"},
)


def _excerpts_to_csv(output_file, excerpts):
    # utility method to create a test CSV file with excerpt or labeled excerpt
    # data; takes a pathlib.Path and list of excerpt objects

    # convert to a list of csv-serializable dicts
    csv_data = [ex.to_csv() for ex in excerpts]
    with output_file.open("w", encoding="utf-8") as filehandle:
        # assuming for now that the first row has all the fields
        # (may not hold generally but ok here)
        csv_writer = csv.DictWriter(filehandle, fieldnames=csv_data[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(csv_data)


def test_has_poem_ids():
    # no poem_id column
    assert has_poem_ids(pl.DataFrame({"a": [1, 2, 3]})) is False
    # poem id is present but has no values
    assert has_poem_ids(pl.DataFrame({"a": [1, 2], "poem_id": [None, None]})) is False
    # poem id is present and has at least one non-null value
    assert has_poem_ids(pl.DataFrame({"a": [1, 2], "poem_id": ["Z12", None]})) is True


def test_fix_datatypes():
    df = pl.DataFrame(
        {
            "poem_id": ["a1", "a2", "a3"],
            "ppa_span_start": ["1", "2", "3"],
            "detection_methods": [
                "manual",
                MULTIVAL_DELIMITER.join(["manual", "passim"]),
                None,
            ],
            "ignore_me": ["foo", "bar", "baz"],
        }
    )
    fixed_df = fix_data_types(df)
    # all columns should match (nothing removed or added)
    assert df.columns == fixed_df.columns
    # column datatypes should differ
    assert df.schema != fixed_df.schema
    assert fixed_df.schema["ppa_span_start"] == pl.Int64
    assert fixed_df.schema["detection_methods"] == pl.List
    # multival fields should be split into lists; empty value should be null
    detect_methods_parsed = fixed_df["detection_methods"].to_list()
    assert detect_methods_parsed == [["manual"], ["manual", "passim"], None]


def test_fix_datatypes_null_to_list():
    df = pl.DataFrame(
        {
            "poem_id": ["a", "b"],
            "ppa_span_start": ["1", "2"],
            "identification_methods": [None, None],
        }
    )
    # if no values are set in a list field, type is pl.Null
    fixed_df = fix_data_types(df)
    # type should be set to list of string
    assert fixed_df.schema["identification_methods"] == pl.List(pl.String)


@patch("corppa.poetry_detection.polars_utils.add_ref_poems_meta")
@patch("corppa.poetry_detection.polars_utils.add_ppa_works_meta")
def test_load_excerpts_df(mock_add_ppa_meta, mock_add_poem_meta, tmp_path):
    ## Base Case ##
    datafile = tmp_path / "excerpts.csv"
    # valid excerpt data
    _excerpts_to_csv(datafile, [excerpt1])
    loaded_df = load_excerpts_df(datafile)
    assert isinstance(loaded_df, pl.DataFrame)
    assert len(loaded_df) == 1
    assert loaded_df.row(0, named=True) == excerpt1.to_dict()
    # set field has been loaded correctly as a list
    assert loaded_df.schema["detection_methods"] == pl.List
    # valid labeled excerpt data
    _excerpts_to_csv(datafile, [excerpt1_label1, excerpt2_label1])
    loaded_df = load_excerpts_df(datafile)
    assert isinstance(loaded_df, pl.DataFrame)
    assert len(loaded_df) == 2
    assert loaded_df.row(0, named=True) == excerpt1_label1.to_dict()
    # set field has been loaded correctly as a list
    assert loaded_df.schema["identification_methods"] == pl.List
    # check that optional methods are not run
    mock_add_ppa_meta.assert_not_called()
    mock_add_poem_meta.assert_not_called()

    # TODO: Check actual calls?
    ## With PPA meta ##
    _ = load_excerpts_df(datafile, ppa_works_meta="ppa_meta")
    mock_add_ppa_meta.assert_called_once()
    mock_add_poem_meta.assert_not_called()

    ## With poem meta ##
    mock_add_ppa_meta.reset_mock()
    _ = load_excerpts_df(datafile, ref_poems_meta="poem_meta")
    mock_add_ppa_meta.assert_not_called()
    mock_add_poem_meta.assert_called_once()

    # With both PPA and poem meta ##
    mock_add_poem_meta.reset_mock()
    _ = load_excerpts_df(
        datafile, ppa_works_meta="ppa_meta", ref_poems_meta="poem_meta"
    )
    mock_add_ppa_meta.assert_called_once()
    mock_add_poem_meta.assert_called_once()

    # invalid - non excerpt data
    with datafile.open("w", encoding="utf-8") as filehandle:
        csv_writer = csv.writer(filehandle)
        csv_writer.writerow(["id", "note"])
        csv_writer.writerow(["p.01", "missing"])

    with pytest.raises(ValueError, match="missing required excerpt fields"):
        load_excerpts_df(datafile)

    # invalid - looks like labeled excerpt data but missing a field
    with datafile.open("w", encoding="utf-8") as filehandle:
        csv_writer = csv.writer(filehandle)
        csv_writer.writerow(["poem_id", "title"])
        csv_writer.writerow(["Z01", "the missing"])

    with pytest.raises(ValueError, match="missing required labeled excerpt fields"):
        load_excerpts_df(datafile)


def test_extract_page_meta():
    ppa_page_ids = ["A01224.100", "yale.39002088447587.00000050", "CW0111540239.0092"]
    excerpts_df = pl.DataFrame(
        [
            {"page_id": page_id, "excerpt_id": f"excerpt_{i}"}
            for i, page_id in enumerate(ppa_page_ids)
        ]
    )
    results = extract_page_meta(excerpts_df)
    # Check column names
    assert set(results.columns) == set(excerpts_df.columns) | {
        "ppa_work_id",
        "page_num",
    }
    # Check row contents
    for i, page_id in enumerate(ppa_page_ids):
        work_id, page_num = page_id.rsplit(".", maxsplit=1)
        expected_row = excerpts_df.row(i, named=True) | {
            "ppa_work_id": work_id,
            "page_num": int(page_num),
        }
        assert results.row(i, named=True) == expected_row


def test_load_meta_df(tmp_path):
    # Prepare metadata file
    ppa_meta = tmp_path / "ppa_meta.csv"
    csv_fields = [
        "work_id",
        "source_id",
        "cluster_id",
        "title",
        "author",
        "pub_year",
        "collections",
        "work_type",
        "source",
    ]
    rows = [
        {
            "work_id": "work_a",
            "source_id": "work_a",
            "cluster_id": "cluster_a",
            "title": "title_a",
            "author": "author_a",
            "pub_year": "1899",
            "collections": "['Linguistic', 'Literary']",
            "work_type": "full-work",
            "source": "source_a",
        },
        {
            "work_id": "work_b-p7",
            "source_id": "work_b",
            "cluster_id": "cluster_b",
            "title": "title_b",
            "author": "author_b",
            "pub_year": "1507",
            "collections": "['Uncategorized']",
            "work_type": "excerpt",
            "source": "source_b",
        },
    ]
    with ppa_meta.open("w", encoding="utf-8") as file:
        csv_writer = csv.DictWriter(file, fieldnames=csv_fields)
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    # Typical Case:
    result_df = load_meta_df(ppa_meta, PPA_FIELDS)
    # Check column names
    assert result_df.columns == list(PPA_FIELDS.values())
    # Check row contents
    for i, row in enumerate(rows):
        row_dict = {v: row[k] for k, v in PPA_FIELDS.items()}
        assert result_df.row(i, named=True) == row_dict

    # Error Case: Input file does not exist
    missing_csv = tmp_path / "missing.csv"
    with pytest.raises(ValueError, match=f"Input file {missing_csv} does not exist"):
        load_meta_df(missing_csv, PPA_FIELDS)

    # Error Case: Input file is missing required field
    for missing_fields in [["author"], ["pub_year", "source"]]:
        bad_csv = tmp_path / "missing_fields.csv"
        with bad_csv.open("w", encoding="utf-8") as file:
            bad_fields = list(set(csv_fields) - set(missing_fields))
            csv_writer = csv.DictWriter(
                file, fieldnames=bad_fields, extrasaction="ignore"
            )
            csv_writer.writeheader()
            csv_writer.writerows(rows)

        missing_str = ", ".join(missing_fields)
        err_msg = f"Input CSV is missing the following required fields: {missing_str}"
        with pytest.raises(ValueError, match=err_msg):
            load_meta_df(bad_csv, PPA_FIELDS)


@patch("corppa.poetry_detection.polars_utils.load_meta_df")
def test_add_ppa_works_meta(mock_load_meta_df):
    # Construct test inputs
    excerpts_df = pl.DataFrame(
        [
            {
                "page_id": "page_a",
                "excerpt_id": "ex_1",
                "ppa_work_id": "work_a",
            },
            {
                "page_id": "page_b",
                "excerpt_id": "ex_1",
                "ppa_work_id": "work_b",
            },
            {
                "page_id": "page_a",
                "excerpt_id": "ex_2",
                "ppa_work_id": "work_a",
            },
        ]
    )
    ppa_meta_rows = []
    for i in ["a", "b", "c"]:
        ppa_meta_rows.append(
            {
                "ppa_work_id": f"work_{i}",
                "ppa_author": f"author_{i}",
                "ppa_title": f"title_{i}",
            }
        )
    ppa_meta_df = pl.DataFrame(ppa_meta_rows)
    # Set-up mock object
    mock_load_meta_df.return_value = ppa_meta_df

    results = add_ppa_works_meta(excerpts_df, "ppa_meta")
    mock_load_meta_df.assert_called_once_with("ppa_meta", PPA_FIELDS)
    # Check columns
    assert set(results.columns) == set(excerpts_df.columns) | set(ppa_meta_df.columns)
    # Check row contents
    assert results.height == 3
    assert results.row(0, named=True) == (
        excerpts_df.row(0, named=True) | ppa_meta_rows[0]
    )
    assert results.row(1, named=True) == (
        excerpts_df.row(1, named=True) | ppa_meta_rows[1]
    )
    assert results.row(2, named=True) == (
        excerpts_df.row(2, named=True) | ppa_meta_rows[0]
    )

    # Error case: missing `ppa_work_id` field
    mock_load_meta_df.reset_mock()
    err_msg = "Missing ppa_work_id field; use extract_page_meta to extract it."
    with pytest.raises(ValueError, match=err_msg):
        bad_df = pl.DataFrame([{"excerpt_id": "a"}, {"excerpt_id": "b"}])
        add_ppa_works_meta(bad_df, "ppa_meta")
    mock_load_meta_df.assert_not_called()


@patch("corppa.poetry_detection.polars_utils.load_meta_df")
def test_add_poems_meta(mock_load_meta_df):
    # Construct test inputs
    excerpts_df = pl.DataFrame(
        [
            {
                "page_id": "page_a",
                "excerpt_id": "ex_1",
                "ref_corpus": "A",
                "poem_id": "a",
            },
            {
                "page_id": "page_b",
                "excerpt_id": "ex_1",
                "ref_corpus": "B",
                "poem_id": "b",
            },
            {
                "page_id": "page_a",
                "excerpt_id": "ex_2",
                "ref_corpus": "A",
                "poem_id": "b",
            },
        ]
    )
    poem_meta_rows = [
        {
            "poem_id": "a",
            "ref_corpus": "A",
            "author": "author_1",
            "title": "title_1",
        },
        {
            "poem_id": "b",
            "ref_corpus": "A",
            "author": "author_2",
            "title": "title_2",
        },
        {
            "poem_id": "b",
            "ref_corpus": "B",
            "author": "author_3",
            "title": "title_3",
        },
        {
            "poem_id": "a",
            "ref_corpus": "C",
            "author": "author_4",
            "title": "title_4",
        },
    ]

    poem_meta_df = pl.DataFrame(poem_meta_rows)
    # Set-up mock object
    mock_load_meta_df.return_value = poem_meta_df

    results = add_ref_poems_meta(excerpts_df, "poem_meta")
    mock_load_meta_df.assert_called_once_with("poem_meta", POEM_FIELDS)
    # Check columns
    assert set(results.columns) == set(excerpts_df.columns) | set(poem_meta_df.columns)
    # Check row contents
    assert results.height == 3
    assert results.row(0, named=True) == (
        excerpts_df.row(0, named=True) | poem_meta_rows[0]
    )
    assert results.row(1, named=True) == (
        excerpts_df.row(1, named=True) | poem_meta_rows[2]
    )
    assert results.row(2, named=True) == (
        excerpts_df.row(2, named=True) | poem_meta_rows[1]
    )

    # Error case: missing required fields
    for missing_fields in [["poem_id"], ["ref_corpus"], ["poem_id", "ref_corpus"]]:
        mock_load_meta_df.reset_mock()
        # Construct bad input dataframe
        ## Add fields that aren't missing
        bad_rows = [{"excerpt_id": "a"}, {"excerpt_id": "b"}]
        for field in {"poem_id", "ref_corpus"} - set(missing_fields):
            for row in bad_rows:
                row[field] = "value"
        bad_df = pl.DataFrame(bad_rows)
        # Test error case
        err_msg = f"Input DataFrame missing the following required fields: "
        err_msg += ", ".join(missing_fields)
        with pytest.raises(ValueError, match=err_msg):
            add_ref_poems_meta(bad_df, "poem_meta")
        mock_load_meta_df.assert_not_called()
