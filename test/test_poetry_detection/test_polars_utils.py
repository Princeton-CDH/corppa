import csv

import polars as pl
import pytest

from corppa.poetry_detection.core import MULTIVAL_DELIMITER, Excerpt, LabeledExcerpt
from corppa.poetry_detection.polars_utils import (
    fix_data_types,
    has_poem_ids,
    load_excerpts_df,
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
            "poem_id": ["a1", "a2", "a3"],
            "ppa_span_start": ["1", "2", "3"],
            "detection_methods": [None, None, None],
        }
    )
    # if no values are set in a list field, type is pl.Null
    fixed_df = fix_data_types(df)
    # type should be set to list of string
    assert fixed_df.schema["detection_methods"] == pl.List(pl.String)


def test_load_excerpts_df(tmp_path):
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
