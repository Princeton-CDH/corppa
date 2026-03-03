import csv
from unittest.mock import patch

import polars as pl
import pytest

from corppa.poetry_detection.ppa_works import (
    add_ppa_works_meta,
    extract_page_meta,
    load_ppa_works_df,
)


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


def test_load_ppa_works_df(tmp_path):
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
            "pub_year": 1899,
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
            "pub_year": 1507,
            "collections": "['Uncategorized']",
            "work_type": "excerpt",
            "source": "source_b",
        },
    ]
    with ppa_meta.open("w", encoding="utf-8") as file:
        csv_writer = csv.DictWriter(file, fieldnames=csv_fields)
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    result_df = load_ppa_works_df(ppa_meta)
    # Check column names are prefixed
    prefixed_fields = [f"ppa_{field}" for field in csv_fields]
    assert result_df.columns == prefixed_fields
    # Check row contents
    for i, row in enumerate(rows):
        # the input dict with prefixed field name should
        # be equivalent to the dataframe row
        row_dict = {f"ppa_{k}": v for k, v in row.items()}
        assert result_df.row(i, named=True) == row_dict

    # Error Case: Input file does not exist
    missing_csv = tmp_path / "missing.csv"
    with pytest.raises(ValueError, match=f"Input file {missing_csv} does not exist"):
        load_ppa_works_df(missing_csv)

    # Error Case: Input file is missing required work_id field
    bad_csv = tmp_path / "missing_fields.csv"
    with bad_csv.open("w", encoding="utf-8") as file:
        # subset of fields; does not include work_id
        bad_fields = ["author", "title"]
        csv_writer = csv.DictWriter(file, fieldnames=bad_fields, extrasaction="ignore")
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    # close the file to write content to disk before loading
    err_msg = "Input CSV is missing required `work_id` field"
    with pytest.raises(ValueError, match=err_msg):
        load_ppa_works_df(bad_csv)


@patch("corppa.poetry_detection.ppa_works.extract_page_meta")
@patch("corppa.poetry_detection.ppa_works.load_ppa_works_df")
def test_add_ppa_works_meta(mock_load_ppa_df, mock_extract_page_meta):
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
    mock_load_ppa_df.return_value = ppa_meta_df

    results = add_ppa_works_meta(excerpts_df, "ppa_meta")
    mock_load_ppa_df.assert_called_once_with("ppa_meta")
    # Check columns
    assert set(results.columns) == set(excerpts_df.columns) | set(ppa_meta_df.columns)
    # Check row contents
    assert results.row(0, named=True) == (
        excerpts_df.row(0, named=True) | ppa_meta_rows[0]
    )
    assert results.row(1, named=True) == (
        excerpts_df.row(1, named=True) | ppa_meta_rows[1]
    )
    assert results.row(2, named=True) == (
        excerpts_df.row(2, named=True) | ppa_meta_rows[0]
    )
    mock_extract_page_meta.assert_not_called()

    # If missing `ppa_work_id` field, automatically calls
    # extract_page_meta before joining
    mock_load_ppa_df.reset_mock()
    unextracted_page_id_df = pl.DataFrame([{"excerpt_id": "a"}, {"excerpt_id": "b"}])
    add_ppa_works_meta(unextracted_page_id_df, "ppa_meta")
    mock_extract_page_meta.assert_called_once()
    mock_load_ppa_df.assert_called()
