import csv

import pytest

from corppa.poetry_detection.ppa_works import (
    PPA_FIELDS,
    add_ppa_work_meta,
    load_ppa_works_df,
)


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
    # Check column names
    assert result_df.columns == list(PPA_FIELDS.values())
    for i, row in enumerate(rows):
        row_dict = {v: row[k] for k, v in PPA_FIELDS.items()}
        assert result_df.row(i, named=True) == row_dict

    # Error Case: Input file does not exist
    missing_csv = tmp_path / "missing.csv"
    with pytest.raises(ValueError, match=f"Input file {missing_csv} does not exist"):
        load_ppa_works_df(missing_csv)

    # Error Case: Input file is missing field
    ## Single field
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
            load_ppa_works_df(bad_csv)
