# Copyright (c) 2026, Center for Digital Humanities, Princeton University
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import polars as pl
import pytest

from corppa.poetry_detection.compile_dataset import (
    get_excerpt_sources,
    load_compiled_excerpts,
    save_ppa_metadata,
)


def test_get_excerpt_sources_empty_dir(tmp_path):
    result = get_excerpt_sources(tmp_path)
    assert result == []


def test_get_excerpt_sources_with_files(tmp_path):
    subdir1 = tmp_path / "subdir1"
    subdir2 = tmp_path / "subdir2"
    subdir1.mkdir()
    subdir2.mkdir()

    (tmp_path / "file1.csv").touch()
    (tmp_path / "file2.csv.gz").touch()
    (subdir1 / "nested.csv").touch()
    (subdir2 / "nested.csv.gz").touch()
    (tmp_path / "file3.txt").touch()

    result = get_excerpt_sources(tmp_path)
    assert len(result) == 4


def test_save_ppa_metadata(tmp_path):
    input_file = tmp_path / "ppa_works.csv"
    output_file = tmp_path / "output.csv"

    input_file.write_text("work_id,title,author\nW001,Test Work,Test Author\n")

    excerpts_df = pl.DataFrame(
        {
            "ppa_work_id": ["W001", "W001", "W001"],
            "excerpt_id": ["e1", "e2", "e3"],
            "poem_id": ["poem-1", "poem-1", "poem-2"],
            "poem_author": ["Author A", "Author A", "Author B"],
        }
    )

    save_ppa_metadata(input_file, output_file, excerpts_df)

    result = pl.read_csv(output_file)
    assert "num_excerpts" in result.columns
    assert "num_poems" in result.columns
    assert "num_poets" in result.columns

    row = result.row(0, named=True)
    assert row["work_id"] == "W001"
    assert row["num_excerpts"] == 3
    assert row["num_poems"] == 2
    assert row["num_poets"] == 2


def test_save_ppa_metadata_not_csv(tmp_path):
    input_file = tmp_path / "ppa_works.json"
    output_file = tmp_path / "output.csv"

    excerpts_df = pl.DataFrame(
        {
            "ppa_work_id": ["W001"],
            "excerpt_id": ["e1"],
            "poem_id": ["poem-1"],
            "poem_author": ["Author A"],
        }
    )

    with pytest.raises(ValueError, match="PPA metadata must be loaded as CSV"):
        save_ppa_metadata(input_file, output_file, excerpts_df)


@patch("corppa.poetry_detection.compile_dataset.pl.read_csv")
@patch("corppa.poetry_detection.compile_dataset.extract_page_meta")
def test_load_compiled_excerpts_uncompressed(
    mock_extract_page_meta, mock_read_csv, tmp_path
):
    # config method populates both paths;
    # load method will choose the first one that exists
    excerpt_file = tmp_path / "excerpts.csv"
    excerpt_file.touch()
    excerpt_gz_file = tmp_path / "excerpts.csv.gz"
    config = {
        "compiled_excerpt_file": excerpt_file,
        "compressed_excerpt_file": excerpt_gz_file,
    }

    result = load_compiled_excerpts(config)
    assert result == mock_extract_page_meta.return_value

    mock_extract_page_meta.assert_called_once_with(mock_read_csv.return_value)
    mock_read_csv.assert_called_once_with(excerpt_file)

    # reset and remove the uncompressed, make the gz exist
    mock_read_csv.reset_mock()
    excerpt_file.unlink()
    excerpt_gz_file.touch()
    load_compiled_excerpts(config)
    mock_read_csv.assert_called_once_with(excerpt_gz_file)


def test_load_compiled_excerpts_file_not_found(tmp_path):
    config = {
        "compiled_excerpt_file": tmp_path / "nonexistent.csv",
        "compressed_excerpt_file": tmp_path / "nonexistent.csv.gz",
    }

    with pytest.raises(ValueError, match="Excerpt data file not found"):
        load_compiled_excerpts(config)
