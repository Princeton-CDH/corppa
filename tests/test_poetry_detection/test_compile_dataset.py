# Copyright (c) 2026, Center for Digital Humanities, Princeton University
# SPDX-License-Identifier: Apache-2.0


import gzip
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from corppa.poetry_detection.compile_dataset import (
    compress_file,
    get_excerpt_sources,
    load_compiled_excerpts,
    main,
    run_merge_step,
    run_poem_metadata_step,
    run_ppa_metadata_step,
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


@pytest.mark.parametrize(
    "args,expected_calls",
    [
        ([], {"merge", "poem_metadata", "ppa_metadata"}),
        (["--merge"], {"merge"}),
        (["--poem_metadata"], {"poem_metadata"}),
        (["--ppa_metadata"], {"ppa_metadata"}),
    ],
)
@patch("corppa.poetry_detection.compile_dataset.run_ppa_metadata_step")
@patch("corppa.poetry_detection.compile_dataset.run_poem_metadata_step")
@patch("corppa.poetry_detection.compile_dataset.run_merge_step")
@patch("corppa.poetry_detection.compile_dataset.load_compilation_config")
def test_main(
    mock_load_config,
    mock_merge,
    mock_poem,
    mock_ppa,
    args,
    expected_calls,
    tmp_path,
):
    mock_load_config.return_value = {
        "test": "config",
        "output_data_dir": tmp_path,
    }

    main(args)

    if "merge" in expected_calls:
        mock_merge.assert_called_once()
    else:
        mock_merge.assert_not_called()

    if "poem_metadata" in expected_calls:
        mock_poem.assert_called_once()
    else:
        mock_poem.assert_not_called()

    if "ppa_metadata" in expected_calls:
        mock_ppa.assert_called_once()
    else:
        mock_ppa.assert_not_called()


@patch("corppa.poetry_detection.compile_dataset.compress_file")
@patch("corppa.poetry_detection.compile_dataset.merge_excerpt_files")
@patch("corppa.poetry_detection.compile_dataset.get_excerpt_sources")
def test_run_merge_step(mock_get_sources, mock_merge, mock_compress, tmp_path):
    compile_opts = {
        "source_excerpt_data": tmp_path / "/data/excerpts",
        "compiled_excerpt_file": tmp_path / "/out/excerpts.csv",
        "compressed_excerpt_file": tmp_path / "/out/excerpts.csv.gz",
    }

    result = run_merge_step(compile_opts, None, compress_excerpts=True)
    # returns result of merge
    assert result == mock_merge.return_value

    # get sources is called on the configured path
    mock_get_sources.assert_called_once_with(compile_opts["source_excerpt_data"])
    # merge is called with the result of get sources and compile option
    mock_merge.assert_called_once_with(
        mock_get_sources.return_value, compile_opts["compiled_excerpt_file"]
    )
    # compress is called
    mock_compress.assert_called_once_with(
        Path("/out/excerpts.csv"), Path("/out/excerpts.csv.gz")
    )

    # call again with no compression
    mock_compress.reset_mock()
    run_merge_step(compile_opts, None, compress_excerpts=False)
    mock_compress.assert_not_called()


@patch("corppa.poetry_detection.compile_dataset.save_poem_metadata")
@patch("corppa.poetry_detection.compile_dataset.extract_page_meta")
@patch("corppa.poetry_detection.compile_dataset.load_compiled_excerpts")
def test_run_poem_metadata_step_with_df(mock_load, mock_extract, mock_save, tmp_path):
    input_df = pl.DataFrame({"id": [1]})
    mock_extract.return_value = pl.DataFrame({"id": [1], "page_id": ["p.1"]})

    compile_opts = {"poem_metadata_file": tmp_path / "/out/poem_meta.csv"}

    run_poem_metadata_step(compile_opts, input_df)

    mock_load.assert_not_called()
    mock_extract.assert_called_once_with(input_df)
    mock_save.assert_called_once_with(
        compile_opts["poem_metadata_file"], mock_extract.return_value
    )


@patch("corppa.poetry_detection.compile_dataset.save_poem_metadata")
@patch("corppa.poetry_detection.compile_dataset.extract_page_meta")
@patch("corppa.poetry_detection.compile_dataset.load_compiled_excerpts")
def test_run_poem_metadata_step(mock_load, mock_extract, mock_save, tmp_path):
    mock_load.return_value = pl.DataFrame({"id": [1]})

    compile_opts = {"poem_metadata_file": tmp_path / "/out/poem_meta.csv"}

    run_poem_metadata_step(compile_opts, None)

    mock_load.assert_called_once_with(compile_opts)
    mock_extract.assert_not_called()
    mock_save.assert_called_once_with(
        compile_opts["poem_metadata_file"], mock_load.return_value
    )


@patch("corppa.poetry_detection.compile_dataset.save_ppa_metadata")
@patch("corppa.poetry_detection.compile_dataset.add_ref_poems_meta")
@patch("corppa.poetry_detection.compile_dataset.extract_page_meta")
@patch("corppa.poetry_detection.compile_dataset.load_compiled_excerpts")
def test_run_ppa_metadata_step(
    mock_load, mock_extract, mock_add_ref_poems, mock_save, tmp_path
):
    input_df = pl.DataFrame({"id": [1]})

    compile_opts = {
        "poem_metadata_file": tmp_path / "/out/poem_meta.csv",
        "source_ppa_metadata": tmp_path / "/data/ppa_works.csv",
        "ppa_metadata_file": tmp_path / "/out/ppa_meta.csv",
    }

    # call with excerpt dataframe provided
    run_ppa_metadata_step(compile_opts, input_df)
    # doesn't load excerpts because provided
    mock_load.assert_not_called()
    # extracts page/work metadata
    mock_extract.assert_called_once_with(input_df)
    # loads reference poem metadata
    mock_add_ref_poems.assert_called_once_with(
        mock_extract.return_value, compile_opts["poem_metadata_file"]
    )
    mock_save.assert_called_once_with(
        compile_opts["source_ppa_metadata"],
        compile_opts["ppa_metadata_file"],
        mock_add_ref_poems.return_value,
    )

    # call without excerpt df
    mock_extract.reset_mock()
    mock_add_ref_poems.reset_mock()
    mock_save.reset_mock()
    run_ppa_metadata_step(compile_opts, None)
    mock_load.assert_called_once_with(compile_opts)
    mock_extract.assert_not_called()
    mock_add_ref_poems.assert_called_once_with(
        mock_load.return_value, compile_opts["poem_metadata_file"]
    )
    mock_save.assert_called_once_with(
        compile_opts["source_ppa_metadata"],
        compile_opts["ppa_metadata_file"],
        mock_add_ref_poems.return_value,
    )


def test_compress_file(tmp_path):
    # integration test to confirm logic works as expected
    uncompressed_file = tmp_path / "excerpts.csv"
    compressed_file = tmp_path / "excerpts.csv.gz"
    # write out content to test round-trip
    file_contents = "excerpt_id,text\n1,hello\n"
    uncompressed_file.write_text(file_contents)

    compress_file(uncompressed_file, compressed_file)
    # uncompressed file should be removed
    assert not uncompressed_file.exists()
    # compressed file should now be present
    assert compressed_file.exists()

    # uncompressed content should match what we wrote out
    with gzip.open(compressed_file, "rt") as f:
        content = f.read()
    assert content == file_contents
