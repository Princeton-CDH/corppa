# Copyright (c) 2024-2026, Center for Digital Humanities, Princeton University
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from zipfile import ZipFile

from corppa.utils.dataset_prep import get_zip_textfiles


def make_zip(tmp_path: Path, files: dict[str, str]) -> Path:
    """Helper: create a zip file at tmp_path/test.zip containing the given
    filename->content mapping. Returns the path to the zip file."""
    zip_path = tmp_path / "test.zip"
    with ZipFile(zip_path, "w") as zf:
        for filename, content in files.items():
            zf.writestr(filename, content)
    return zip_path


def test_get_zip_textfiles_single(tmp_path):
    zip_path = make_zip(tmp_path, {"00000001.txt": "page one text"})
    results = list(get_zip_textfiles(zip_path))
    assert results == [("00000001", "page one text")]


def test_get_zip_textfiles_multiple(tmp_path):
    files = {
        "00000001.txt": "page one",
        "00000002.txt": "page two",
        "00000003.txt": "page three",
    }
    zip_path = make_zip(tmp_path, files)
    results = dict(get_zip_textfiles(zip_path))
    assert results == {
        "00000001": "page one",
        "00000002": "page two",
        "00000003": "page three",
    }


def test_get_zip_textfiles_skips_non_txt(tmp_path):
    files = {
        "00000001.txt": "page text",
        "image.jpg": "not text",
        "metadata.xml": "<xml/>",
    }
    zip_path = make_zip(tmp_path, files)
    results = list(get_zip_textfiles(zip_path))
    assert len(results) == 1
    assert results[0] == ("00000001", "page text")


def test_get_zip_textfiles_empty_zip(tmp_path):
    zip_path = make_zip(tmp_path, {})
    results = list(get_zip_textfiles(zip_path))
    assert results == []


def test_get_zip_textfiles_no_txt_files(tmp_path):
    zip_path = make_zip(tmp_path, {"image.jpg": "binary", "readme.md": "docs"})
    results = list(get_zip_textfiles(zip_path))
    assert results == []


def test_get_zip_textfiles_prefixed_filename(tmp_path):
    # some works have filenames like OSU_32435051461309_00000602.txt
    zip_path = make_zip(tmp_path, {"OSU_32435051461309_00000602.txt": "page text"})
    results = list(get_zip_textfiles(zip_path))
    assert results == [("OSU_32435051461309_00000602", "page text")]


def test_get_zip_textfiles_returns_iterator(tmp_path):
    zip_path = make_zip(tmp_path, {"00000001.txt": "text"})
    result = get_zip_textfiles(zip_path)
    # should be a generator/iterator, not a list
    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")


def test_get_zip_textfiles_utf8_content(tmp_path):
    content = "café naïve résumé"
    zip_path = make_zip(tmp_path, {"00000001.txt": content})
    results = list(get_zip_textfiles(zip_path))
    assert results == [("00000001", content)]
