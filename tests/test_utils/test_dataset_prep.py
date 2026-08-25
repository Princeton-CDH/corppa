# Copyright (c) 2024-2026, Center for Digital Humanities, Princeton University
# SPDX-License-Identifier: Apache-2.0

import tarfile
from pathlib import Path
from zipfile import ZipFile

import polars as pl
import pytest

from corppa.utils.dataset_prep import (
    add_zip_file_to_tar,
    align_pages,
    get_zip_textfiles,
)

WORK_ID = "htid:test.12345678"

# Realistic multi-word page texts so str_fuzz scores high on identical content
PAGE_TEXTS = {
    "00000001": "The quick brown fox jumps over the lazy dog.",
    "00000002": "To be or not to be, that is the question.",
    "00000003": "It was the best of times, it was the worst of times.",
}


def make_zip(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create tmp_path/test.zip with the given filename->content mapping."""
    zip_path = tmp_path / "test.zip"
    with ZipFile(zip_path, "w") as zf:
        for filename, content in files.items():
            zf.writestr(filename, content)
    return zip_path


def make_pages_df(page_ids: list[str]) -> pl.DataFrame:
    """Build a minimal pages DataFrame from dot- or underscore-separated ids
    like 'work.00000001' or 'work_00000001', looked up against PAGE_TEXTS."""
    import re

    return pl.DataFrame(
        {
            "id": page_ids,
            "text": [
                PAGE_TEXTS[re.search(r"[0-9]+$", pid).group()] for pid in page_ids
            ],
        }
    )


@pytest.fixture
def aligned_zip(tmp_path):
    """Zip with all three PAGE_TEXTS entries using plain numeric filenames."""
    files = {f"{pid}.txt": text for pid, text in PAGE_TEXTS.items()}
    return make_zip(tmp_path, files)


@pytest.fixture
def pages_df():
    """Pages DataFrame with dot-separated ids for all three PAGE_TEXTS entries."""
    return make_pages_df(["work.00000001", "work.00000002", "work.00000003"])


# --- get_zip_textfiles ---


def test_get_zip_textfiles_returns_iterator(tmp_path):
    with ZipFile(make_zip(tmp_path, {"00000001.txt": "text"})) as zf:
        result = get_zip_textfiles(zf)
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")


def test_get_zip_textfiles_multiple(tmp_path):
    files = {"00000001.txt": "page one", "00000002.txt": "page two"}
    with ZipFile(make_zip(tmp_path, files)) as zf:
        assert dict(get_zip_textfiles(zf)) == {
            "00000001": "page one",
            "00000002": "page two",
        }


def test_get_zip_textfiles_skips_non_txt(tmp_path):
    files = {
        "00000001.txt": "page text",
        "image.jpg": "binary",
        "metadata.xml": "<xml/>",
    }
    with ZipFile(make_zip(tmp_path, files)) as zf:
        assert list(get_zip_textfiles(zf)) == [("00000001", "page text")]


def test_get_zip_textfiles_no_txt_files(tmp_path):
    # covers both empty zip and zip with only non-txt files
    with ZipFile(make_zip(tmp_path, {})) as zf:
        assert list(get_zip_textfiles(zf)) == []
    with ZipFile(make_zip(tmp_path, {"image.jpg": "binary"})) as zf:
        assert list(get_zip_textfiles(zf)) == []


def test_get_zip_textfiles_prefixed_filename(tmp_path):
    # OSU-style filenames: stem is preserved as-is
    with ZipFile(
        make_zip(tmp_path, {"OSU_32435051461309_00000602.txt": "page text"})
    ) as zf:
        assert list(get_zip_textfiles(zf)) == [
            ("OSU_32435051461309_00000602", "page text")
        ]


def test_get_zip_textfiles_utf8_content(tmp_path):
    content = "café naïve résumé"
    with ZipFile(make_zip(tmp_path, {"00000001.txt": content})) as zf:
        assert list(get_zip_textfiles(zf)) == [("00000001", content)]


# --- add_zip_file_to_tar ---


def test_add_zip_file_to_tar(tmp_path):
    content = b"image data"
    zip_path = make_zip(tmp_path, {"00000001.jpg": content})
    tar_path = tmp_path / "images.tar"
    with ZipFile(zip_path) as zf:
        with tarfile.open(tar_path, "w") as tar:
            add_zip_file_to_tar(zf, "00000001.jpg", tar, "work_id/00000001.jpg")
    with tarfile.open(tar_path) as tar:
        assert tar.getnames() == ["work_id/00000001.jpg"]
        assert tar.extractfile("work_id/00000001.jpg").read() == content


# --- align_pages ---


def test_align_pages_good_match_returns_mapping(pages_df, aligned_zip):
    with ZipFile(aligned_zip) as zf:
        result = align_pages(WORK_ID, pages_df, zf)
    assert result == {
        "00000001": "00000001",
        "00000002": "00000002",
        "00000003": "00000003",
    }


def test_align_pages_low_match_returns_none(tmp_path, pages_df):
    zip_path = make_zip(
        tmp_path,
        {
            "00000001.txt": "zzz qqq xxx aaa bbb ccc ddd eee fff ggg",
            "00000002.txt": "111 222 333 444 555 666 777 888 999 000",
            "00000003.txt": "alpha beta gamma delta epsilon zeta eta",
        },
    )
    with ZipFile(zip_path) as zf:
        assert align_pages(WORK_ID, pages_df, zf) is None


def test_align_pages_join_mismatch_returns_none(tmp_path, pages_df):
    # Zip is missing one page -> join count mismatch -> returns None
    zip_path = make_zip(
        tmp_path,
        {
            "00000001.txt": PAGE_TEXTS["00000001"],
            "00000002.txt": PAGE_TEXTS["00000002"],
        },
    )
    with ZipFile(zip_path) as zf:
        assert align_pages(WORK_ID, pages_df, zf) is None


def test_align_pages_insufficient_zip_pages(tmp_path, pages_df):
    # Zip has fewer pages than corpus -> join mismatch -> returns None
    zip_path = make_zip(tmp_path, {"00000001.txt": PAGE_TEXTS["00000001"]})
    with ZipFile(zip_path) as zf:
        assert align_pages(WORK_ID, pages_df, zf) is None


def test_align_pages_prefixed_filenames(tmp_path):
    # OSU-style zip filenames: page_id extracted from numeric suffix
    pages_df = make_pages_df(["work.00000001", "work.00000002"])
    zip_path = make_zip(
        tmp_path,
        {
            "OSU_32435051461309_00000001.txt": PAGE_TEXTS["00000001"],
            "OSU_32435051461309_00000002.txt": PAGE_TEXTS["00000002"],
        },
    )
    with ZipFile(zip_path) as zf:
        assert align_pages(WORK_ID, pages_df, zf) == {
            "00000001": "OSU_32435051461309_00000001",
            "00000002": "OSU_32435051461309_00000002",
        }


def test_align_pages_underscore_page_id(aligned_zip):
    # Corpus page ids use underscore separator instead of dot
    pages_df = make_pages_df(["work_00000001", "work_00000002", "work_00000003"])
    with ZipFile(aligned_zip) as zf:
        result = align_pages(WORK_ID, pages_df, zf)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"00000001", "00000002", "00000003"}
