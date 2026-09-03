# Copyright (c) 2024-2026, Center for Digital Humanities, Princeton University
# SPDX-License-Identifier: Apache-2.0

import tarfile
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import orjsonl
import polars as pl
import pytest

from corppa.utils.dataset_prep import (
    add_zip_file_to_tar,
    align_pages,
    align_shifted_pages,
    get_zip_textfiles,
    main,
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
    # mapping is keyed by the full page id (matches how process_ht_work looks it up)
    assert result == {
        "work.00000001": "00000001",
        "work.00000002": "00000002",
        "work.00000003": "00000003",
    }


def test_align_pages_low_match_falls_through_to_shifted(tmp_path):
    # Content differs entirely -> avg score is low -> falls through to
    # align_shifted_pages, which finds no matches and returns an empty
    # mapping, so align_pages returns None.
    # align_shifted_pages needs an `order` column and long-enough texts.
    pages_df = pl.DataFrame(
        {
            "id": ["work.00000001", "work.00000002", "work.00000003"],
            "order": [1, 2, 3],
            "text": [_long_text(f"alpha-{i}") for i in range(3)],
        }
    )
    zip_path = make_zip(
        tmp_path,
        {f"0000000{i + 1}.txt": _long_text(f"zzzzz-{i}-qqqqq") for i in range(3)},
    )
    with ZipFile(zip_path) as zf:
        assert align_pages(WORK_ID, pages_df, zf) is None


def test_align_pages_join_mismatch_returns_partial(tmp_path, pages_df):
    # Zip is missing one page -> join count mismatch is warned about but the
    # partial mapping for the pages that did join is still returned.
    zip_path = make_zip(
        tmp_path,
        {
            "00000001.txt": PAGE_TEXTS["00000001"],
            "00000002.txt": PAGE_TEXTS["00000002"],
        },
    )
    with ZipFile(zip_path) as zf:
        assert align_pages(WORK_ID, pages_df, zf) == {
            "work.00000001": "00000001",
            "work.00000002": "00000002",
        }


def test_align_pages_insufficient_zip_pages(tmp_path, pages_df):
    # Zip has only one of the corpus's three pages -> partial mapping returned.
    zip_path = make_zip(tmp_path, {"00000001.txt": PAGE_TEXTS["00000001"]})
    with ZipFile(zip_path) as zf:
        assert align_pages(WORK_ID, pages_df, zf) == {
            "work.00000001": "00000001",
        }


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
            "work.00000001": "OSU_32435051461309_00000001",
            "work.00000002": "OSU_32435051461309_00000002",
        }


# --- align_shifted_pages ---


def _long_text(seed: str, length: int = 700) -> str:
    """Repeat `seed` until the result exceeds `length` chars so it survives
    the >600-char filter inside align_shifted_pages, while staying distinct
    from other seeds (each repeated seed produces a unique long string)."""
    reps = (length // len(seed)) + 2
    return (seed + " ") * reps


def _make_shifted_frames(page_orders, zip_orders, seeds):
    """Build (pages_df, zip_pages_df) where pages_df.order[i] and
    zip_pages_df.order[i] share the same long text derived from seeds[i]."""
    texts = [_long_text(s) for s in seeds]
    pages_df = pl.DataFrame(
        {
            "id": [f"work.{o:08d}" for o in page_orders],
            "order": page_orders,
            "text": texts,
        }
    )
    zip_pages_df = pl.DataFrame(
        {
            "page_filename": [f"{o:08d}" for o in zip_orders],
            "order": zip_orders,
            "text": texts,
        }
    )
    return pages_df, zip_pages_df


def test_align_shifted_pages_consistent_shift():
    # pages orders 1..5 correspond to zip orders 11..15 (uniform shift = -10);
    # every page should map to its shifted zip counterpart, including the
    # first and last anchors.
    seeds = [f"chapter-{i}-unique-content" for i in range(5)]
    pages_df, zip_pages_df = _make_shifted_frames(
        page_orders=list(range(1, 6)),
        zip_orders=list(range(11, 16)),
        seeds=seeds,
    )

    result = align_shifted_pages(pages_df, zip_pages_df)

    assert result is not None
    mapping = dict(result.select(["id", "page_filename"]).iter_rows())
    assert mapping == {
        "work.00000001": "00000011",
        "work.00000002": "00000012",
        "work.00000003": "00000013",
        "work.00000004": "00000014",
        "work.00000005": "00000015",
    }


def test_align_shifted_pages_includes_head_pages():
    # Pages before the first anchor are typically short pages that got
    # filtered out. They must still be included in the mapping via the
    # first anchor's shift.
    seeds = [f"chapter-{i}-unique-content" for i in range(6)]
    zip_pages_df = pl.DataFrame(
        {
            "page_filename": [f"{11 + i:08d}" for i in range(6)],
            "order": [11 + i for i in range(6)],
            "text": [_long_text(s) for s in seeds],
        }
    )
    # page 1 is short (below the 600-char filter); pages 2-6 are long and
    # share text with zip pages at orders 12-16 (uniform shift = -10)
    pages_df = pl.DataFrame(
        {
            "id": [f"work.{1 + i:08d}" for i in range(6)],
            "order": [1 + i for i in range(6)],
            "text": ["short leading page"] + [_long_text(s) for s in seeds[1:]],
        }
    )

    result = align_shifted_pages(pages_df, zip_pages_df)

    assert result is not None
    mapping = dict(result.select(["id", "page_filename"]).iter_rows())
    # page 1 (short, before the first anchor) is mapped via the anchor's shift
    assert mapping["work.00000001"] == "00000011"
    # last page still covered by the tail fix
    assert mapping["work.00000006"] == "00000016"


def test_align_shifted_pages_returns_id_and_filename_columns():
    seeds = [f"page-{i}-content" for i in range(3)]
    pages_df, zip_pages_df = _make_shifted_frames([1, 2, 3], [5, 6, 7], seeds)

    result = align_shifted_pages(pages_df, zip_pages_df)

    assert result is not None
    assert set(result.columns) >= {"id", "page_filename"}


def test_align_shifted_pages_no_content_match():
    # No shared content between pages and zip -> no page clears the cutoff
    pages_df, _ = _make_shifted_frames(
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [f"alpha-{i}" for i in range(5)],
    )
    _, zip_pages_df = _make_shifted_frames(
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [f"zzzzz-{i}-qqqqq" for i in range(5)],
    )

    result = align_shifted_pages(pages_df, zip_pages_df)

    assert result.is_empty()


def test_align_shifted_pages_small_df_uses_all_anchors():
    # Only 2 long pages in pages_df -> can't sample first/middle/last without
    # duplicates; all long pages get used as anchors, and if they agree on a
    # shift, every original page (short or long) gets mapped.
    zip_pages_df = pl.DataFrame(
        {
            "page_filename": [f"{11 + i:08d}" for i in range(3)],
            "order": [11, 12, 13],
            "text": [_long_text(f"chapter-{i}") for i in range(3)],
        }
    )
    # pages 1 and 2 are long (used as anchors), page 3 is short
    pages_df = pl.DataFrame(
        {
            "id": ["work.00000001", "work.00000002", "work.00000003"],
            "order": [1, 2, 3],
            "text": [
                _long_text("chapter-0"),
                _long_text("chapter-1"),
                "short trailing page",
            ],
        }
    )

    result = align_shifted_pages(pages_df, zip_pages_df)

    assert result is not None
    mapping = dict(result.select(["id", "page_filename"]).iter_rows())
    # all three pages mapped via the shared shift (-10)
    assert mapping == {
        "work.00000001": "00000011",
        "work.00000002": "00000012",
        "work.00000003": "00000013",
    }


def test_align_shifted_pages_single_long_anchor():
    # Only one long page; the single-anchor shift is applied to every
    # original page via the head-chunk-extended-to-end branch.
    zip_pages_df = pl.DataFrame(
        {
            "page_filename": ["00000011", "00000012", "00000013"],
            "order": [11, 12, 13],
            "text": [_long_text(f"chapter-{i}") for i in range(3)],
        }
    )
    pages_df = pl.DataFrame(
        {
            "id": ["work.00000001", "work.00000002", "work.00000003"],
            "order": [1, 2, 3],
            "text": ["short", _long_text("chapter-1"), "also short"],
        }
    )

    result = align_shifted_pages(pages_df, zip_pages_df)

    mapping = dict(result.select(["id", "page_filename"]).iter_rows())
    assert mapping == {
        "work.00000001": "00000011",
        "work.00000002": "00000012",
        "work.00000003": "00000013",
    }


def test_align_shifted_pages_all_pages_short_returns_empty():
    # Every page below the 600-char filter -> no long-enough anchors.
    # Short-chunk guard returns an empty mapping instead of crashing.
    pages_df = pl.DataFrame(
        {
            "id": [f"work.{i:08d}" for i in range(1, 4)],
            "order": [1, 2, 3],
            "text": ["short one", "short two", "short three"],
        }
    )
    _, zip_pages_df = _make_shifted_frames(
        [1, 2, 3], [1, 2, 3], ["short one", "short two", "short three"]
    )

    result = align_shifted_pages(pages_df, zip_pages_df)

    assert result is not None
    assert result.is_empty()


def test_align_pages_underscore_page_id(aligned_zip):
    # Corpus page ids use underscore separator instead of dot
    pages_df = make_pages_df(["work_00000001", "work_00000002", "work_00000003"])
    with ZipFile(aligned_zip) as zf:
        result = align_pages(WORK_ID, pages_df, zf)
    assert isinstance(result, dict)
    assert set(result.keys()) == {
        "work_00000001",
        "work_00000002",
        "work_00000003",
    }


# --- main / --continue ---


def _pages_through(work_id, pages, image_dir, tar):
    """Stand-in for process_work that yields pages unchanged (no images)."""
    yield from pages


def _write_corpus(path: Path, page_records: list[dict]) -> None:
    """Write a list of page dicts to a JSONL corpus file."""
    orjsonl.save(path, page_records)


def _run_main(input_path, image_dir, output_dir, extra_args=None):
    """Invoke main() with the given positional args (+ optional extras),
    patching process_work so no image/zip handling is exercised."""
    argv = [
        "dataset_prep.py",
        str(input_path),
        str(image_dir),
        str(output_dir),
    ]
    if extra_args:
        argv += extra_args
    with (
        patch("sys.argv", argv),
        patch(
            "corppa.utils.dataset_prep.process_work",
            side_effect=_pages_through,
        ),
    ):
        main()


@pytest.fixture
def corpus_input(tmp_path):
    """A small two-work corpus with two pages each."""
    input_path = tmp_path / "input.jsonl"
    _write_corpus(
        input_path,
        [
            {"work_id": "workA", "id": "workA.0001", "text": "a1"},
            {"work_id": "workA", "id": "workA.0002", "text": "a2"},
            {"work_id": "workB", "id": "workB.0001", "text": "b1"},
            {"work_id": "workB", "id": "workB.0002", "text": "b2"},
        ],
    )
    return input_path


def test_main_writes_all_works(tmp_path, corpus_input):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"

    _run_main(corpus_input, image_dir, output_dir)

    output_pages = output_dir / "ppa_pages.jsonl"
    output_tar = output_dir / "ppa_images.tar"
    assert output_pages.exists()
    # tar is uncompressed (not .tar.gz) so it can be appended to on continue
    assert output_tar.exists()
    assert not (output_dir / "ppa_images.tar.gz").exists()

    written = list(orjsonl.stream(output_pages))
    assert [p["id"] for p in written] == [
        "workA.0001",
        "workA.0002",
        "workB.0001",
        "workB.0002",
    ]


def test_main_continue_skips_completed_works(tmp_path, corpus_input):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    output_pages = output_dir / "ppa_pages.jsonl"
    output_tar = output_dir / "ppa_images.tar"
    # simulate a previous run that already completed workA
    _write_corpus(
        output_pages,
        [
            {"work_id": "workA", "id": "workA.0001", "text": "a1"},
            {"work_id": "workA", "id": "workA.0002", "text": "a2"},
        ],
    )
    # and produced an existing (uncompressed) tar
    with tarfile.open(output_tar, "w"):
        pass

    _run_main(corpus_input, image_dir, output_dir, extra_args=["--continue"])

    written = list(orjsonl.stream(output_pages))
    # workA pages are preserved and only appear once; workB is appended
    assert [p["id"] for p in written] == [
        "workA.0001",
        "workA.0002",
        "workB.0001",
        "workB.0002",
    ]


def test_main_continue_does_not_rename_existing_output(tmp_path, corpus_input):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    output_pages = output_dir / "ppa_pages.jsonl"
    _write_corpus(
        output_pages,
        [{"work_id": "workA", "id": "workA.0001", "text": "a1"}],
    )

    _run_main(corpus_input, image_dir, output_dir, extra_args=["--continue"])

    # continue appends in place; it must not create a .bak backup
    assert not (output_dir / "ppa_pages.jsonl.bak").exists()


def test_main_continue_missing_output_starts_fresh(tmp_path, corpus_input):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"

    # --continue with no existing output should behave like a fresh run
    _run_main(corpus_input, image_dir, output_dir, extra_args=["--continue"])

    written = list(orjsonl.stream(output_dir / "ppa_pages.jsonl"))
    assert [p["id"] for p in written] == [
        "workA.0001",
        "workA.0002",
        "workB.0001",
        "workB.0002",
    ]


def test_main_without_continue_renames_existing_output(tmp_path, corpus_input):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    output_pages = output_dir / "ppa_pages.jsonl"
    _write_corpus(
        output_pages,
        [{"work_id": "old", "id": "old.0001", "text": "old"}],
    )

    _run_main(corpus_input, image_dir, output_dir)

    # existing output is renamed to a .bak file and rewritten fresh
    backup = output_dir / "ppa_pages.jsonl.bak"
    assert backup.exists()
    assert [p["id"] for p in orjsonl.stream(backup)] == ["old.0001"]
    written = list(orjsonl.stream(output_pages))
    assert [p["id"] for p in written] == [
        "workA.0001",
        "workA.0002",
        "workB.0001",
        "workB.0002",
    ]
