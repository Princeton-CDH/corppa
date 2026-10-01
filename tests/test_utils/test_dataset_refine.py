# Copyright (c) 2024-2026, Center for Digital Humanities, Princeton University
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import polars as pl
import pytest

import corppa.utils.dataset_refine as dataset_refine


def _make_corpus(tmp_path: Path, rows, refined_rows) -> tuple[Path, Path]:
    """Write ppa_metadata.csv/json + refined.csv; returns (corpus_dir, refined_csv)."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    pl.DataFrame(rows).write_csv(corpus_dir / "ppa_metadata.csv")
    refined_path = tmp_path / "refined.csv"
    pl.DataFrame(refined_rows).write_csv(refined_path)
    with (corpus_dir / "ppa_metadata.json").open("w") as f:
        json.dump(rows, f, indent=2)
    return corpus_dir, refined_path


def _csv_columns(corpus_dir: Path) -> dict[str, list]:
    return pl.read_csv(corpus_dir / "ppa_metadata.csv").to_dict(as_series=False)


def _json_rows(corpus_dir: Path) -> list[dict]:
    with (corpus_dir / "ppa_metadata.json").open() as f:
        return json.load(f)


def test_propagates_refined_author_and_pub_place(tmp_path):
    rows = [
        {
            "work_id": "w1",
            "author": "Alice A.",
            "pub_place": "Cambridge",
            "added": "2024-01-01",
        },
        {
            "work_id": "w2",
            "author": "Bob B.",
            "pub_place": "Oxford",
            "added": "2024-01-01",
        },
    ]
    refined_rows = [
        {
            "work_id": "w1",
            "author": "Alice Adams",
            "pub_place": "Cambridge (UK)",
            "updated": "2024-02-01",
        },
        {
            "work_id": "w2",
            "author": "Bob Brown",
            "pub_place": "Oxford",
            "updated": "2024-02-01",
        },
    ]
    corpus_dir, refined = _make_corpus(tmp_path, rows, refined_rows)

    dataset_refine.refine_metadata(
        refined, corpus_dir / "ppa_metadata.csv", corpus_dir / "ppa_metadata.json"
    )

    out = _csv_columns(corpus_dir)
    assert out["author"] == ["Alice Adams", "Bob Brown"]
    assert out["pub_place"] == ["Cambridge (UK)", "Oxford"]
    js = _json_rows(corpus_dir)
    assert js[0]["author"] == "Alice Adams"
    assert js[0]["pub_place"] == "Cambridge (UK)"
    assert "added" in out  # extra columns are preserved


def test_unmatched_record_keeps_original_values(tmp_path):
    # a work absent from the refined set must not have its author/pub_place
    # wiped to null
    rows = [
        {
            "work_id": "w1",
            "author": "Alice A.",
            "pub_place": "Cambridge",
            "added": "2024-01-01",
        },
        {
            "work_id": "w2",
            "author": "Bob B.",
            "pub_place": "Oxford",
            "added": "2024-01-01",
        },
    ]
    refined_rows = [
        {
            "work_id": "w1",
            "author": "Alice Adams",
            "pub_place": "Cambridge (UK)",
            "updated": "2024-02-01",
        },
    ]
    corpus_dir, refined = _make_corpus(tmp_path, rows, refined_rows)

    dataset_refine.refine_metadata(
        refined, corpus_dir / "ppa_metadata.csv", corpus_dir / "ppa_metadata.json"
    )

    out = _csv_columns(corpus_dir)
    assert out["author"] == ["Alice Adams", "Bob B."]
    assert out["pub_place"] == ["Cambridge (UK)", "Oxford"]
    js = _json_rows(corpus_dir)
    assert js[1]["author"] == "Bob B."


def test_fills_null_original_author_from_refined(tmp_path):
    rows = [
        {"work_id": "w1", "author": None, "pub_place": "Boston", "added": "2024-01-01"}
    ]
    refined_rows = [
        {
            "work_id": "w1",
            "author": "Anonymous",
            "pub_place": "Boston",
            "updated": "2024-02-01",
        }
    ]
    corpus_dir, refined = _make_corpus(tmp_path, rows, refined_rows)

    dataset_refine.refine_metadata(
        refined, corpus_dir / "ppa_metadata.csv", corpus_dir / "ppa_metadata.json"
    )

    assert _csv_columns(corpus_dir)["author"] == ["Anonymous"]


def test_post_refine_records_updated_via_unambiguous_lookup(tmp_path):
    rows = [
        {
            "work_id": "w1",
            "author": "Doe",
            "pub_place": "Cambridge",
            "added": "2024-01-01",
        },
        {
            "work_id": "w2",
            "author": "Doe",
            "pub_place": "Cambridge",
            "added": "2024-07-01",
        },
    ]
    refined_rows = [
        {
            "work_id": "w1",
            "author": "John Doe",
            "pub_place": "Cambridge (UK)",
            "updated": "2024-06-01",
        },
    ]
    corpus_dir, refined = _make_corpus(tmp_path, rows, refined_rows)

    dataset_refine.refine_metadata(
        refined, corpus_dir / "ppa_metadata.csv", corpus_dir / "ppa_metadata.json"
    )

    out = _csv_columns(corpus_dir)
    assert out["author"] == ["John Doe", "John Doe"]
    assert out["pub_place"] == ["Cambridge (UK)", "Cambridge (UK)"]


def test_ambiguous_author_lookup_is_not_applied(tmp_path):
    # "Doe" resolves to two different refined authors, so a post-refine record
    # with author "Doe" cannot be updated automatically and keeps its value
    rows = [
        {"work_id": "w1", "author": "Doe", "pub_place": "X", "added": "2024-01-01"},
        {"work_id": "w2", "author": "Doe", "pub_place": "Y", "added": "2024-01-01"},
        {"work_id": "w3", "author": "Doe", "pub_place": "Z", "added": "2024-07-01"},
    ]
    refined_rows = [
        {
            "work_id": "w1",
            "author": "John Doe",
            "pub_place": "X",
            "updated": "2024-06-01",
        },
        {
            "work_id": "w2",
            "author": "Jane Doe",
            "pub_place": "Y",
            "updated": "2024-06-01",
        },
    ]
    corpus_dir, refined = _make_corpus(tmp_path, rows, refined_rows)

    dataset_refine.refine_metadata(
        refined, corpus_dir / "ppa_metadata.csv", corpus_dir / "ppa_metadata.json"
    )

    out = _csv_columns(corpus_dir)
    assert out["author"] == ["John Doe", "Jane Doe", "Doe"]
    # no rows were lost or duplicated by the ambiguous lookup
    assert len(out["author"]) == 3


def test_ambiguous_pub_place_lookup_is_not_applied(tmp_path):
    # "Cambridge" resolves to two distinct refined places; a post-refine record
    # with pub_place "Cambridge" keeps its original value
    rows = [
        {
            "work_id": "w1",
            "author": "A",
            "pub_place": "Cambridge",
            "added": "2024-01-01",
        },
        {
            "work_id": "w2",
            "author": "B",
            "pub_place": "Cambridge",
            "added": "2024-01-01",
        },
        {
            "work_id": "w3",
            "author": "C",
            "pub_place": "Cambridge",
            "added": "2024-07-01",
        },
    ]
    refined_rows = [
        {
            "work_id": "w1",
            "author": "A",
            "pub_place": "Cambridge (US)",
            "updated": "2024-06-01",
        },
        {
            "work_id": "w2",
            "author": "B",
            "pub_place": "Cambridge (UK)",
            "updated": "2024-06-01",
        },
    ]
    corpus_dir, refined = _make_corpus(tmp_path, rows, refined_rows)

    dataset_refine.refine_metadata(
        refined, corpus_dir / "ppa_metadata.csv", corpus_dir / "ppa_metadata.json"
    )

    out = _csv_columns(corpus_dir)
    assert out["pub_place"] == ["Cambridge (US)", "Cambridge (UK)", "Cambridge"]
    assert len(out["pub_place"]) == 3


def test_duplicate_work_id_in_refined_raises(tmp_path):
    # a duplicated work_id in the refined set inflates the join and trips the
    # no-records-lost assertion
    rows = [{"work_id": "w1", "author": "A", "pub_place": "X", "added": "2024-01-01"}]
    refined_rows = [
        {"work_id": "w1", "author": "B", "pub_place": "X", "updated": "2024-02-01"},
        {"work_id": "w1", "author": "C", "pub_place": "X", "updated": "2024-02-01"},
    ]
    corpus_dir, refined = _make_corpus(tmp_path, rows, refined_rows)

    with pytest.raises(AssertionError):
        dataset_refine.refine_metadata(
            refined, corpus_dir / "ppa_metadata.csv", corpus_dir / "ppa_metadata.json"
        )


def test_backup_files_created(tmp_path):
    rows = [{"work_id": "w1", "author": "A", "pub_place": "X", "added": "2024-01-01"}]
    refined_rows = [
        {"work_id": "w1", "author": "B", "pub_place": "Y", "updated": "2024-02-01"}
    ]
    corpus_dir, refined = _make_corpus(tmp_path, rows, refined_rows)

    dataset_refine.refine_metadata(
        refined, corpus_dir / "ppa_metadata.csv", corpus_dir / "ppa_metadata.json"
    )

    assert (corpus_dir / "ppa_metadata.csv.bk").is_file()
    assert (corpus_dir / "ppa_metadata.json.bk").is_file()
    # backups carry the original, unrefined values
    assert pl.read_csv(corpus_dir / "ppa_metadata.csv.bk")["author"].to_list() == ["A"]


def test_column_order_and_row_count_preserved(tmp_path):
    rows = [
        {
            "work_id": "w1",
            "title": "T1",
            "author": "A",
            "pub_place": "X",
            "added": "2024-01-01",
        },
        {
            "work_id": "w2",
            "title": "T2",
            "author": "B",
            "pub_place": "Y",
            "added": "2024-01-01",
        },
    ]
    refined_rows = [
        {"work_id": "w1", "author": "C", "pub_place": "Z", "updated": "2024-02-01"}
    ]
    corpus_dir, refined = _make_corpus(tmp_path, rows, refined_rows)
    csv_path = corpus_dir / "ppa_metadata.csv"
    original_columns = pl.read_csv(csv_path).columns

    dataset_refine.refine_metadata(refined, csv_path, corpus_dir / "ppa_metadata.json")

    out_df = pl.read_csv(csv_path)
    assert out_df.columns == original_columns
    assert out_df.height == 2


def test_json_row_without_author_pub_place_keys_unchanged(tmp_path):
    rows = [
        {"work_id": "w1", "author": "A", "pub_place": "X", "added": "2024-01-01"},
        {"work_id": "w2", "title": "no author field", "added": "2024-01-01"},
    ]
    refined_rows = [
        {"work_id": "w1", "author": "B", "pub_place": "Y", "updated": "2024-02-01"}
    ]
    corpus_dir, refined = _make_corpus(tmp_path, rows, refined_rows)

    dataset_refine.refine_metadata(
        refined, corpus_dir / "ppa_metadata.csv", corpus_dir / "ppa_metadata.json"
    )

    js = _json_rows(corpus_dir)
    assert js[0]["author"] == "B"
    # the second json row has neither key, so it stays untouched even though its
    # csv counterpart's author was (coalesced)...
    assert "author" not in js[1]
    assert js[1]["title"] == "no author field"


def test_json_order_mismatch_raises(tmp_path):
    rows = [
        {"work_id": "w1", "author": "A", "pub_place": "X", "added": "2024-01-01"},
        {"work_id": "w2", "author": "B", "pub_place": "Y", "added": "2024-01-01"},
    ]
    refined_rows = [
        {"work_id": "w1", "author": "C", "pub_place": "Z", "updated": "2024-02-01"}
    ]
    corpus_dir, refined = _make_corpus(tmp_path, rows, refined_rows)
    # shuffle the json rows so they no longer match the csv order
    with (corpus_dir / "ppa_metadata.json").open("w") as f:
        json.dump([rows[1], rows[0]], f, indent=2)

    with pytest.raises(AssertionError):
        dataset_refine.refine_metadata(
            refined, corpus_dir / "ppa_metadata.csv", corpus_dir / "ppa_metadata.json"
        )


# --- main() CLI ---


def test_main_missing_refined_file_exits(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        ["dataset_refine.py", str(tmp_path / "nope.csv"), str(corpus_dir)],
    )

    with pytest.raises(SystemExit, match="Input metadata file"):
        dataset_refine.main()


def test_main_missing_csv_exits(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    refined = tmp_path / "refined.csv"
    refined.write_text("work_id,author,pub_place,updated\n")
    with (corpus_dir / "ppa_metadata.json").open("w") as f:
        json.dump([], f)  # only json present; csv is missing
    monkeypatch.setattr(
        "sys.argv", ["dataset_refine.py", str(refined), str(corpus_dir)]
    )

    with pytest.raises(SystemExit, match="ppa_metadata.csv not found"):
        dataset_refine.main()


def test_main_full_run_exits_zero(tmp_path, monkeypatch, capsys):
    rows = [{"work_id": "w1", "author": "A", "pub_place": "X", "added": "2024-01-01"}]
    refined_rows = [
        {"work_id": "w1", "author": "B", "pub_place": "Y", "updated": "2024-02-01"}
    ]
    corpus_dir, refined = _make_corpus(tmp_path, rows, refined_rows)
    monkeypatch.setattr(
        "sys.argv", ["dataset_refine.py", str(refined), str(corpus_dir)]
    )

    dataset_refine.main()

    assert "Updated author in 1 works" in capsys.readouterr().out
    assert _csv_columns(corpus_dir)["author"] == ["B"]
