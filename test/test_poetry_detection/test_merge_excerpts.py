import csv
from dataclasses import replace
from unittest.mock import patch

import polars as pl
import pytest
from test_polars_utils import _excerpts_to_csv

from corppa.poetry_detection.core import Excerpt, LabeledExcerpt
from corppa.poetry_detection.merge_excerpts import (
    combine_excerpts,
    main,
    merge_labeled_excerpts,
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


def test_combine_excerpts_1ex_1label():
    # excerpt + labeled excerpt (same id)
    df = pl.from_dicts([excerpt1.to_dict()])
    other_df = pl.from_dicts([excerpt1_label1.to_dict()])
    merged = combine_excerpts(df, other_df)
    # expect one row
    assert len(merged) == 1
    # should have all columns for labeled excerpt (order-agnostic)
    assert set(merged.columns) == set(LabeledExcerpt.fieldnames())
    row = merged.row(0, named=True)
    merged_excerpt = LabeledExcerpt.from_dict(row)
    # result should exactly match the labeled excerpt since all other fields are same
    assert merged_excerpt == excerpt1_label1


def test_combine_excerpts_1ex_2labels():
    # excerpt + two labeled excerpt (same excerpt id, two different ref ids)
    df = pl.from_dicts([excerpt1.to_dict()])
    other_df = pl.from_dicts([excerpt1_label1.to_dict(), excerpt1_label2.to_dict()])
    merged = combine_excerpts(df, other_df)
    # expect two rows with  two different labels
    assert len(merged) == 2
    # polars preserves order, so we can check that they match what was sent in
    # results should exactly match the labeled excerpts since all other fields are same
    assert LabeledExcerpt.from_dict(merged.row(0, named=True)) == excerpt1_label1
    assert LabeledExcerpt.from_dict(merged.row(1, named=True)) == excerpt1_label2


def test_combine_excerpts_1ex_note_1label():
    # excerpt with note + labeled excerpt (same id)
    ex1_notes = replace(excerpt1, notes="detection information")
    df = pl.from_dicts([ex1_notes.to_dict()])
    other_df = pl.from_dicts([excerpt1_label1.to_dict()])
    merged = combine_excerpts(df, other_df)
    # expect one row
    assert len(merged) == 1
    # should have all columns for labeled excerpt - no extra notes field
    assert set(merged.columns) == set(LabeledExcerpt.fieldnames())
    merged_excerpt = LabeledExcerpt.from_dict(merged.row(0, named=True))
    # result should match the labeled excerpt except for the updated notes field
    assert merged_excerpt != excerpt1_label1
    expected_notes = "\n".join([ex1_notes.notes, excerpt1_label1.notes])
    excerpt_with_notes = replace(excerpt1_label1, notes=expected_notes)
    assert merged_excerpt == excerpt_with_notes


def test_combine_excerpts_1ex_different_label():
    # excerpt 2 + labeled excerpt 1 - should preserve both
    df = pl.from_dicts([excerpt2.to_dict()])
    other_df = pl.from_dicts([excerpt1_label1.to_dict()])
    merged = combine_excerpts(df, other_df)
    # expect two rows
    assert len(merged) == 2
    # should have all columns for labeled excerpt (order-agnostic)
    assert set(merged.columns) == set(LabeledExcerpt.fieldnames())
    # the first row is our unlabeled excerpt
    row = merged.row(0, named=True)
    # filter out null values (unset labeled excerpt fields) and init as Excerpt
    row_subset = {k: v for k, v in row.items() if v is not None}
    merged_excerpt2 = Excerpt.from_dict(row_subset)
    assert merged_excerpt2 == excerpt2
    # second row should be our labeled excerpt
    merged_excerpt1_label1 = LabeledExcerpt.from_dict(merged.row(1, named=True))
    assert merged_excerpt1_label1 == excerpt1_label1


def test_combine_excerpts_two_different_labels():
    # two different labeled excerpts should not be merged
    assert excerpt1_label1.excerpt_id != excerpt2_label1.excerpt_id
    df = pl.from_dicts([excerpt1_label1.to_dict()])
    other_df = pl.from_dicts([excerpt2_label1.to_dict()])
    merged = combine_excerpts(df, other_df)
    # expect two rows
    assert len(merged) == 2
    # should have all columns for labeled excerpt (order-agnostic)
    assert set(merged.columns) == set(LabeledExcerpt.fieldnames())
    # order should match input labeled excerpts
    assert LabeledExcerpt.from_dict(merged.row(0, named=True)) == excerpt1_label1
    assert LabeledExcerpt.from_dict(merged.row(1, named=True)) == excerpt2_label1


def test_combine_excerpts_1ex_2labels_diffmethod():
    # unlabeled excerpt + two matching labeled excerpts
    # - same excerpt id, two labels with same ref ids but different method
    # combine method does not merge these
    df = pl.from_dicts([excerpt1.to_dict()])
    # everything the same except for the method (unlikely!)
    excerpt1_label1_method2 = replace(
        excerpt1_label1, identification_methods={"refmatcha"}
    )
    other_df = pl.from_dicts(
        [excerpt1_label1.to_dict(), excerpt1_label1_method2.to_dict()]
    )
    # left and right are merged but duplicate ids within the other df
    # are not merged by this method
    merged = combine_excerpts(df, other_df)
    assert len(merged) == 2


def test_combine_different_labels():
    # combine should NOT merge labeled excerpts with different poem id
    df = pl.from_dicts([excerpt1_label1.to_dict()])
    excerpt1_diff_label = replace(excerpt1_label1, poem_id="Z1234")
    other_df = pl.from_dicts([excerpt1_diff_label.to_dict()])

    # distinct poem ids should NOT be merged
    merged = combine_excerpts(df, other_df)
    assert len(merged) == 2


def test_merge_labeled_excerpts():
    # excerpt + two matching labeled excerpts
    # - same excerpt id, two labels with same ref ids but different method

    # everything the same except for the method (unlikely!)
    excerpt1_label1_method2 = replace(
        excerpt1_label1, identification_methods={"refmatcha"}
    )
    df = pl.from_dicts([excerpt1_label1.to_dict(), excerpt1_label1_method2.to_dict()])
    merged = merge_labeled_excerpts(df)
    assert len(merged) == 1
    # should have all columns for labeled excerpt (order-agnostic)
    assert set(merged.columns) == set(LabeledExcerpt.fieldnames())
    excerpt = LabeledExcerpt.from_dict(merged.row(0, named=True))
    # should have both methods
    assert excerpt.identification_methods == {"manual", "refmatcha"}

    # more likely scenario: manual label with no ref span, system label with more details
    excerpt1_label1_other = replace(
        excerpt1_label1,
        ref_span_start=None,
        ref_span_end=None,
        ref_span_text=None,
        identification_methods={"other"},
    )
    df = pl.from_dicts([excerpt1_label1.to_dict(), excerpt1_label1_other.to_dict()])
    merged = merge_labeled_excerpts(df)
    assert len(merged) == 1
    # should have all columns for labeled excerpt (order-agnostic)
    assert set(merged.columns) == set(LabeledExcerpt.fieldnames())
    # should have both methods
    assert merged.row(0, named=True)["identification_methods"] == ["manual", "other"]
    excerpt = LabeledExcerpt.from_dict(merged.row(0, named=True))
    assert excerpt.identification_methods == {"manual", "other"}

    # order should not matter
    df = pl.from_dicts([excerpt1_label1_other.to_dict(), excerpt1_label1.to_dict()])
    merged = merge_labeled_excerpts(df)
    assert len(merged) == 1
    excerpt = LabeledExcerpt.from_dict(merged.row(0, named=True))
    assert excerpt.identification_methods == {"manual", "other"}
    # should have the non-null ref values
    assert excerpt.ref_span_start == excerpt1_label1.ref_span_start
    assert excerpt.ref_span_end == excerpt1_label1.ref_span_end
    assert excerpt.ref_span_text == excerpt1_label1.ref_span_text


def test_main_argparse_errors(capsys, tmp_path):
    # call with only one input file (two is minimum required)
    with patch("sys.argv", ["merge_excerpts.py", "input", "-o", "output"]):
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "at least two input files are required for merging" in captured.err

    # output file already exists
    outfile = tmp_path / "merged.csv"
    outfile.touch()
    with patch("sys.argv", ["merge_excerpts.py", "input", "-o", str(outfile)]):
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert f"{outfile} already exists, not overwriting" in captured.err

    # input files don't exist
    input1 = tmp_path / "excerpts.csv"
    input2 = tmp_path / "more_excerpts.csv"
    # both input files don't actually eixst
    with patch(
        "sys.argv", ["merge_excerpts.py", str(input1), str(input2), "-o", "output"]
    ):
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "input files not found" in captured.err
        assert str(input1) in captured.err
        assert str(input2) in captured.err
    # one file exists, the other doesn't
    input1.touch()
    with patch(
        "sys.argv", ["merge_excerpts.py", str(input1), str(input2), "-o", "output"]
    ):
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "input files not found" in captured.err
        assert str(input1) not in captured.err
        assert str(input2) in captured.err
    # input file order shouldn't matter - same error if inputs reversed
    with patch(
        "sys.argv", ["merge_excerpts.py", str(input2), str(input1), "-o", "output"]
    ):
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "input files not found" in captured.err
        assert str(input1) not in captured.err
        assert str(input2) in captured.err


def test_main_invalid_input(capsys, tmp_path):
    excerpt_datafile = tmp_path / "excerpts.csv"
    # valid excerpt data
    _excerpts_to_csv(excerpt_datafile, [excerpt1])
    other_data = tmp_path / "other.csv"
    # invalid - non excerpt data
    # NOTE: copied from earlier test; consider converting to fixture
    with other_data.open("w", encoding="utf-8") as filehandle:
        csv_writer = csv.writer(filehandle)
        csv_writer.writerow(["id", "note"])
        csv_writer.writerow(["p.01", "missing"])

    with patch(
        "sys.argv",
        ["merge_excerpts.py", str(excerpt_datafile), str(other_data), "-o", "output"],
    ):
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert f"{other_data} is missing required excerpt fields" in captured.err

    # should get the same error no matter what order we specify input files
    with patch(
        "sys.argv",
        ["merge_excerpts.py", str(other_data), str(excerpt_datafile), "-o", "output"],
    ):
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert f"{other_data} is missing required excerpt fields" in captured.err


def test_main_successful(capsys, tmp_path):
    # test a succesful run
    excerpt_datafile = tmp_path / "excerpts.csv"
    _excerpts_to_csv(excerpt_datafile, [excerpt1, excerpt2])
    # valid excerpt data
    labeled_excerpt_datafile = tmp_path / "excerpt_ids.csv"

    # copy excerpt1_label1 to confirm set output in csv
    # - everything the same except for the method (unlikely!)
    excerpt1_label1_method2 = replace(
        excerpt1_label1, identification_methods={"refmatcha"}
    )
    _excerpts_to_csv(
        labeled_excerpt_datafile, [excerpt1_label1, excerpt1_label1_method2]
    )

    output_file = tmp_path / "merged.csv"
    with patch(
        "sys.argv",
        [
            "merge_excerpts.py",
            str(excerpt_datafile),
            str(labeled_excerpt_datafile),
            "-o",
            str(output_file),
        ],
    ):
        main()
        captured = capsys.readouterr()

    # summary output
    assert "Loaded 4 excerpts from 2 files" in captured.out
    assert "2 total excerpts after merging; 1 labeled excerpts" in captured.out

    with output_file.open(encoding="utf-8") as merged_csv:
        csv_reader = csv.DictReader(merged_csv)
        merged_excerpts = list(iter(csv_reader))

    # row 1: excerpt 2 unchanged (no labels to combine)
    # NOTE: can't initialize as excerpt without removing unset label fields from csv
    merged_ex2 = LabeledExcerpt.from_dict(merged_excerpts[0])
    assert merged_ex2.excerpt_id == excerpt2.excerpt_id
    assert not merged_ex2.poem_id

    # row 2: excerpt 1 with merged labels
    # NOTE: can't initialize as excerpt without removing unset label fields from csv
    merged_ex1 = LabeledExcerpt.from_dict(merged_excerpts[1])
    assert merged_ex1.excerpt_id == excerpt1.excerpt_id
    # poem id and reference data preserved
    assert merged_ex1.poem_id == excerpt1_label1.poem_id
    assert merged_ex1.ref_span_start == excerpt1_label1.ref_span_start
    assert merged_ex1.ref_span_end == excerpt1_label1.ref_span_end
    assert merged_ex1.ref_span_text == excerpt1_label1.ref_span_text
    # id methods combined
    assert merged_ex1.identification_methods == {"manual", "refmatcha"}
