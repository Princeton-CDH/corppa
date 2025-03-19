import csv
from dataclasses import replace
from unittest.mock import patch

import polars as pl
import pytest
from test_polars_utils import _excerpts_to_csv

from corppa.poetry_detection.core import Excerpt, LabeledExcerpt
from corppa.poetry_detection.merge_excerpts import (
    main,
    merge_excerpts,
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


def test_merge_excerpts_1ex_1label():
    # excerpt + labeled excerpt (same id)
    df = pl.from_dicts([excerpt1.to_dict(), excerpt1_label1.to_dict()])
    merged = merge_excerpts(df)
    # expect one row
    assert len(merged) == 1
    # should have all columns for labeled excerpt (order-agnostic)
    assert set(merged.columns) == set(LabeledExcerpt.fieldnames())
    row = merged.row(0, named=True)
    merged_excerpt = LabeledExcerpt.from_dict(row)
    # result should exactly match the labeled excerpt since all other fields are same
    assert merged_excerpt == excerpt1_label1


def test_merge_excerpts_1ex_2labels(capsys):
    # excerpt + two labeled excerpt (same excerpt id, two different ref ids)
    df = pl.from_dicts(
        [excerpt1.to_dict(), excerpt1_label1.to_dict(), excerpt1_label2.to_dict()]
    )
    merged = merge_excerpts(df)
    # expect two rows with two different labels
    assert len(merged) == 2
    # original order is not guaranteed, so check presence in list
    result_excerpts = [
        LabeledExcerpt.from_dict(row) for row in merged.iter_rows(named=True)
    ]
    # results should exactly match the labeled excerpts since all other fields are same
    # input excerpts should both be present unchanged in the output
    assert excerpt1_label1 in result_excerpts
    assert excerpt1_label2 in result_excerpts


def test_merge_excerpts_1ex_note_1label():
    # excerpt with note + labeled excerpt (same id)
    ex1_notes = replace(excerpt1, notes="detection information")
    df = pl.from_dicts([ex1_notes.to_dict(), excerpt1_label1.to_dict()])
    merged = merge_excerpts(df)
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


def test_merge_excerpts_1ex_different_label():
    # excerpt 2 + labeled excerpt 1 - should preserve both
    df = pl.from_dicts([excerpt2.to_dict(), excerpt1_label1.to_dict()])
    merged = merge_excerpts(df)
    # expect two rows
    assert len(merged) == 2
    # should have all columns for labeled excerpt (order-agnostic)
    assert set(merged.columns) == set(LabeledExcerpt.fieldnames())
    # the row with no poem_id is the unlabeled excerpt
    row = merged.filter(pl.col("poem_id").is_null()).row(0, named=True)
    # filter out null values (unset labeled excerpt fields) and init as Excerpt
    row_subset = {k: v for k, v in row.items() if v is not None}
    merged_excerpt2 = Excerpt.from_dict(row_subset)
    assert merged_excerpt2 == excerpt2
    # row with a poem_id set is the labeled excerpt
    row = merged.filter(pl.col("poem_id").is_not_null()).row(0, named=True)
    merged_excerpt1_label1 = LabeledExcerpt.from_dict(row)
    assert merged_excerpt1_label1 == excerpt1_label1


def test_merge_excerpts_two_different_labels():
    # two different labeled excerpts should not be merged
    assert excerpt1_label1.excerpt_id != excerpt2_label1.excerpt_id
    df = pl.from_dicts([excerpt1_label1.to_dict(), excerpt2_label1.to_dict()])
    merged = merge_excerpts(df)
    # expect two rows
    assert len(merged) == 2
    # should have all columns for labeled excerpt (order-agnostic)
    assert set(merged.columns) == set(LabeledExcerpt.fieldnames())
    # order is not guaranteed to match output, so check for presence
    result_excerpts = [
        LabeledExcerpt.from_dict(row) for row in merged.iter_rows(named=True)
    ]
    # input excerpts should both be present unchanged in the output
    assert excerpt1_label1 in result_excerpts
    assert excerpt2_label1 in result_excerpts


def test_merge_excerpts_multiple_diff_labels(capsys):
    # excerpt + two labeled excerpt (same excerpt id, two different ref ids)
    df = pl.from_dicts(
        [excerpt1.to_dict(), excerpt1_label1.to_dict(), excerpt1_label2.to_dict()]
    )
    # add the dataframe to itself so we have two of everything
    # = two labeled excerpts each for the two poem_ids in label 1 and label 2
    df = df.extend(df)
    merged = merge_excerpts(df)
    # expect two rows with two different labels
    assert len(merged) == 2
    # order is not guaranteed to match output, so check for presence in output
    result_excerpts = [
        LabeledExcerpt.from_dict(row) for row in merged.iter_rows(named=True)
    ]
    # input excerpts should both be present unchanged in the output
    assert excerpt1_label1 in result_excerpts
    assert excerpt1_label2 in result_excerpts


def test_merge_excerpts_1ex_2labels_diffmethod():
    # unlabeled excerpt + two matching labeled excerpts
    # - same excerpt id, two labels with same ref ids but different method
    # combine method does not merge these

    # everything the same except for the method (unlikely!)
    excerpt1_label1_method2 = replace(
        excerpt1_label1, identification_methods={"refmatcha"}
    )
    df = pl.from_dicts(
        [
            excerpt1.to_dict(),
            excerpt1_label1.to_dict(),
            excerpt1_label1_method2.to_dict(),
        ]
    )
    merged = merge_excerpts(df)
    assert len(merged) == 1


def test_merge_different_labels():
    # combine should NOT merge labeled excerpts with different poem id
    excerpt1_diff_label = replace(excerpt1_label1, poem_id="Z1234")
    df = pl.from_dicts([excerpt1_label1.to_dict(), excerpt1_diff_label.to_dict()])

    # distinct poem ids should NOT be merged
    merged = merge_excerpts(df)
    assert len(merged) == 2


# revise to merge labeled + unlabeled excerpts
def test_merge_unlabeled_labeled_excerpts():
    # excerpt + one matching labeled excerpt
    df = pl.from_dicts([excerpt1.to_dict(), excerpt1_label1.to_dict()])
    merged = merge_excerpts(df)
    # we expect a single row
    assert len(merged) == 1
    excerpt = LabeledExcerpt.from_dict(merged.row(0, named=True))
    # should match the labeled excerpt, since everything else was the same
    assert excerpt == excerpt1_label1

    # excerpt + excerpt with notes
    excerpt_with_notes = replace(excerpt1, notes="could not identify")
    df = pl.from_dicts([excerpt_with_notes.to_dict(), excerpt1_label1.to_dict()])
    merged = merge_excerpts(df)
    # we expect a single row
    assert len(merged) == 1
    excerpt = LabeledExcerpt.from_dict(merged.row(0, named=True))
    # should not match the labeled excerpt, since notes should be combined
    assert excerpt != excerpt1_label1
    # notes contents from both merged excerpts should be present
    assert excerpt_with_notes.notes in excerpt.notes
    assert excerpt1_label1.notes in excerpt.notes
    assert excerpt.notes == f"{excerpt_with_notes.notes}\n{excerpt1_label1.notes}"

    # excerpt with notes and two labeled excerpts that can't be merged
    # - notes are merged to the first matching labeled excerpt
    excerpt_with_notes = replace(excerpt1, notes="could not identify")
    df = pl.from_dicts(
        [
            excerpt_with_notes.to_dict(),
            excerpt1_label1.to_dict(),
            excerpt1_label2.to_dict(),
        ]
    )
    merged = merge_excerpts(df)
    # we expect two rows
    assert len(merged) == 2
    # order is not guaranteed; test against a list of merged note contents
    merged_notes = merged["notes"].to_list()
    # unlabeled excerpt and excerpt1 label 1 are combined
    assert f"{excerpt_with_notes.notes}\n{excerpt1_label1.notes}" in merged_notes
    # second labeled excerpt does not currently get unlabeled excerpt notes
    assert excerpt1_label2.notes in merged_notes


def test_merge_excerpts():
    # excerpt + two matching labeled excerpts
    # - same excerpt id, two labels with same ref ids but different method

    # everything the same except for the method (unlikely!)
    excerpt1_label1_method2 = replace(
        excerpt1_label1, identification_methods={"refmatcha"}
    )
    df = pl.from_dicts([excerpt1_label1.to_dict(), excerpt1_label1_method2.to_dict()])
    merged = merge_excerpts(df)
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
    df = pl.from_dicts(
        [excerpt1.to_dict(), excerpt1_label1.to_dict(), excerpt1_label1_other.to_dict()]
    )
    merged = merge_excerpts(df)
    assert len(merged) == 1
    # should have all columns for labeled excerpt (order-agnostic)
    assert set(merged.columns) == set(LabeledExcerpt.fieldnames())
    # should have both methods; order doesn't matter (and may not be reliable)
    assert set(merged.row(0, named=True)["identification_methods"]) == set(
        ["manual", "other"]
    )
    excerpt = LabeledExcerpt.from_dict(merged.row(0, named=True))
    assert excerpt.identification_methods == {"manual", "other"}

    # order should not matter
    df = pl.from_dicts([excerpt1_label1_other.to_dict(), excerpt1_label1.to_dict()])
    merged = merge_excerpts(df)
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
    assert "Loaded 4 excerpts from 2 files (2 labeled)" in captured.out
    assert "2 excerpts after merging; 1 labeled excerpts" in captured.out

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
