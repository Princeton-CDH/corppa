# Copyright (c) 2024-2026, Center for Digital Humanities, Princeton University
# SPDX-License-Identifier: Apache-2.0

import csv
from dataclasses import replace
from unittest.mock import patch

import polars as pl
import pytest
from test_polars_utils import _excerpts_to_csv

from corppa.poetry_detection.core import Excerpt, LabeledExcerpt, Span
from corppa.poetry_detection.merge_excerpts import (
    identify_overlapping_excerpts,
    main,
    merge_excerpts,
)
from corppa.poetry_detection.polars_utils import standardize_dataframe

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
    # excerpt + labeled excerpt (same id, same method, same span)
    df = pl.from_dicts([excerpt1.to_dict(), excerpt1_label1.to_dict()])
    merged = merge_excerpts(df)
    # expect one row (excerpts have been merged)
    assert len(merged) == 1
    # should have all columns for labeled excerpt (order-agnostic)
    assert set(merged.columns) == set(LabeledExcerpt.fieldnames())
    row = merged.row(0, named=True)
    merged_excerpt = LabeledExcerpt.from_dict(row)
    # existing notes should be present
    assert excerpt1_label1.notes in merged_excerpt.notes
    # merge info should be added to notes
    assert "merge: ppa exact span, 2 excerpts" in merged_excerpt.notes

    # result should exactly match the labeled excerpt since all fields are same
    # other than notes; override the notes to simplify the check
    excerpt1_label1_notes = LabeledExcerpt.from_excerpt(
        excerpt1_label1, notes=merged_excerpt.notes
    )
    assert merged_excerpt == excerpt1_label1_notes


def test_merge_excerpts_1ex_2labels(capsys):
    # excerpt + two labeled excerpt (same excerpt id, two different poem ids)
    df = pl.from_dicts(
        [excerpt1.to_dict(), excerpt1_label1.to_dict(), excerpt1_label2.to_dict()]
    )
    merged = merge_excerpts(df)
    # expect one row with combined labels
    assert len(merged) == 1
    merged_excerpt = LabeledExcerpt.from_dict(merged.row(0, named=True))
    # existing notes should be present
    for excerpt in [excerpt1, excerpt1_label1, excerpt1_label2]:
        if excerpt.notes:
            assert excerpt.notes in merged_excerpt.notes
    # merge info should be added to notes
    assert "merge: ppa exact span, 3 excerpts" in merged_excerpt.notes

    # id methods should be combined
    merged_excerpt.identification_methods == excerpt1_label1.identification_methods & excerpt1_label2.identification_methods
    # first poem id is selected as primary
    assert merged_excerpt.poem_id == excerpt1_label1.poem_id
    # alternate poem ids collected in a separate field
    assert merged_excerpt.alt_poem_ids == {excerpt1_label2.poem_id}

    # all other fields should be unchanged
    for field in LabeledExcerpt.fieldnames():
        # all other fields should have the same content in the merged excerpt
        if field not in ["notes", "poem_id", "identification_methods", "alt_poem_ids"]:
            assert getattr(merged_excerpt, field) == getattr(excerpt1_label1, field)


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
    # notes should be combined, and merge info should be added
    expected_merge_note = "merge: ppa exact span, 2 excerpts"
    expected_notes = "; ".join(
        [ex1_notes.notes, excerpt1_label1.notes, expected_merge_note]
    )
    assert merged_excerpt.notes == expected_notes
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


def test_merge_passim_match_len():
    # passim match length should take precedence over sort by poem id
    long_excerpt1 = replace(
        excerpt1_label1, poem_id="z", notes="passim: 442 char matches"
    )
    shorter_excerpt2 = replace(
        excerpt1_label1, poem_id="a", notes="passim: 213 char matches"
    )
    df = pl.from_dicts([shorter_excerpt2.to_dict(), long_excerpt1.to_dict()])
    merged = merge_excerpts(df)
    # expect one row
    assert len(merged) == 1
    # longer match should take precedence
    merged_excerpt = LabeledExcerpt.from_dict(merged.row(0, named=True))
    assert merged_excerpt.poem_id == long_excerpt1.poem_id


def test_merge_excerpts_multiple_diff_labels(capsys):
    # excerpt + two labeled excerpt (same excerpt id, two different ref ids)
    df = pl.from_dicts(
        [excerpt1.to_dict(), excerpt1_label1.to_dict(), excerpt1_label2.to_dict()]
    )
    # add the dataframe to itself so we have two of everything
    # = two labeled excerpts each for the two poem_ids in label 1 and label 2
    df = df.extend(df)
    merged = merge_excerpts(df)
    # expect one rows with combined poem id
    assert len(merged) == 1
    merged_excerpt = LabeledExcerpt.from_dict(merged.row(0, named=True))
    # notes should be combined, and merge info should be added
    expected_merge_note = "merge: ppa exact span, 6 excerpts"
    # excerpt1 has no notes
    expected_notes = "; ".join(
        [excerpt1_label1.notes, excerpt1_label2.notes, expected_merge_note]
    )
    assert merged_excerpt.notes == expected_notes

    # identification methods should be combined
    merged_excerpt.identification_methods == excerpt1_label1.identification_methods & excerpt1_label2.identification_methods

    # first poem id chosen as primary; others collected as alternate
    assert merged_excerpt.poem_id == excerpt1_label1.poem_id
    assert merged_excerpt.alt_poem_ids == {excerpt1_label2.poem_id}

    for field in LabeledExcerpt.fieldnames():
        # all other fields should have the same content in the merged excerpt
        if field not in ["notes", "poem_id", "identification_methods", "alt_poem_ids"]:
            assert getattr(merged_excerpt, field) == getattr(excerpt1_label1, field)


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
    # revised merge logic SHOULD merge labeled excerpts with different poem id
    alt_poem_id = "Z1234"
    excerpt1_diff_label = replace(excerpt1_label1, poem_id=alt_poem_id)
    df = pl.from_dicts([excerpt1_label1.to_dict(), excerpt1_diff_label.to_dict()])

    # distinct poem ids combined when span matches exactly
    merged = merge_excerpts(df)
    assert len(merged) == 1
    merged_result = merged.row(0, named=True)
    # based on current sort logic, alt poem id will be chosen as primary poem id
    assert merged_result["poem_id"] == alt_poem_id
    assert merged_result["alt_poem_ids"] == [excerpt1_label1.poem_id]


# revise to merge labeled + unlabeled excerpts
def test_merge_unlabeled_labeled_excerpts():
    # excerpt + one matching labeled excerpt
    df = pl.from_dicts([excerpt1.to_dict(), excerpt1_label1.to_dict()])
    merged = merge_excerpts(df)
    # we expect a single row
    assert len(merged) == 1
    excerpt = LabeledExcerpt.from_dict(merged.row(0, named=True))
    # merge info added to notes
    expected_merge_note = "merge: ppa exact span, 2 excerpts"
    assert expected_merge_note in excerpt.notes
    # should match the labeled excerpt, other than notes; everything else was the same
    assert replace(excerpt, notes=None) == replace(excerpt1_label1, notes=None)

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
    for note in [excerpt_with_notes.notes, excerpt1_label1.notes, expected_merge_note]:
        assert note in excerpt.notes

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
    # we expect one row with combined poem ids
    assert len(merged) == 1
    merged_excerpt = merged.row(0, named=True)
    expected_merge_note = "merge: ppa exact span, 3 excerpts"
    # notes contents from both merged excerpts should be present
    for note in [
        excerpt_with_notes.notes,
        excerpt1_label1.notes,
        excerpt1_label2.notes,
        expected_merge_note,
    ]:
        assert note in merged_excerpt["notes"]

    assert merged_excerpt["poem_id"] == excerpt1_label1.poem_id
    assert merged_excerpt["alt_poem_ids"] == [excerpt1_label2.poem_id]


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
    assert "Loaded 4 excerpts from 2 files (4 unique; 2 labeled)" in captured.out
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


### test for identify_overlapping_excerpts

# test scenarios that should result in no overlapping pairs
no_overlap_inputs = [
    # list of excerpts, reason this example has no overlapping pairs
    ([excerpt2], "single excerpt"),
    ([excerpt2, excerpt1_label1], "excerpts on different page"),
    # construct a second excerpt on the same page by using replace and relative offset span start/end
    (
        [
            excerpt1,
            replace(
                excerpt1,
                ppa_span_start=excerpt1.ppa_span_end + 100,
                ppa_span_end=excerpt1.ppa_span_end + 120,
            ),
        ],
        "same page, no overlap",
    ),
    # construct a short second excerpt on the same page with minimal overlap
    (
        [
            excerpt1,
            replace(
                excerpt1,
                ppa_span_start=excerpt1.ppa_span_end - 1,
                ppa_span_end=excerpt1.ppa_span_end + 3,
            ),
        ],
        "very small overlap",
    ),
]


@pytest.mark.parametrize("excerpts, reason", no_overlap_inputs)
def test_identify_overlapping_excerpts_no_pairs(excerpts, reason):
    # construct a standardized dataframe from the list of excerpts given
    excerpts_df = standardize_dataframe(
        pl.from_dicts([ex.to_dict() for ex in excerpts])
    )
    pairs_df = identify_overlapping_excerpts(excerpts_df)
    assert pairs_df.height == 0, f"expected 0 overlapping pairs: {reason}"


def test_identify_overlapping_excerpts():
    # create a pair with high overlap starting with fixture 1
    # for convenience, we use the existing Span object to construct
    # an overlapping span and check the overlap length / factor logic
    ppa_span1 = Span(start=excerpt1.ppa_span_start, end=excerpt1.ppa_span_end, label="")
    # create a second span; offset start by 1/9 the length of the first span
    ppa_span2 = Span(
        start=int(ppa_span1.start + len(ppa_span1) / 9), end=ppa_span1.end + 1, label=""
    )
    excerpt1_overlap = replace(
        excerpt1, ppa_span_start=ppa_span2.start, ppa_span_end=ppa_span2.end
    )
    # use existing span logic as coherence check for new method
    overlap_len = ppa_span1.overlap_length(ppa_span2)
    assert overlap_len >= 9
    overlap_factor = ppa_span1.overlap_factor(ppa_span2, ignore_label=True)
    assert overlap_factor >= 0.9
    # construct a standardized dataframe from the two test excerpts
    excerpts = [excerpt1, excerpt1_overlap]
    excerpts_df = standardize_dataframe(
        pl.from_dicts([ex.to_dict() for ex in excerpts])
    )
    pairs_df = identify_overlapping_excerpts(
        excerpts_df, min_overlap_chars=9, min_overlap_factor=0.9
    )
    # we expect one pair
    assert pairs_df.height == 1
    # inspect the fields in the one returned pair
    pair_result = pairs_df.row(0, named=True)
    assert pair_result["page_id"] == excerpt1.page_id
    # both excerpt ids present (order agnostic)
    pair_exc_ids = set([pair_result["excerpt_id"], pair_result["excerpt_id_right"]])
    assert pair_exc_ids == set([excerpt1.excerpt_id, excerpt1_overlap.excerpt_id])
    assert pair_result["overlap_len"] == overlap_len
    assert pair_result["overlap_factor"] == overlap_factor

    # confirm that if we adjust the parameters, this pair is not returned
    assert (
        identify_overlapping_excerpts(
            excerpts_df, min_overlap_chars=10, min_overlap_factor=0.9
        ).height
        == 0
    )
    assert (
        identify_overlapping_excerpts(
            excerpts_df, min_overlap_chars=9, min_overlap_factor=0.95
        ).height
        == 0
    )
    # defaults options exclude this pair
    assert identify_overlapping_excerpts(excerpts_df).height == 0
