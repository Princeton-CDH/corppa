from dataclasses import replace

import polars as pl

from corppa.poetry_detection.core import Excerpt, LabeledExcerpt
from corppa.poetry_detection.merge_excerpts import merge_excerpts

#
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


def test_merge_excerpts():
    # excerpt + labeled excerpt (same id)
    df = pl.from_dicts([excerpt1.to_dict()])
    other_df = pl.from_dicts([excerpt1_label1.to_dict()])
    merged = merge_excerpts(df, other_df)
    # expect one row
    assert len(merged) == 1
    # should have all columns for labeled excerpt (order-agnostic)
    assert set(merged.columns) == set(LabeledExcerpt.fieldnames())
    row = merged.row(0, named=True)
    merged_excerpt = LabeledExcerpt.from_dict(row)
    # result should exactly match the labeled excerpt since all other fields are same
    assert merged_excerpt == excerpt1_label1

    # excerpt + two labeled excerpt (same excerpt id, two different ref ids)
    other_df = pl.from_dicts([excerpt1_label1.to_dict(), excerpt1_label2.to_dict()])
    merged = merge_excerpts(df, other_df)
    # expect two rows with  two different labels
    assert len(merged) == 2
    # polars preserves order, so we can check that they match what was sent in
    # results should exactly match the labeled excerpts since all other fields are same
    assert LabeledExcerpt.from_dict(merged.row(0, named=True)) == excerpt1_label1
    assert LabeledExcerpt.from_dict(merged.row(1, named=True)) == excerpt1_label2

    # excerpt with note + labeled excerpt (same id)
    ex1_notes = replace(excerpt1, notes="detection information")
    df = pl.from_dicts([ex1_notes.to_dict()])
    other_df = pl.from_dicts([excerpt1_label1.to_dict()])
    merged = merge_excerpts(df, other_df)
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
