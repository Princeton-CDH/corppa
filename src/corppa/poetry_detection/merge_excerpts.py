#!/usr/bin/env python
"""
This script merges detected poems excerpts (i.e. :class:~`corppa.poetry_detection.core.Excerpt`)
with identified poem excerpts (i.e. :class:~`corppa.poetry_detection.core.LabeledExcerpt`);
it also handles merging duplicate poem identifications in simple cases.

It takes two or more input files of excerpt or labeled excerpt data in CSV format,
merges the excerpts, and outputs a CSV of the merged excerpt data.  All excerpts
are included in the output, whether they were merged with any other records or not.
This means that in most cases, the output will likely be a mix of labeled
and unlabeled excerpts.

Merging logic is as follows:
- Excerpts are merged on the combination of page id and excerpt id
- When working with two sets of labeled excerpts, records are merged on the
  combination of page id, excerpt id, and poem id
    - If the same excerpt has different labels (different `poem_id` values), both
      labeled excerpts will be included in the output
    - If the same excerpt has duplicate labels (i.e., the same `poem_id` from two
      different identification methods), they will be merged
      into a single labeled excerpt; the `identification_methods` in the
      resulting labeled excerpt will be the union of methods in the merged excerpts
- When merging excerpts where both records have notes, the notes content
  will be combined. (Combined notes order follows input file order.)

After all input files are combined, the script checks for duplicate
excerpt idenfications that can be consolidated. This currently only handles
these simple cases:
- All poem identification and reference fields match (poem_id, ref_span_start, ref_span_text, ref_span_end)
- Poem identification matches and reference fields are null in one set
    (e.g. manual identification, which does not include reference fields, and
    refmatcha identification)

Limitations:
- Generally assumes excerpts do not require merging within a single input file
- Merging based on poem_id does not compare or consolidate reference span indices
  and text fields; supporting multiple identification methods that output
  span information will require revision
- CSV input and output only (JSONL may be added in future)

"""

import argparse
import pathlib
import sys

import polars as pl

from corppa.poetry_detection.core import MULTIVAL_DELIMITER, Excerpt, LabeledExcerpt
from corppa.poetry_detection.polars_utils import (
    FIELD_TYPES,
    LABELED_EXCERPT_FIELDS,
    REQ_LABELED_EXCERPT_FIELDS,
    has_poem_ids,
    load_excerpts_df,
    standardize_dataframe,
)


def combine_excerpts(df: pl.DataFrame, other_df: pl.DataFrame) -> pl.DataFrame:
    """Combine two Polars dataframes with excerpt or labeled excerpt data.
    Excerpts are joined on the combination of page id and excerpt id.
    All excerpts from both dataframes are included in the resulting dataframe.
    Excerpts are combined as follows:
    - an unlabeled excerpt and a labeled excerpt for the same excerpt
      will be combined
    - if combined excerpts both have content in the notes, the notes text
      will be combined
    - multiple labeled excerpts for the same excerpt id are NOT combined
    """
    # simplest option is to do a LEFT join on page id and excerpt id
    join_fields = ["page_id", "excerpt_id"]

    # if poem_id is present and not empty in both dataframes,
    # include that in the join fields to avoid collapsing different ids
    if has_poem_ids(df) and has_poem_ids(other_df):
        # NOTE: for now, the script does not care about variations between
        # reference span start, end, and text if the poem identifications match
        # That assumption is valid for the current set, since manual ids
        # do not have spans, but we may need to revisit in future
        join_fields.append("poem_id")

    # before joining, drop redundant fields that will be the same
    # on any excerpt with matching page & excerpt id
    other_join = other_df.drop(
        "detection_methods", "ppa_span_start", "ppa_span_end", "ppa_span_text"
    )
    merged = df.join(other_join, on=join_fields, how="left")

    # if notes_right is present, then we have notes coming from both sides
    # of the join; combine the notes into a single notes field
    if "notes_right" in merged.columns:
        # update notes field by combining left and right notes with a newline,
        # and then strip any outer newlines
        merged = merged.with_columns(
            notes=pl.col("notes")
            .str.strip_chars()
            .add(pl.lit("\n"))
            .add(pl.col("notes_right").str.strip_chars())
            .str.strip_chars("\n")
        ).drop("notes_right")

    if "identification_methods_right" in merged.columns:
        # use list set union method to merge values, ignoring nulls
        # - if left value is null, use right side
        # - if right value is null, use left
        # - if both are non-null, combine
        # NOTE: null check is required to avoid null + value turning into a null
        # although there may be a more elegant polars way to handle this
        merged = merged.with_columns(
            identification_methods=pl.when(pl.col("identification_methods").is_null())
            .then(pl.col("identification_methods_right"))
            .when(pl.col("identification_methods_right").is_null())
            .then(pl.col("identification_methods"))
            .otherwise(
                pl.col("identification_methods").list.set_union(
                    pl.col("identification_methods_right")
                )
            )
        ).drop("identification_methods_right")

    # the left join omits any excerpts in other_df that are not in the main df
    # use an "anti" join starting with the other df to get all the rows
    # in other_df that are not present in the first df
    right_df = other_df.join(df, on=join_fields, how="anti")
    if not right_df.is_empty():
        # ensure field order and types match, then append the
        # excerpts from the right dataframe to the end of the merged dataframe
        merged = standardize_dataframe(merged)
        merged = merged.select(LABELED_EXCERPT_FIELDS).extend(
            standardize_dataframe(right_df)
        )

    return merged


def merge_labeled_excerpts(df: pl.DataFrame) -> pl.DataFrame:
    """Takes a polars Dataframe that includes labeled excerpts and attempts to
    merges excerpts with same page id, excerpt id, and poem id. Returns the resulting
    dataframe, with any duplicate excerpts merged.
    For now, merging is only done on the simple cases where reference
    fields match exactly, or where reference fields are present in one labeled
    excerpt and null in the other.
    """

    # copy the df and add a row index for removal
    updated_df = df.with_row_index()
    # create a df with the same schema but no data to collect merged excerpts
    merged_excerpts = updated_df.clear()

    # group by page id, excerpt id, and poem id to find repeated identificatins
    for group, data in updated_df.group_by(["page_id", "excerpt_id", "poem_id"]):
        # group is a tuple of values for page id, excerpt id, poem id
        # data is a df of the grouped rows for this set

        # sort so any empty values for optional fields are last,
        # then fill values forward - i.e., treat nulls as duplicates
        data = data.sort(
            "ref_span_start", "ref_span_end", "ref_span_text", nulls_last=True
        ).select(pl.all().forward_fill())

        # identify repeats where reference values all agree
        # (either same values or don't conflict because unset)
        repeats = data.filter(
            data.drop("identification_methods", "index").is_duplicated()
        )

        if not repeats.is_empty():
            # convert list of id methods to string in each row, then combine all rows
            # TODO: switch to list set union methods here
            repeats = (
                repeats.with_columns(
                    # convert list of ids in each row to string
                    id_meth=pl.col("identification_methods").list.join(",")
                )
                # combine all the ids across row as a string
                .with_columns(combined_id_string=pl.col("id_meth").str.join(","))
                # split again to convert to list format
                .with_columns(
                    identification_methods=pl.col("combined_id_string").str.split(",")
                )
                # drop the interim fields
                .drop("id_meth", "combined_id_string")
            )
            # remove the repeats from the main dataframe
            updated_df = updated_df.filter(
                ~pl.col("index").is_in(repeats.select(pl.col("index")))
            )
            # add the consolidated row to the merged df
            merged_excerpts.extend(repeats[:1])

    # combine and return
    return updated_df.extend(merged_excerpts).drop("index")


def main():
    parser = argparse.ArgumentParser(
        description="Merge excerpts with labeled excerpts or notes"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output filename for merged excerpts (CSV)",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Two or more input files with excerpt or labeled excerpt data",
        type=pathlib.Path,
    )

    args = parser.parse_args()
    # output file should not exist
    if args.output.exists():
        print(
            f"Error: output file {args.output} already exists, not overwriting",
            file=sys.stderr,
        )
        sys.exit(-1)
    # we need at least two input files
    if len(args.input_files) < 2:
        print(
            "Error: at least two input files are required for merging", file=sys.stderr
        )
        sys.exit(-1)

    # make sure input files exist
    non_existent_input = [f for f in args.input_files if not f.exists()]
    if non_existent_input:
        print(
            f"Error: input files not found: {', '.join([str(f) for f in non_existent_input])}",
            file=sys.stderr,
        )
        sys.exit(-1)

    total_excerpts = 0
    excerpts = None

    # load files in order specified, and merge them in one by one
    for input_file in args.input_files:
        try:
            merge_df = load_excerpts_df(input_file)
        except ValueError as err:
            # if any input file does not have minimum required fields, bail out
            print(err, file=sys.stderr)
            sys.exit(-1)
        total_excerpts += len(merge_df)

        # on the first loop, main excerpts df is unset, nothing to merge
        if excerpts is None:
            excerpts = merge_df
        else:
            # on every loop after the first, update excerpts by
            # merging with the new input file
            excerpts = combine_excerpts(excerpts, merge_df)

    excerpts = merge_labeled_excerpts(excerpts)

    # write the merged data to the requested output file
    # (in future, support multiple formats - at least csv/jsonl)

    # convert list fields for output to csv and reporting
    excerpts = excerpts.with_columns(
        detection_methods=pl.col("detection_methods")
        .list.sort()
        .list.join(MULTIVAL_DELIMITER),
        identification_methods=pl.col("identification_methods")
        .list.sort()
        .list.join(MULTIVAL_DELIMITER),
    )

    labeled_excerpts = excerpts.filter(pl.col("poem_id").is_not_null())

    # summary information about the content and what as done
    print(
        f"""Loaded {total_excerpts:,} excerpts from {len(args.input_files)} files.
{len(excerpts):,} total excerpts after merging; {len(labeled_excerpts):,} labeled excerpts. """
    )
    detectmethod_counts = excerpts["detection_methods"].value_counts()
    idmethod_counts = labeled_excerpts["identification_methods"].value_counts()
    print("Total by detection method:")
    for row in detectmethod_counts.iter_rows():
        # row is a tuple of value, count
        print(f"\t{row[0]}: {row[1]:,}")
    print("Total by identification method:")
    for row in idmethod_counts.iter_rows():
        # row is a tuple of value, count
        print(f"\t{row[0]}: {row[1]:,}")

    excerpts.write_csv(args.output)


if __name__ == "__main__":
    main()
