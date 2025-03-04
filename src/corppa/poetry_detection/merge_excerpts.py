#!/usr/bin/env python
"""
This script merges detected poems excerpts (i.e. :class:~`corppa.poetry_detection.core.Excerpt`)
with identified poem excerpts (i.e. :class:~`corppa.poetry_detection.core.LabeledExcerpt`);
it also handles merging duplicate poem identifications in simple cases.

It takes two or more input files of excerpt or labeled excerpt data in CSV format,
merges the excerpts, and outputs a CSV of the merged excerpt data.  All excerpts
are included in the output, whether they were merged with any other records or not.

Merging logic is as follows:
- Excerpts are merged on the combination of page id and excerpt id
- When working with two sets of labeled excerpts, records are merged on the
  combination of page id, excerpt id, and poem id
    - If the same excerpt has different identifications, both
      labeled excerpts will be included in the output
    - If the same excerpt has duplicate identifications, they will be merged
      into a single excerpt that includes both identification methods
- When merging excerpts where both records have notes, the notes content
  will be combined

After all input files are combined, the script checks for duplicate
excerpt idenfications that can be consolidated. This currently only handles
these simple cases:
- All poem identification and reference fields match (poem_id, ref_span_start, ref_span_text, ref_span_end)
- Poem identification matches and reference fields are null in one set
    (e.g. manual identification and refmatcha identification)

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

#: List of required fields for excerpt data
REQ_EXCERPT_FIELDS = set(Excerpt.fieldnames(required_only=True))
#: List required fields for labeled excerpt data
REQ_LABELED_EXCERPT_FIELDS = set(LabeledExcerpt.fieldnames(required_only=True))
#: All fields for labeled excerpts, in the expected order
LABELED_EXCERPT_FIELDS = LabeledExcerpt.fieldnames()

#: dictionary of excerpt field names with associated data types
FIELD_TYPES = LabeledExcerpt.field_types()
# override set types with list, since Polars does not have a set type
FIELD_TYPES["detection_methods"] = pl.List
FIELD_TYPES["identification_methods"] = pl.List


def excerpts_df(input_file: pathlib.Path) -> pl.DataFrame:
    """Load the specified input file as a Polars dataframe,
    with column names based on fields in
    :class:`~corppa.poetry_detection.core.LaebledExcerpt`."""
    # load input file as a polars dataframe
    # for now assume csv; in future may add support for jsonl
    df = pl.read_csv(input_file)
    # check that we have the required fields for either
    # excerpt or labeled excerpt data
    columns = set(df.columns)
    expected_type = "excerpt"
    missing_columns = []
    # treat presence of poem ids as indication of labeled excerpt
    if has_poem_ids(df):
        expected_type = "labeled excerpt"
        missing_columns = REQ_LABELED_EXCERPT_FIELDS - columns
    else:
        missing_columns = REQ_EXCERPT_FIELDS - columns

    if missing_columns:
        raise ValueError(
            f"Input file {input_file} is missing required {expected_type} fields: {', '.join(missing_columns)}"
        )

    # set the correct data types for excerpt fields before returning
    return fix_data_types(df)


def fix_data_types(df):
    """Return a modified polars DataFrame with excerpt or labeled excerpt data
    with the appropriate data types. In particular, this handles converting
    multivalue method fields into Polars lists.
    """

    # get expected field types for columns that match excerpt / label excerpt fields
    df_types = {column: FIELD_TYPES.get(column) for column in df.columns}
    for c, ctype in df_types.items():
        # for list (set) types, split strings on multival delimiter to convert to list
        if ctype is pl.List:
            # only split if column is currently a string
            if df.schema[c] == pl.String:
                df = df.with_columns(pl.col(c).str.split(MULTIVAL_DELIMITER))
        # for any other field type, cast the column to the expected type
        elif ctype is not None:
            df = df.with_columns(pl.col(c).cast(ctype))

    return df


def fix_columns(df):
    """Ensure a polars dataframe has all expected columns for
    fields in :class:~`corppa.poetry_detection.core.LabeledExcerpt`,
    in the expected order, so that dataframes can be combined consistently.
    Any fields not present will be added as a series of null values
    with the appropriate type.
    """
    df_columns = set(df.columns)
    expected_columns = set(LABELED_EXCERPT_FIELDS)
    missing_columns = expected_columns - df_columns
    # if any columns are missing, add them and make sure types are correct
    if missing_columns:
        df = df.with_columns([pl.lit(None).alias(field) for field in missing_columns])
        df = fix_data_types(df)

    # set consistent order to allow extending/appending
    return df.select(LABELED_EXCERPT_FIELDS)


def has_poem_ids(df: pl.DataFrame) -> bool:
    """Check if a Polars DataFrame has poem_id values. Returns true if 'poem_id'
    is present in the list of columns and there is at least one non-null value."""
    return bool("poem_id" in df.columns and df["poem_id"].count())


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
        merged = fix_columns(merged)
        merged = merged.select(LABELED_EXCERPT_FIELDS).extend(fix_columns(right_df))

    return merged


def merge_duplicate_ids(df):
    # look for multiple rows for the same excerpt id and poem id,
    # try to merge them (only handles simple cases for now)

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
            # TODO: standardize on delimited string or list when loading/serializing
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
        description="Merge excerpts with identified excerpts or notes"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output filename for merged excerpts",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Two or more input files with excerpt or labeled except data",
        type=pathlib.Path,
    )

    args = parser.parse_args()
    # output file should not exist
    if args.output.exists():
        print(f"Error: output file {args.output} already exists, not overwriting")
        sys.exit(-1)
    # we need at least two input files
    if len(args.input_files) < 2:
        print("Error: at least two input files are required for merging")
        sys.exit(-1)

    # load the first input file into a polars dataframe
    # content is either excerpt or labeled excerpt
    excerpts = excerpts_df(args.input_files[0])
    total_excerpts = len(excerpts)

    # starting with the second input file, merge into the main excerpt
    for input_file in args.input_files[1:]:
        try:
            merge_df = excerpts_df(input_file)
        except ValueError as err:
            # if any input file does not have minimum required fields, bail out
            print(err)
            sys.exit(-1)
        total_excerpts += len(merge_df)
        excerpts = combine_excerpts(excerpts, merge_df)

    excerpts = merge_duplicate_ids(excerpts)

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
