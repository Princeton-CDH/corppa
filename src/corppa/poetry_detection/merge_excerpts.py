#!/usr/bin/env python

import argparse
import pathlib
import sys

import polars as pl

from corppa.poetry_detection.core import Excerpt, LabeledExcerpt

EXCERPT_FIELDS = Excerpt.fieldnames()
LABELED_EXCERPT_FIELDS = LabeledExcerpt.fieldnames()
LABEL_ONLY_FIELDS = set(LABELED_EXCERPT_FIELDS) - set(EXCERPT_FIELDS)


FIELD_TYPES = LabeledExcerpt.field_types()


def excerpts_df(input_file: pathlib.Path) -> pl.DataFrame:
    """Load the specified input file as a Polars dataframe,
    with column names based on fields in
    :class:`~corppa.poetry_detection.core.LaebledExcerpt`."""
    # load input file as a polars dataframe
    # excerpt fields are a subset of labeled excerpt, so load as labeled
    # for now assume csv; in future may add support for jsonl

    # TODO: requiring all fields means adjudication data poem ids are ignored
    # what is the minimum set we need to check for?
    return pl.read_csv(input_file)  # , columns=LABELED_EXCERPT_FIELDS)

    try:
        return pl.read_csv(input_file, columns=LABELED_EXCERPT_FIELDS)
    except pl.exceptions.ColumnNotFoundError as err:
        print(err)
        # if label fields are missing, load as unlabeled excerpts
        return pl.read_csv(input_file, columns=EXCERPT_FIELDS)


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
    # if any columns are missing, add them with the appropriate type
    if missing_columns:
        df = df.with_columns(
            [
                pl.lit(None).alias(field).cast(FIELD_TYPES[field])
                for field in missing_columns
            ]
        )

    # set consistent order
    return df.select(LABELED_EXCERPT_FIELDS)


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

    # smplest option is to do a LEFT join on page id and excerpt id
    join_fields = ["page_id", "excerpt_id"]
    # before joining, drop redundant fields that will be the same
    # on any excerpt with matching page & excerpt id
    other_join = other_df.drop(
        "detection_methods", "ppa_span_start", "ppa_span_end", "ppa_span_text"
    )
    merged = df.join(other_join, on=join_fields, how="left")

    # NOTE: could also have right columns for poem_id_right, ref_corpus_right
    # if they are the same, merge; otherwise... split into two rows

    # if notes_right is present, then we have notes coming from both sides
    # of the join; combine the notes into a single notes field
    if "notes_right" in merged.columns:
        # update notes field by combining left and right notes with a newline,
        # and then strip any outer newlines
        merged = merged.with_columns(
            notes=pl.col("notes")
            .add(pl.lit("\n"))
            .add(pl.col("notes_right"))
            .str.strip_chars("\n")
        ).drop("notes_right")

    if "identification_methods_right" in merged.columns:
        merged = merged.with_columns(
            identification_methods=pl.when(
                pl.col("identification_methods_right").is_null()
            )
            .then(pl.col("identification_methods"))
            .when(pl.col("identification_methods").is_null())
            .then(pl.col("identification_methods_right"))
            .otherwise(
                pl.concat_list(
                    pl.col("identification_methods"),
                    pl.col("identification_methods_right"),
                ).sort()
            )
        ).drop("identification_methods_right")

    # the left join omits any excerpts in other_df that are not in the main df
    # use an "anti" join starting with the other df to get all the rows
    # in other_df that are not present in the first df
    right_df = other_df.join(df, on=join_fields, how="anti")
    if not right_df.is_empty():
        # ensure field order is exactly the same, then append the
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

        # TODO: if id is same but ref start/span/text differs, should we merge?
        # (probably not in this round)

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
        merge_df = excerpts_df(input_file)
        total_excerpts += len(merge_df)
        excerpts = combine_excerpts(excerpts, merge_df)

    excerpts = merge_duplicate_ids(excerpts)
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

    # write the merged data to the requested output file
    # (in future, support multiple formats - at least csv/jsonl)
    excerpts.write_csv(args.output)


if __name__ == "__main__":
    main()
