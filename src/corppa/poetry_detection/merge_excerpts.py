#!/usr/bin/env python

import argparse
import pathlib
import sys

import polars as pl

from corppa.poetry_detection.core import Excerpt, LabeledExcerpt

EXCERPT_FIELDS = Excerpt.fieldnames()
LABELED_EXCERPT_FIELDS = LabeledExcerpt.fieldnames()
LABEL_ONLY_FIELDS = set(LABELED_EXCERPT_FIELDS) - set(EXCERPT_FIELDS)
print(LABEL_ONLY_FIELDS)

LABEL_ONLY_FIELD_TYPES = {
    "ref_span_end": int,
    "poem_id": str,
    "ref_corpus": str,
    "ref_span_text": str,
    "ref_span_start": int,
    "identification_methods": str,
}
# .cast(pl.Float64)
# field_types =


def excerpts_df(input_file: pathlib.Path) -> pl.DataFrame:
    """Load the specified input file as a Polars dataframe,
    with column names based on fields in
    :class:`~corppa.poetry_detection.core.LaebledExcerpt`."""
    # load input file as a polars dataframe
    # excerpt fields are a subset of labeled excerpt, so load as labeled
    # for now assume csv; in future may add support for jsonl
    try:
        return pl.read_csv(input_file, columns=LABELED_EXCERPT_FIELDS)
    except pl.exceptions.ColumnNotFoundError:
        # for merging, we need the same columns everywhere
        # add label-only excerpt fields with null values
        # return pl.read_csv(input_file, columns=EXCERPT_FIELDS)

        return (
            pl.read_csv(input_file, columns=EXCERPT_FIELDS)
            .with_columns(
                [
                    pl.lit(None).alias(field).cast(LABEL_ONLY_FIELD_TYPES[field])
                    for field in LABEL_ONLY_FIELDS
                ]
            )
            .select(LABELED_EXCERPT_FIELDS)
        )


def merge_excerpts(df, other_df):
    print(f"merging excerpt dataframes: {len(df)} rows + {len(other_df)} rows")

    # do a left (full?) join based on page_id + excerpt_id
    other_df = other_df.drop(
        "detection_methods", "ppa_span_start", "ppa_span_end", "ppa_span_text"
    )
    merged = df.join(other_df, on=["page_id", "excerpt_id"], how="left")
    # if notes_right is present, then notes need to be merged
    if "notes_right" in merged.columns:
        # create a new notes field combining left and right notes with a newline,
        # and then strip any outer newlines
        merged = merged.with_columns(
            notes=pl.col("notes")
            .add(pl.lit("\n"))
            .add(pl.col("notes_right"))
            .str.strip_chars("\n")
        ).drop("notes_right")

    return merged

    print(f"merged dataframes: {len(merged)} rows")
    print(merged.columns)
    print(merged.head(10))

    return
    print("df columns:")
    print(df.columns)
    print("other columns:")
    print(other_df.columns)

    # extend one dataframe with the next
    extended = df.extend(other_df)
    print(f"extended dataframes: {len(extended)} rows")
    grouped = extended.group_by(pl.col("page_id"), pl.col("excerpt_id")).all()
    print(grouped)


#     pl.col("poem_id"),
#     pl.col("notes"),
#     # [pl.col(field) for field in LABELED_EXCERPT_FIELDS]
# )
# print(grouped.head(10))


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
    print(args)
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
    # starting with the second input file, merge into the main excerpt
    for input_file in args.input_files[1:]:
        merge_df = excerpts_df(input_file)
        excerpts = merge_excerpts(excerpts, merge_df)

    # write the merged data to the requested output file
    # (in future, support multiple formats - at least csv/jsonl)
    # excerpts.write_csv(args.output)


if __name__ == "__main__":
    main()
