#!/usr/bin/env python
"""
This script merges detected poems excerpts (:class:`~corppa.poetry_detection.core.Excerpt`)
with identified poem excerpts (:class:`~corppa.poetry_detection.core.LabeledExcerpt`);
it also handles merging duplicate poem identifications in simple cases.

It takes two or more input files of excerpt or labeled excerpt data in CSV format,
merges the excerpts, and outputs a CSV of the merged excerpt data.  All excerpts
are included in the output, whether they were merged with any other records or not.
This means that in most cases, the output will likely be a mix of labeled
and unlabeled excerpts.

Merging logic is as follows:

- Excerpts are grouped on the combination of page id and excerpt id and then merged if all reference fields match exactly, or where reference fields are present in one excerpt and null in the other.
    - If the same excerpt has different labels (different `poem_id` values), both
      labeled excerpts will be included in the output
    - If the same excerpt has duplicate labels (i.e., the same `poem_id` from two
      different identification methods), they will be merged
      into a single labeled excerpt; the `identification_methods` in the
      resulting labeled excerpt will be the union of methods in the merged excerpts
- When merging excerpts where both records have notes, the notes content
  will be combined.

Example usage:

``./src/corppa/poetry_detection/merge_excerpts.py adjudication_excerpts.csv labeled_excerpts.csv -o merged_excerpts.csv``

Limitations:

- Merging based on poem_id does not compare or consolidate reference span indices
  and text fields; supporting multiple identification methods that output
  span information will require revision
- CSV input and output only (JSONL may be added in future)

"""

import argparse
import itertools
import pathlib
import sys

import polars as pl

from corppa.poetry_detection.core import MULTIVAL_DELIMITER
from corppa.poetry_detection.polars_utils import load_excerpts_df, standardize_dataframe


def combine_duplicate_methods_notes(repeats_df: pl.DataFrame) -> pl.DataFrame:
    """Takes a dataframe of repeated excerpts with duplicate information,
    and combines detection_methods, identification_methods, and notes.
    Returns the updated dataframe with the combined fields.
    """
    # get id methods as a list of lists, use itertools to unwrap
    # the lists; consume the itertools generator and use set to uniquify,
    # then convert back to list to put back in the polars dataframe
    detect_methods = repeats_df["detection_methods"].drop_nulls().to_list()
    combined_detect_methods = list(
        set(list(itertools.chain.from_iterable(detect_methods)))
    )
    # id methods could be all unset even in a group
    id_methods = repeats_df["identification_methods"].drop_nulls().to_list()
    combined_id_methods = None
    if id_methods:
        combined_id_methods = list(set(list(itertools.chain.from_iterable(id_methods))))
    # join all unique notes within this group; don't repeat notes
    # preserve order (unlabeled excerpt notes first)
    unique_notes = repeats_df["notes"].drop_nulls().unique(maintain_order=True)
    combined_notes = "\n".join([n for n in unique_notes if n.strip()])

    repeats_df = repeats_df.with_columns(
        detection_methods=pl.lit(combined_detect_methods),
        identification_methods=pl.lit(combined_id_methods),
        notes=pl.lit(combined_notes),
    )
    return repeats_df


def merge_excerpts(df: pl.DataFrame) -> pl.DataFrame:
    """Takes a polars Dataframe that includes labeled excerpts and merges.
    For now, merging is only done on the simple cases where reference
    fields match exactly, or where reference fields are present in one labeled
    excerpt and unset in the other:
    - unlabeled excerpts with matching labeled excerpts
    - multiple labeled excerpts with the same label

    When excerpts are merged, the detection_methods, identification_methods,
    and notes fields are all combined to preserve all information.

    Returns the a dataframe with duplicate excerpts merged.

    """

    # copy the df and add a row index for removal
    updated_df = df.with_row_index()
    # create a df with the same schema but no data to collect merged excerpts
    merged_excerpts = updated_df.clear()

    # merge unlabeled excerpts with matching labeled excerpt
    # OR excerpt with and without notes
    # group by page id and excerpt id only
    for group, data in updated_df.group_by(["page_id", "excerpt_id"]):
        # group is a tuple of values for page id, excerpt id, poem id
        # data is a df of the grouped rows for this set

        # sort so any empty values for optional fields are first,
        # then fill values backward - i.e., treat nulls as duplicates,
        # but keep unlabeled excerpts first
        data = data.sort(
            "poem_id",
            "ref_corpus",
            "ref_span_start",
            "ref_span_end",
            "ref_span_text",
            nulls_last=False,
        ).select(pl.all().backward_fill())

        # identify repeats where everything is the same but the row index,
        # methods, and notes
        # (other values must either be the same or don't conflict because they were unset)
        repeats = data.filter(
            data.drop(
                "index", "detection_methods", "identification_methods", "notes"
            ).is_duplicated()
        )

        if not repeats.is_empty():
            repeats = combine_duplicate_methods_notes(repeats)
            # remove the repeats from the main dataframe
            updated_df = updated_df.filter(
                ~pl.col("index").is_in(repeats.select(pl.col("index")))
            )
            # add the consolidated row with combined values to the merged df
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
    input_dfs = []

    # load files and combine into a single excerpt dataframe
    for input_file in args.input_files:
        try:
            input_dfs.append(load_excerpts_df(input_file))
        except ValueError as err:
            # if any input file does not have minimum required fields, bail out
            print(err, file=sys.stderr)
            sys.exit(-1)

    # combine input dataframes with a "diagonal" concat, which aligns
    # columns and fills in nulls for missing columns in any of the dataframes
    excerpts = pl.concat(input_dfs, how="diagonal")
    # get initial totals before merging
    total_excerpts = excerpts.height
    initial_labeled_excerpts = excerpts.filter(pl.col("poem_id").is_not_null()).height
    # output summary information about input data
    print(
        f"Loaded {total_excerpts:,} excerpts from {len(args.input_files)} files ({initial_labeled_excerpts:,} labeled)."
    )

    # merge labeled + unlabeled excerpts AND duplicate labeled excerpts
    excerpts = merge_excerpts(excerpts)
    # standardize columns so we have all expected fields and no extras
    excerpts = standardize_dataframe(excerpts)

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
        f"{len(excerpts):,} excerpts after merging; {len(labeled_excerpts):,} labeled excerpts."
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
