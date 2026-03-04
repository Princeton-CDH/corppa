#!/usr/bin/env python
# Copyright (c) 2024-2026, Center for Digital Humanities, Princeton University
# SPDX-License-Identifier: Apache-2.0

"""
This script and associated method merges labeled and unlabeled poem excerpts 
with matching spans in the PPA page text.

It takes two or more input files of excerpt data (labeled or unlabeled) in CSV format,
merges any excerpts that can be combined, and outputs a CSV with the updated excerpt data.
All excerpts in the input data files are preserved in the output, whether
they were merged with any other records or not. This means that in most cases,
the output will likely be a mix of labeled and unlabeled excerpts.

Merging logic is as follows:

- Excerpts are grouped based on exact span match in PPA text (i.e., the
  combination of `page_id`, `ppa_span_start`, and `ppa_span_end`)
  even when poem identifications differ, and combined as follows:

     - Excerpts are sorted by `poem_id`, `ref_span_start`, and passim
       match length with nulls last and longest passim match first.
       Reference information (`poem_id`, `ref_span_start`, `ref_span_end`, 
       `ref_span_text`, `ref_corpus`) is taken from the first excerpt in 
       the group.
     - When merged excerpts have different poem identifications, all unique
       poem ids after the first are collected into `alt_poem_ids`
     - The `detection_methods` and `identification_methods` fields are combined
       to the unique set of methods used in the merged excerpts.
     - The `notes` field is combined with the set of all unique content from 
       notes in merged excerpts with an additional note about the merge.

Example usage: ::

./src/corppa/poetry_detection/merge_excerpts.py adjudication_excerpts.csv \
labeled_excerpts.csv -o merged_excerpts.csv

Limitations:

- Merge logic collapses different poem ids that may not correspond;
  they may be subsets of the same poem, different editions, or entirely
  different poems.
- Currently supports CSV input and output only.

"""

import argparse
import pathlib
import sys

import polars as pl

from corppa.poetry_detection.core import MULTIVAL_DELIMITER
from corppa.poetry_detection.polars_utils import load_excerpts_df, standardize_dataframe


def merge_excerpts(
    df: pl.DataFrame, disable_progress=True, verbose=False
) -> pl.DataFrame:
    """Takes a polars DataFrame that includes labeled or unlabeled excerpts,
    and merges excerpts based primarily on `page_id` and `excerpt_id`.
    For now, merging is only done on the simple cases where reference
    fields match exactly, or where reference fields are present in one labeled
    excerpt and unset in the other:
    - unlabeled excerpts with matching labeled excerpts
    - multiple labeled excerpts with matching `poem_id` and non-conflicting
    reference information

    When excerpts are merged, the detection_methods, identification_methods,
    and notes fields are all combined to preserve all information.

    Returns a dataframe that contains all unique excerpts and merged
    versions of duplicated excerpts.
    """

    # TEMPORARY - make sure internet poem ref corpus ids match before merging
    df = df.with_columns(
        ref_corpus=pl.when(pl.col("ref_corpus").eq("internet-poems"))
        .then(pl.lit("internet_poems"))
        .otherwise(pl.col.ref_corpus)
    )

    # group by page id and excerpt id to get potential matches
    # use aggregation to get the count of excerpts in each group,
    # then split input dataframe into singletons and merge candidates
    # NOTE: span start/end to merge across systems, because excerpt id includes detection method
    grouped = df.group_by(["page_id", "ppa_span_start", "ppa_span_end"]).agg(
        pl.len().alias("group_size")
    )
    # any excerpts with group size one will not be merged;
    # add to output df and don't process further
    output_df = (
        df.join(grouped, on=["page_id", "ppa_span_start", "ppa_span_end"])
        .filter(pl.col("group_size").eq(1))
        .drop("group_size")
    )
    if output_df.is_empty():
        output_df = df.clear()

    # any excerpts with group size > 1 are candidates for merging
    merge_candidates = (
        df.join(grouped, on=["page_id", "ppa_span_start", "ppa_span_end"])
        .filter(pl.col("group_size").gt(1))
        .drop("group_size")
    )

    # sort by page then poem id, with nulls last, to ensure we select
    # a non-null poem id and reference data
    merge_groups = (
        merge_candidates.with_columns(
            # extract passim match length so we can prioritize longer matches
            passim_match_len=pl.col("notes").str.extract(r"passim: (\d+) char matches")
        )
        .sort(
            "page_id",
            "passim_match_len",
            "poem_id",
            "ref_span_start",
            nulls_last=True,
            descending=[False, True, False, False],  # sort longest passim matches first
        )
        .group_by(["page_id", "ppa_span_start", "ppa_span_end"], maintain_order=True)
    )
    num_merge_groups = merge_groups.len().height
    if verbose:
        print(
            f"Identified {merge_candidates.height:,} merge candidates in {num_merge_groups:,} groups."
        )

    merged_output_df = (
        merge_groups.agg(
            pl.first("ppa_span_text"),  # should match exactly
            pl.col("detection_methods")
            .explode()
            .unique(),  # combine in a single list, no repeats
            # combine notes but don't repeat duplicate info (like passim char match count)
            pl.col("notes").explode().unique().sort().str.join("; "),
            # construct merged excerpt id manually; c= prefix for combined
            # (although strictly speaking should only be if > 1 detection method)
            pl.concat_str(
                pl.lit("c@"),
                pl.col("ppa_span_start").first(),
                pl.lit(":"),
                pl.col("ppa_span_end").first(),
            ).alias("excerpt_id"),
            # pick the first poem id (relies on previous sorting)
            pl.col("poem_id").explode().unique().first(),
            # and store all others in alt poem ids field
            pl.col("poem_id")
            .explode()
            .unique()
            .drop_nulls()
            .slice(1)
            .alias("alt_poem_ids"),
            pl.col("ref_corpus").explode().first(),
            # use first reference span and text so numbers are useful; ignore nulls
            pl.col("ref_span_start").first(),
            pl.col("ref_span_end").first(),
            pl.col("ref_span_text").first(),
            # combine unique list of id methods
            pl.col("identification_methods")
            .explode()
            .unique()
            .drop_nulls(),  # combine in a single list, no repeats, ignore nulls (not identified before merging)
            pl.len().alias("group_size"),  # count number in the group
        )
        .with_columns(
            notes=pl.concat_str(
                pl.col("notes"),
                pl.lit("; merge: ppa exact span, "),
                pl.col("group_size"),
                pl.lit(" excerpts"),
            ),
            # if alt poem ids is empty, replace with None
            alt_poem_ids=pl.when(pl.col("alt_poem_ids").list.len() > 0)
            .then(pl.col("alt_poem_ids"))
            .otherwise(pl.lit(None)),
        )
        .drop("group_size")  # drop group size column
    )

    if verbose:
        multi_id = merged_output_df.filter(
            pl.col("alt_poem_ids").list.len().gt(0)
        ).height
        print(
            f"{merged_output_df.height:,} merged excerpts; {multi_id:,} with multiple poem ids."
        )

    # combined merged records with the output
    # use a diagonal concat instead of vstack/extend
    # to avoid having to reconcile columns first
    return pl.concat([output_df, merged_output_df], how="diagonal")


def merge_excerpt_files(input_files, output_file):
    total_excerpts = 0
    input_dfs = []

    # load files and combine into a single excerpt dataframe
    for input_file in input_files:
        try:
            input_dfs.append(load_excerpts_df(input_file))
        except ValueError as err:
            # if any input file does not have minimum required fields, bail out
            print(err, file=sys.stderr)
            sys.exit(-1)

    # combine input dataframes with a "diagonal" concat, which aligns
    # columns and fills in nulls for missing columns in any of the dataframes
    # NOTE: very important to standardize columns so that extraneous input
    # columns do not prevent duplicate excerpts from merging
    excerpts = standardize_dataframe(pl.concat(input_dfs, how="diagonal"))
    # get initial totals before any uniquifying or merging
    total_excerpts = excerpts.height
    # use unique to drop exact duplicates
    excerpts = excerpts.unique()
    initial_labeled_excerpts = excerpts.filter(pl.col("poem_id").is_not_null()).height
    # output summary information about input data
    print(
        f"Loaded {total_excerpts:,} excerpts from {len(input_files)} files ({excerpts.height:,} unique; {initial_labeled_excerpts:,} labeled)."
    )

    # merge labeled + unlabeled excerpts AND duplicate labeled excerpts
    # display progress bar & output summary information
    excerpts = merge_excerpts(excerpts, disable_progress=False, verbose=True)
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
        alt_poem_ids=pl.col("alt_poem_ids").list.join(MULTIVAL_DELIMITER),
    )

    labeled_excerpts = excerpts.filter(pl.col("poem_id").is_not_null())

    # summary information about the content and what as done
    print(
        f"\n{len(excerpts):,} excerpts after merging; {len(labeled_excerpts):,} labeled excerpts."
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

    excerpts.write_csv(output_file)


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

    merge_excerpt_files(args.input_files, args.output)


if __name__ == "__main__":
    main()
