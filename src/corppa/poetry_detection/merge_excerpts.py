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


def merge_excerpt_groups(
    grouped_df: pl.dataframe.group_by.GroupBy, merge_reason: str = "ppa exact span"
) -> pl.DataFrame:
    """Takes a GroupBy dataframe of excerpts (created by calling `group_by`), and combines
    groups of excerpts into merged excerpts. Merges as follows:
      - first ppa_span_text
      - combined unique set of detection methods
      - combined unique set of notes
      - updated excerpt id
      - first poem id (dataframe should be sorted so preferred id is first)
      - any other poem ids are listed in alt_poem_ids
      - first reference corpus id
      - first reference span and text
      - combined unique list of identification methods
    After merging, it adds a note documenting the group, with the specified reason,
    and the number of raw excerpts in the merged set.
    """
    return (
        grouped_df.agg(
            # TODO: how to handle for overlapping spans
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
                pl.lit(f"; merge: {merge_reason}, "),
                pl.col("group_size"),
                pl.lit(" excerpts"),
            ),
            # if alt poem ids is empty, replace with None
            alt_poem_ids=pl.when(pl.col("alt_poem_ids").list.len() > 0)
            .then(pl.col("alt_poem_ids"))
            .otherwise(pl.lit(None)),
        )
        .drop("group_size")
    )  # drop group size column


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
        # if none were found, create an empty
        output_df = df.clear()

    # any excerpts with group size > 1 are candidates for merging
    merge_candidates = (
        df.join(grouped, on=["page_id", "ppa_span_start", "ppa_span_end"])
        .filter(pl.col("group_size").gt(1))
        .drop("group_size")
    )

    # sort by page then poem id, with nulls last, to ensure we select
    # a non-null poem id and reference data;
    # extract passim match length and sort longest passim matches first
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

    merged_output_df = merge_excerpt_groups(merge_groups)

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
    output_df = pl.concat([output_df, merged_output_df], how="diagonal")

    return output_df


def merge_overlapping_excerpts(excerpts_df) -> pl.DataFrame:
    # TODO: expose overlap factor/length options
    # call identify_overlapping_excerpts to identify overlapping excerpts
    overlaps_df = identify_overlapping_excerpts(excerpts_df)
    # retain non overlaps - identify excerpts not in the overlapping set
    # based on page id & excerpt id (somehow)

    # consolidate overlapping spans into groups of overlapping spans
    overlap_groups = combine_groups(
        overlaps_df.select("group_ids", "page_id"), "group_ids"
    )
    # explode the groups into group + id for merging
    overlap_group_ids = overlap_groups.with_columns(
        excerpt_id=pl.col("group_ids")
    ).explode("excerpt_id")

    # THEN: join to get group ids
    to_merge_df = excerpts_df.join(overlap_group_ids, on=["page_id", "excerpt_id"])
    print(to_merge_df.head())

    # prep/setup is the same as for the exact merge, except
    # now we group by the combined set of group ids
    result_df = merge_excerpt_groups(
        to_merge_df.with_columns(
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
        .group_by(["group_ids"], maintain_order=True),
        merge_reason="98% overlap",
    )
    # combine with other / unmerged?
    # uniquify after merging?
    return result_df


def identify_overlapping_excerpts(
    excerpts_df: pl.DataFrame,
    min_overlap_factor: float = 0.98,
    min_overlap_chars: int = 10,
) -> pl.DataFrame:
    """
    Takes a DataFrame of excerpts and identifies pairs of overlapping excerpts.
    Overlapping excerpts are on the same page, with some shared span of text.
    We exclude short overlaps based on the minimum character parameter,
    and an overlap factor, which is calculated by the length of the shared
    span and the length of the longer of the two spans.

    Returns a DataFrame of excerpt pairs, which includes the page id,
    two excerpt ids, overlap length, and overlap factor.
    """

    # TODO: what about small spans completely inside another?
    # overlap factor would be small

    # identify excerpts with partial overlap
    overlaps_df = (
        excerpts_df
        # Filter to excerpts on pages with multiple excerpts
        .filter(pl.col("page_id").is_duplicated())
        .join_where(
            excerpts_df,
            # 1. Excerpts are on the same page
            pl.col("page_id") == pl.col("page_id_right"),
            # 2. Excerpts overlap:
            #    left span starts before right span ends
            pl.col("ppa_span_start") < pl.col("ppa_span_end_right"),
            #  and right span starts before left span ends
            pl.col("ppa_span_start_right") < pl.col("ppa_span_end"),
            # 3. Exclude self-matches
            pl.col("excerpt_id") != pl.col("excerpt_id_right"),
        )
        .with_columns(
            # make a sorted combined id so we can drop duplicate copies of the same pair
            group_ids=pl.concat_list(
                [pl.col("excerpt_id"), pl.col("excerpt_id_right")]
            ).list.sort()
        )
        # excerpt ids are ONLY unique within a page
        # drop duplicate copies of the same overlapping pair on the same page
        .unique(["group_ids", "page_id"])
        .with_columns(
            # calculate length of the overlap: smaller end minus larger start
            overlap_len=pl.min_horizontal(
                pl.col("ppa_span_end"), pl.col("ppa_span_end_right")
            ).sub(
                pl.max_horizontal(
                    pl.col("ppa_span_start"), pl.col("ppa_span_start_right")
                )
            ),
        )
        .with_columns(
            overlap_factor=pl.col("overlap_len").truediv(
                pl.max_horizontal(
                    pl.col("ppa_span_text").str.len_chars(),
                    pl.col("ppa_span_text_right").str.len_chars(),
                )
            )
        )
        # filter to requested overlap / length to limit to high confidence overlaps
        .filter(
            pl.col("overlap_factor").gt(min_overlap_factor),
            pl.col("overlap_len").gt(min_overlap_chars),
        )
    )

    # what fields are needed here?
    return overlaps_df.select(
        "page_id",
        "excerpt_id",
        "excerpt_id_right",
        "group_ids",  # drop?
        "overlap_len",
        "overlap_factor",
        # these are not strictly needed but may be helpful for investigating
        "notes",
        "notes_right",
        "ppa_span_text",
        "ppa_span_start",
        "ppa_span_end",
        "ppa_span_text_right",
        "ppa_span_start_right",
        "ppa_span_end_right",
        "ref_span_text",
        "ref_span_text_right",
    )


def combine_groups(
    df: pl.DataFrame, group_field: str, group_id: str = "page_id"
) -> pl.DataFrame:
    # take a dataframe with a group field, assumed to be a list of ids
    # iteratively (recursively?) merge all lists with overlaps
    # return the dataframe with the consolidated sets

    left_group_field = pl.col(group_field)
    right_group_field = pl.col(f"{group_field}_right")

    print(f"combining groups, initial dataframe has {df.height:,} rows")

    merged_df = (
        df.join_where(
            df,
            # limit to matching pages first, so that the
            # list intersection is run on the smallest set of possible cases
            pl.col(group_id).eq(pl.col(f"{group_id}_right"))
            & left_group_field.list.set_intersection(right_group_field)
            .list.len()
            .ne(0),
        )
        .with_columns(superset=left_group_field.list.set_union(right_group_field))
        .filter(~left_group_field.eq(right_group_field))
        .unique(["superset", group_id])
        .with_columns(superset_size=pl.col("superset").list.len())
    )
    # if everything is pairs, expect groups of three now
    size_min = merged_df["superset_size"].min()
    size_max = merged_df["superset_size"].max()
    size_mean = merged_df["superset_size"].mean()

    print(
        f"first merge, dataframe has {merged_df.height:,} rows, group size {size_min} min, {size_max} max, {size_mean} mean"
    )
    # get the unmerged groups - anything that didn't match and become part of a new superset
    # calculate group size
    unmerged_df = df.filter(
        ~left_group_field.is_in(merged_df[group_field])
    ).with_columns(group_size=pl.col(group_field).list.len())

    print(f"unmerged groups: {unmerged_df.height:,} rows")

    # now repeat the merge
    if merged_df.height != df.height:
        merged_df = combine_groups(
            merged_df.select("page_id", "superset").rename({"superset": "group_ids"}),
            "group_ids",
        )
    else:
        print("# rows after merging unchanged, bailing out")
        # rename columns to match returned
        merged_df = merged_df.select("page_id", "superset", "superset_size").rename(
            {"superset": "group_ids", "superset_size": "group_size"}
        )

    # combine merged and unmerged
    return pl.concat([unmerged_df, merged_df], how="diagonal")

    return merged_df


def merge_excerpt_files(
    input_files: list[pathlib.Path], output_file: pathlib.Path
) -> pl.DataFrame:
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

    # polars supports compression; but not sure what version it
    # was added in, and documentation says it is unstable. Use that in future
    excerpts.write_csv(output_file)
    # return excerpt data frame
    return excerpts


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
