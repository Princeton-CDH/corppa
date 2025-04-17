#!/usr/bin/env python
"""
This script merges labeled and unlabeled poem excerpts, combining
notes for any merged excerpts, and merging duplicate poem identifications
in simple cases.

It takes two or more input files of excerpt data (labeled or unlabeled) in CSV format,
merges any excerpts that can be combined, and outputs a CSV with the updated excerpt data.
All excerpts in the input data files are preserved in the output, whether
they were merged with any other records or not. This means that in most cases,
the output will likely be a mix of labeled and unlabeled excerpts.

Merging logic is as follows:

- Excerpts are grouped on the combination of page id and excerpt id,
  and then merged if all reference fields match exactly, or where
  reference fields are present in one excerpt and unset in the other.

    - If the same excerpt has different labels (different `poem_id` values), both
      labeled excerpts will be included in the output
    - If the same excerpt has duplicate labels (i.e., the same `poem_id` from two
      different identification methods), they will be merged
      into a single labeled excerpt; the `identification_methods` in the
      resulting labeled excerpt will be the union of methods in the merged excerpts

- When merging excerpts where both records have notes, notes content
  will be combined.

Example usage: ::

./src/corppa/poetry_detection/merge_excerpts.py adjudication_excerpts.csv \
labeled_excerpts.csv -o merged_excerpts.csv

Limitations:

- Labeled excerpts with the same poem_id but different reference data
  will not be merged; supporting multiple identification methods that output
  span information will likely require more sophisticated merge logic
- CSV input and output only (JSONL may be added in future)
- Notes are currently merged only with the first matching excerpt; if an 
  unlabeled excerpt with notes has multiple labels, only the first match
  will have combined notes

"""

import argparse
import itertools
import pathlib
import sys

import polars as pl
from tqdm import tqdm

from corppa.poetry_detection.core import MULTIVAL_DELIMITER
from corppa.poetry_detection.polars_utils import load_excerpts_df, standardize_dataframe


def combine_duplicate_methods_notes(repeats_df: pl.DataFrame) -> pl.DataFrame:
    """
    Takes a dataframe of repeated excerpts with duplicate information,
    and updates all rows with the combined set of unique
    detection_methods, identification_methods, and notes. Returns the
    updated dataframe with the combined fields.

    Intended for use on grouped excerpts in :meth:`merge_excerpts`.
    """
    # get detection methods as a list of lists, use itertools to unwrap
    # the lists; consume the itertools generator and use set to uniquify,
    # then convert back to list to put back in the polars dataframe
    detect_methods = repeats_df["detection_methods"].drop_nulls().to_list()
    combined_detect_methods = list(set(itertools.chain.from_iterable(detect_methods)))
    # id methods could be all unset even in a group
    id_methods = repeats_df["identification_methods"].drop_nulls().to_list()
    combined_id_methods = None
    if id_methods:
        combined_id_methods = list(set(itertools.chain.from_iterable(id_methods)))
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
    grouped = df.group_by(["page_id", "excerpt_id"]).agg(pl.len().alias("group_size"))
    # any excerpts with group size one will not be merged;
    # add to output df and don't process further
    output_df = (
        df.join(grouped, on=["page_id", "excerpt_id"])
        .filter(pl.col("group_size").eq(1))
        .drop("group_size")
    )
    if output_df.is_empty():
        output_df = df.clear()

    # any excerpts with group size > 1 are candidates for merging
    merge_candidates = (
        df.join(grouped, on=["page_id", "excerpt_id"])
        .filter(pl.col("group_size").gt(1))
        .drop("group_size")
    )

    merge_groups = merge_candidates.group_by(["page_id", "excerpt_id"])
    num_merge_groups = merge_groups.len().height
    if verbose:
        print(
            f"Identified {merge_candidates.height:,} merge candidates in {num_merge_groups:,} groups.\n"
        )

    progress_groups = tqdm(
        merge_groups,
        total=num_merge_groups,
        desc="Merging...",
        disable=disable_progress,
    )
    merge_count = 0
    for group, data in progress_groups:
        # group is a tuple of values for page id, excerpt id, poem id
        # data is a df of the grouped rows for this set

        # sort so any empty values for optional reference fields are first,
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

        # in case this set of excerpts has multiple different poem ids
        # which should be merged with each other, group again on poem id
        for poem_group, poem_data in data.group_by(["poem_id"]):
            # group of 1 : no merge, add to the output
            if poem_data.height == 1:
                output_df.extend(poem_data)
            # otherwise, look for repeats
            else:
                # combine if everything is the same but methods, and notes
                # (other values must either be the same or don't conflict because they were unset)
                repeat_counts = poem_data.with_columns(
                    duplicate=poem_data.drop(
                        "detection_methods", "identification_methods", "notes"
                    ).is_duplicated()
                )
                # repeats will be consolidated
                repeats = repeat_counts.filter(pl.col("duplicate")).drop("duplicate")
                # any non-repeats should be included in output as-is
                output_df.extend(
                    repeat_counts.filter(~pl.col("duplicate")).drop("duplicate")
                )

                if not repeats.is_empty():
                    repeats = combine_duplicate_methods_notes(repeats)
                    # add one copy of the consolidated information to the merge df
                    output_df.extend(repeats[:1])
                    merge_count += 1
                    progress_groups.set_postfix_str(f"Merged {merge_count:,}")

    return output_df


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
