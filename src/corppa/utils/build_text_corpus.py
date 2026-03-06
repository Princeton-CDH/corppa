# Copyright (c) 2024-2025, Center for Digital Humanities, Princeton University
# SPDX-License-Identifier: Apache-2.0

"""
Script for building a text corpus file (JSONL) from a directory of texts.

This script converts each text (``.txt``) file within an input directory
(including  nested files), and compiles them into a single output text corpus
where each record corresponds to a single text file with the following fields:

    - ``id``: The name of the file (without prefix)
    - ``text``: The text of the file (assumes UTF-8 formatting)

Note that the output file can also be written in any compressed form supported
by :mod:`orjsonl`. If no suffix is provided, ``.jsonl`` will be used.

Example usage: ::

    python build_text_corpus.py input_dir out_corpus

    python build_text_corpus.py input_dir out_corpus.jsonl

    python build_text_corpus.py input_dir out_corpus.jsonl.gz

"""

import argparse
import pathlib
import sys
import tarfile

import orjsonl
from tqdm import tqdm


def get_text_record(text_file: pathlib.Path) -> dict[str, str]:
    """
    Create basic text record for input text file
    """
    record = {
        "id": text_file.stem,
        "text": text_file.read_text(encoding="utf-8"),
    }
    return record


def build_text_corpus(
    input_path: pathlib.Path, disable_progress: bool = False
) -> dict[str, str]:
    """
    Generates text records for each text file within input directory
    """
    progress_bar = tqdm(
        input_path.glob("**/*.txt"),
        bar_format="Read {n:,} pages{postfix} | elapsed: {elapsed}",
        disable=disable_progress,
    )
    for text_file in progress_bar:
        yield get_text_record(text_file)


def text_corpus_from_tarfile(
    input_path: pathlib.Path, disable_progress: bool = False
) -> dict[str, str]:
    """
    Generate text records for each text file within a tar.gz archive
    """
    # NOTE: could make gz compression optional, curerntly assumes
    # todo: currently no progressbar
    with tarfile.open(str(input_path), "r:gz") as tar_archive:
        for member in tqdm(
            tar_archive.getmembers(),
            bar_format="Read {n:,} files{postfix} | elapsed: {elapsed}",
            disable=disable_progress,
        ):
            # skip any OSX metadata files included in the archive
            if "._" in member.name:
                continue

            # read contents of text files and yield filename and contents
            if member.name.endswith(".txt"):
                txtfile = tar_archive.extractfile(member)
                if txtfile is not None:
                    yield {
                        "id": pathlib.Path(member.name).stem,
                        "text": txtfile.read().decode("utf-8"),
                    }


def save_text_corpus(
    input_dir: pathlib.Path,
    output_file=pathlib.Path,
    disable_progress: bool = False,
) -> None:
    """
    From the text files within the provided input directory, build a
    text corpus JSONL file.
    """
    orjsonl.save(output_file, build_text_corpus(input_dir))


def main():
    parser = argparse.ArgumentParser(
        description="Build text corpus (JSONL) from directory of text files."
    )
    # Required arguments
    parser.add_argument(
        "input_dir",
        help="Top-level input directory containing text files that will make up the text corpus",
        type=pathlib.Path,
    )
    parser.add_argument("output", help="Filename of output JSONL", type=pathlib.Path)
    # Optional arguments
    parser.add_argument(
        "--progress",
        help="Show progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(
            f"Error: input directory {args.input_dir} does not exist", file=sys.stderr
        )
        sys.exit(1)
    # If output file has no extension, add ".jsonl"
    output_file = args.output
    if not output_file.suffix:
        output_file = output_file.with_suffix(".jsonl")

    save_text_corpus(args.input_dir, output_file, disable_progress=not args.progress)


if __name__ == "__main__":
    main()
