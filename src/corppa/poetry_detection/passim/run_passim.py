"""
Script for running passim. Requires Java 8*/11/17 because of Spark dependency.

Examples:
```
python run_passim.py --ppa-corpus ppa_passim.jsonl --ref-corpus internet-poems_passim.jsonl \
        --ref-corpus chadwyck-healey_passim.jsonl --output passim_results

python run_passim.py --ppa-corpus ppa_passim.jsonl --ref-corpus ref-poems_passim.jsonl \
        --output passim_results --ngram-size 10 -min-align 25
```
"""

import argparse
import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from subprocess import CalledProcessError, run
from timeit import default_timer as timer


def set_spark_env_vars(
    local_ip="127.0.0.1",
    submit_args="--master local[6] --driver-memory 8G --executor-memory 8G",
) -> None:
    """
    Sets Spark environment variables.
    """
    os.environ["SPARK_LOCAL_IP"] = local_ip
    os.environ["SPARK_SUBMIT_ARGS"] = submit_args


def get_java_version() -> str:
    """
    Checks and returns Java version. If the version is unsupported, a RuntimeError is raised.
    """
    # Get major version of Java
    p = run("java -version", shell=True, capture_output=True, text=True)
    java_version_search = re.search(r'"([^"]*)"', p.stderr)
    java_version = "" if java_version_search is None else java_version_search.group(1)
    if java_version.startswith("11.") or java_version.startswith("17."):
        # Supported: Java 11, Java 17
        return java_version
    elif java_version.startswith("1.8."):
        # Supported: Java 8, update 371 and higher
        if "_" in java_version:
            update_version = int(java_version.split("_")[1])
            if update_version >= 371:
                return java_version
        err_msg = f"Java {java_version} is unsupported. Only versions 8u371 and higher are supported."
    else:
        err_msg = f"Java {java_version} is unsupported. Spark requires Java 8*/11/17."
    raise RuntimeError(err_msg)


def build_input_string(ppa_corpus: Path, ref_corpora: Iterable[Path]) -> str:
    """
    Constructs a Passim input string
    """
    corpus_files = [ppa_corpus] + ref_corpora
    return f"{{{','.join(map(str, corpus_files))}}}"


def run_passim(
    ppa_corpus: Path,
    ref_corpora: Iterable[Path],
    output_dir: Path,
    max_df: int = 100,
    min_match: int = 5,
    ngram_size: int = 25,
    gap: int = 600,
    min_align: int = 50,
    floating_ngrams: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Runs passim with provided input arguments.

    Raises RuntimeError if Java version is unsupported.
    """
    # Check java version
    java_version = get_java_version()
    if verbose:
        print(f"DEBUG: Running passim with Java {java_version}")

    input_str = build_input_string(ppa_corpus, ref_corpora)

    passim_args = [
        "passim",
        input_str,
        output_dir,
        "--fields",
        "corpus",
        "--filterpairs",
        "corpus <> 'ppa' AND corpus2 = 'ppa'",
        "--pairwise",
        "-u",
        f"{max_df}",
        "-m",
        f"{min_match}",
        "-n",
        f"{ngram_size}",
        "-g",
        f"{gap}",
        "-a",
        f"{min_align}",
    ]

    if floating_ngrams:
        passim_args.append("--floating-ngrams")

    # Run passim
    try:
        # TODO: Capturing passim output to "hide" it from view
        _ = run(passim_args, check=True, capture_output=not verbose)
    except CalledProcessError:
        print("ERROR: An error occurred while running passim", file=sys.stderr)
        return False

    if not output_dir.joinpath("align.json", "_SUCCESS").is_file():
        print("ERROR: An error occurred while running passim", file=sys.stderr)
        return False

    return True


def main():
    parser = argparse.ArgumentParser("Run passim to identify poetry excerpts")
    # Required arguments
    parser.add_argument(
        "--ppa-corpus",
        help="Path to PPA passim-friendly corpus file (JSONL)",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--ref-corpus",
        help="Path to reference passim-friendly corpus file (JSONL). Can specify multiple",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        help="Pathnname to the top-level output directory where results will be written",
        type=Path,
        required=True,
    )
    # Optional arguments
    parser.add_argument(
        "--max-df",
        help="Passim parameter (maxDF): upper limit on document frequency",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--min-match",
        help="Passim parameter (min-match): minimum number of n-gram matches between documents",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--ngram-size",
        help="Passim parameter (n): n-gram order",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--floating-ngrams",
        help="Passim parameter (floating-ngrams): allow n-grams to float from word boundaries",
        action="store_true",
    )
    parser.add_argument(
        "--gap",
        help="Passim parameter (gap): minimum size of gap that separates passage",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--min-align",
        help="Passim paramaeter (min-align): minimum length of alignment",
        type=int,
        default=25,
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    # Validate paths
    if not args.ppa_corpus.is_file():
        print(
            f"Error: PPA corpus {args.ppa_corpus} does not exist",
            file=sys.stderr,
        )
        sys.exit(1)
    for ref in args.ref_corpus:
        if not ref.is_file():
            print(
                f"Error: reference corpus {ref} does not exist",
                file=sys.stderr,
            )
            sys.exit(1)

    # Set spark env vars
    # TODO: Create args so that the env vars can be modified
    set_spark_env_vars()

    # Run passim
    start = timer()
    success = run_passim(
        args.ppa_corpus,
        args.ref_corpus,
        args.output_dir,
        max_df=args.max_df,
        min_match=args.min_match,
        ngram_size=args.ngram_size,
        gap=args.gap,
        min_align=args.min_align,
        floating_ngrams=args.floating_ngrams,
        verbose=args.verbose,
    )
    end = timer()

    # Report success/failure & time elapsed
    time_elapsed = end - start
    if success:
        print(f"Passim run completed successfully in {time_elapsed:.1f}s")
    else:
        print(f"Passim run failed in {time_elapsed: .1f}s")


if __name__ == "__main__":
    main()
