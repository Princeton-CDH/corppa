"""
This script compiles the PPA Found Poems dataset.

It depends on compiled_dataset and reference_corpora configuration
settings, as described in the project readme and seen in `sample_config.yml`.

To run compilation with all steps (default behavior)::
```console
compile-dataset
```

To run one or more specific steps, specify which steps you want to run.
Any string that is distinct will be enough to select the step.
```console
compile-dataset --merge
compile-dataset --poem-metadata
compile-dataset --poem-metadata --ppa-metadata
compile-dataset --m --poem -ppa
```

"""

import argparse
import gzip
import pathlib
import shutil
import sys

import polars as pl

from corppa.config import get_config
from corppa.poetry_detection.merge_excerpts import merge_excerpt_files
from corppa.poetry_detection.polars_utils import add_ref_poems_meta
from corppa.poetry_detection.ppa_works import extract_page_meta
from corppa.poetry_detection.ref_corpora import save_poem_metadata

DEFAULT_CONFIGS = {
    "source_excerpt_data": "excerpt-data",
    "source_ppa_metadata": "ppa-data/ppa_works.csv",
}

#: compile script config options, for run_step method type hints
CompileOpts = dict[str, pathlib.Path]


def load_compilation_config():
    """Load configuration for dataset compilation,
    validating that required configurations are present, paths exist, etc.
    """
    config_opts = get_config()
    required_sections = ["compiled_dataset", "reference_corpora"]
    for section in required_sections:
        if section not in config_opts:
            print(
                f"Configuration error: '{section}' not found in config file",
                file=sys.stderr,
            )
            sys.exit(-1)

    # output directory
    try:
        output_data_dir = pathlib.Path(config_opts["compiled_dataset"]["data_dir"])
    except KeyError as err:
        raise ValueError(
            "Configuration error: config file requires `compiled_dataset.data_dir` path"
        ) from err
    if not output_data_dir.exists():
        raise ValueError(
            f"Configuration error: compiled dataset path {output_data_dir} does not exist"
        )
    if not output_data_dir.is_dir():
        raise ValueError(
            f"Configuration error: compiled dataset path {output_data_dir} is not a directory"
        )

    # filenames where compiled data will be saved
    compiled_excerpt_file = output_data_dir / "excerpts.csv"
    compressed_excerpt_file = output_data_dir / "excerpts.csv.gz"
    poem_metadata_file = output_data_dir / "poem_meta.csv"
    ppa_metadata_file = output_data_dir / "ppa_work_metadata.csv"

    # source directories
    try:
        source_base_dir = pathlib.Path(config_opts["data_ingredients_dir"])
    except KeyError:
        print(
            "Configuration error: `data_ingredients_dir` not found in config file",
            file=sys.stderr,
        )
        sys.exit(-1)

    if not source_base_dir.exists():
        raise ValueError(
            f"Configuration error: compiled dataset source dir {source_base_dir} does not exist"
        )
    if not source_base_dir.is_dir():
        raise ValueError(
            f"Configuration error: compiled dataset source dir {source_base_dir} is not a directory"
        )

    # excerpt data dir - get from config if set
    excerpt_data_dir = pathlib.Path(
        config_opts["compiled_dataset"].get(
            "source_excerpt_data", DEFAULT_CONFIGS["source_excerpt_data"]
        )
    )
    # if path is not absolute, make relative to source base directory
    if not excerpt_data_dir.is_absolute():
        excerpt_data_dir = source_base_dir / excerpt_data_dir

    # ppa metadata
    source_ppa_metadata = pathlib.Path(
        config_opts["compiled_dataset"].get(
            "source_ppa_metadata", DEFAULT_CONFIGS["source_ppa_metadata"]
        )
    )
    # if path is not absolute, make relative to source base directory
    if not source_ppa_metadata.is_absolute():
        source_ppa_metadata = source_base_dir / source_ppa_metadata
    if not source_ppa_metadata.exists() or not source_ppa_metadata.is_file():
        raise ValueError(
            f"Configuration error: PPA metadata file {source_ppa_metadata} does not exist"
        )

    return {
        # outputs
        "output_data_dir": output_data_dir,
        "compiled_excerpt_file": compiled_excerpt_file,
        "compressed_excerpt_file": compressed_excerpt_file,
        "poem_metadata_file": poem_metadata_file,
        "ppa_metadata_file": ppa_metadata_file,
        # sources
        "source_excerpt_data": excerpt_data_dir,
        "source_ppa_metadata": source_ppa_metadata,
    }


def load_compiled_excerpts(config) -> pl.DataFrame:
    for datafile in [
        config["compiled_excerpt_file"],
        config["compressed_excerpt_file"],
    ]:
        if datafile.exists():
            # extract ppa work id and page number (needed for both poem and ppa metadata)
            return extract_page_meta(pl.read_csv(datafile))
    raise ValueError(
        f"Excerpt data file not found (checked {config['compiled_excerpt_file']} and {config['compressed_excerpt_file']}"
    )


def get_excerpt_sources(excerpt_data_dir: pathlib.Path) -> list[pathlib.Path]:
    return list(excerpt_data_dir.glob("**/*.csv")) + list(
        excerpt_data_dir.glob("**/*.csv.gz")
    )


def save_ppa_metadata(
    input_file: pathlib.Path, output_file: pathlib.Path, excerpts_df: pl.DataFrame
):
    # copy as-is, do not rename or subset any fields
    # NOTE: currently assumes and only supports PPA metadata in csv format
    if input_file.suffix != ".csv":
        raise ValueError(
            f"PPA metadata must be loaded as CSV, got {input_file.suffix.lstrip('.')}"
        )
    ppa_meta_df = pl.read_csv(input_file)

    # get work-level aggregate excerpt totals
    excerpt_totals_df = excerpts_df.group_by("ppa_work_id").agg(
        pl.col("excerpt_id").n_unique().alias("num_excerpts"),
        pl.col("poem_id").n_unique().alias("num_poems"),
        pl.col("poem_author").n_unique().alias("num_poets"),
    )

    # combine the totals with ppa work metadata
    ppa_meta_df = ppa_meta_df.join(
        excerpt_totals_df, left_on="work_id", right_on="ppa_work_id", how="left"
    ).with_columns(
        # fill any missing values with zeroes
        pl.col("num_excerpts").fill_null(pl.lit(0)),
        pl.col("num_poems").fill_null(pl.lit(0)),
        pl.col("num_poets").fill_null(pl.lit(0)),
    )

    ppa_meta_df.write_csv(output_file)


def compress_file(uncompressed_file, compressed_file):
    with open(str(uncompressed_file), "rb") as inputfile:
        with gzip.open(str(compressed_file), "wb") as output_file:
            shutil.copyfileobj(inputfile, output_file)
    # report sizes before/after? maybe return them?
    # remove the uncompressed file
    uncompressed_file.unlink()


def run_merge_step(
    compile_opts: CompileOpts, excerpts_df: pl.DataFrame | None, compress_excerpts: bool
) -> pl.DataFrame:
    print("## Merging excerpts")
    excerpt_sources = get_excerpt_sources(compile_opts["source_excerpt_data"])
    excerpts_df = merge_excerpt_files(
        excerpt_sources, compile_opts["compiled_excerpt_file"]
    )
    if compress_excerpts:
        print(
            f"Compressing excerpt data... {compile_opts['compiled_excerpt_file']} → {compile_opts['compressed_excerpt_file']}"
        )
        compress_file(
            compile_opts["compiled_excerpt_file"],
            compile_opts["compressed_excerpt_file"],
        )
    return excerpts_df


def run_poem_metadata_step(
    compile_opts: CompileOpts, excerpts_df: pl.DataFrame | None = None
) -> None:
    print("\n## Compiling reference corpora metadata")
    if excerpts_df is None:
        excerpts_df = load_compiled_excerpts(compile_opts)
    else:
        excerpts_df = extract_page_meta(excerpts_df)
    save_poem_metadata(compile_opts["poem_metadata_file"], excerpts_df)


def run_ppa_metadata_step(
    compile_opts: CompileOpts, excerpts_df: pl.DataFrame | None = None
) -> None:
    print("\n## PPA work-level metadata")
    if excerpts_df is None:
        excerpts_df = load_compiled_excerpts(compile_opts)
    else:
        excerpts_df = extract_page_meta(excerpts_df)

    excerpts_df = add_ref_poems_meta(excerpts_df, compile_opts["poem_metadata_file"])

    save_ppa_metadata(
        compile_opts["source_ppa_metadata"],
        compile_opts["ppa_metadata_file"],
        excerpts_df,
    )


def main(*args):
    parser = argparse.ArgumentParser(description="Compile PPA found-poems dataset")
    parser.add_argument(
        "--compress-excerpts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    # add an argument group to allow easily specifying specific steps
    step_arg_group = parser.add_argument_group(
        "Step",
        "Only run specific compilation steps",
    )
    compilation_steps = {
        "merge": "Merge excerpts",
        "poem_metadata": "Compile reference corpus poetry metadata",
        "ppa_metadata": "Compile filtered and renamed PPA work-level metadata",
    }
    for step, description in compilation_steps.items():
        step_arg_group.add_argument(
            f"--{step}",
            help=description,
            metavar="",
            dest="steps",
            action="append_const",
            const=step,
        )
    args = parser.parse_args(args)
    # if not specified, run all steps
    compilation_steps = args.steps if args.steps else list(compilation_steps.keys())

    compile_opts = load_compilation_config()

    excerpts_df = None
    if "merge" in compilation_steps:
        excerpts_df = run_merge_step(compile_opts, excerpts_df, args.compress_excerpts)

    if "poem_metadata" in compilation_steps:
        run_poem_metadata_step(compile_opts, excerpts_df)

    if "ppa_metadata" in compilation_steps:
        run_ppa_metadata_step(compile_opts, excerpts_df)

    # probably not relevant anymore, not using git-lfs for this data...
    print(f"Output files in {compile_opts['output_data_dir']}")
    # print("\nRemember to commit and push the updated data files")
    # print(f"cd {compile_opts['output_data_dir'].parent} && git add data/*")


if __name__ == "__main__":
    main(sys.argv)
