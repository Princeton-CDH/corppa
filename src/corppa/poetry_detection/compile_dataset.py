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

from corppa.config import get_config
from corppa.poetry_detection.merge_excerpts import merge_excerpt_files
from corppa.poetry_detection.ppa_works import load_ppa_works_df

# from corppa.utils.path_utils import find_relative_paths
from corppa.poetry_detection.ref_corpora import save_poem_metadata

DEFAULT_CONFIGS = {
    "source_excerpt_data": "excerpt-data",
    "source_ppa_metadata": "ppa-data/ppa_works.csv",
}


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
    output_data_dir = pathlib.Path(config_opts["compiled_dataset"]["data_dir"])
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
        source_base_dir = pathlib.Path(
            config_opts["compiled_dataset"]["source_base_dir"]
        )
    except KeyError:
        print(
            "Configuration error: `compiled_dataset.source_base_dir` not found in config file",
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


def get_excerpt_sources(excerpt_data_dir: pathlib.Path) -> list[pathlib.Path]:
    return list(excerpt_data_dir.glob("**/*.csv")) + list(
        excerpt_data_dir.glob("**/*.csv.gz")
    )
    # wondered about using find_relative_paths here, but we actually
    # want non-relative paths and we need to handle a two-part extension
    # return [
    #     excerpt_data_dir / rel_path
    #     for rel_path in find_relative_paths(excerpt_data_dir, exts=[".csv", ".gz"]) # can we assume .gz == .csv.gz ?
    # ]


def save_ppa_metadata(input_file: pathlib.Path, output_file: pathlib.Path):
    load_ppa_works_df(input_file).write_csv(output_file)


def compress_file(uncompressed_file, compressed_file):
    # FIXME: this is the example in the docs but does not seem to work 😬
    with open(str(uncompressed_file), "rb") as inputfile:
        with gzip.open(str(compress_file), "wb") as output_file:
            shutil.copyfileobj(inputfile, output_file)


def main():
    parser = argparse.ArgumentParser(description="Compile PPA found-poems dataset")
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
    args = parser.parse_args()
    compilation_steps = args.steps  # None or list of steps

    compile_opts = load_compilation_config()

    if compilation_steps is None or "merge" in compilation_steps:
        print("## Merging excerpts")
        # find excerpt source files to be included in the compiled dataset file
        excerpt_sources = get_excerpt_sources(compile_opts["source_excerpt_data"])
        # merge into a single uncompressed csv
        # (polars doesn't currently support writing directly to a csv.gz)
        merge_excerpt_files(excerpt_sources, compile_opts["compiled_excerpt_file"])
        # compress the resulting file
        print(
            f"Compressing excerpt data... ({compile_opts['compiled_excerpt_file']} → {compile_opts['compressed_excerpt_file']})"
        )
        compress_file(
            compile_opts["compiled_excerpt_file"],
            compile_opts["compressed_excerpt_file"],
        )

    if compilation_steps is None or "poem_metadata" in compilation_steps:
        print("\n## Compiling reference corpora metadata")
        save_poem_metadata(compile_opts["poem_metadata_file"])

    if compilation_steps is None or "ppa_metadata" in compilation_steps:
        print("\n## PPA work-level metadata")
        save_ppa_metadata(
            compile_opts["source_ppa_metadata"], compile_opts["ppa_metadata_file"]
        )

    print("\nRemember to commit and push the updated data files")
    print(f"cd {compile_opts['output_data_dir'].parent} && git add data/*")


if __name__ == "__main__":
    main()
