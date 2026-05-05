# Copyright (c) 2024-2025, Center for Digital Humanities, Princeton University
# SPDX-License-Identifier: Apache-2.0

"""
Load local configuration options
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:  # pragma: no cover
    from yaml import Loader  # pragma: no cover

#: src dir relative to this file (assuming dev environment for now)
CORPPA_SRC_DIR = Path(__file__).parent.parent.absolute()

#: expected path for local config file (non-versioned)
CORPPA_CONFIG_PATH = CORPPA_SRC_DIR.parent / "corppa_config.yml"
#: expected path for example config file
SAMPLE_CONFIG_PATH = CORPPA_SRC_DIR.parent / "sample_config.yml"


@dataclass
class CorpusConfig:
    name: str
    base_dir: Optional[Path] = None
    text_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    relative_dir: Optional[Path] = None

    _path_suffix: ClassVar[dict[str, str]] = {
        "text": "_texts.tar.gz",
        "metadata": ".csv",
    }

    def __post_init__(self):
        # set paths based on defaults for any paths not passed in
        if self.base_dir is None:
            self.base_dir = Path(self.name)
        # optionally make base dir relative to another dir
        # (most important for case where default base dir is inferred from name)
        if self.relative_dir is not None and not self.base_dir.is_relative_to(
            self.relative_dir
        ):
            self.base_dir = self.relative_dir / self.base_dir
        # if not set, use the default path
        if self.text_path is None:
            self.text_path = self.base_dir / f"{self.name}{self._path_suffix['text']}"
        else:
            # allow passing as config to simplify optional config logic
            if isinstance(self.text_path, str):
                self.text_path = Path(self.text_path)
            if not self.text_path.is_absolute() and not self.text_path.is_relative_to(
                self.base_dir
            ):
                # if set, make path is relative to base dir
                self.text_path = self.base_dir / self.text_path

        if self.metadata_path is None:
            self.metadata_path = (
                self.base_dir / f"{self.name}{self._path_suffix['metadata']}"
            )
        else:
            if isinstance(self.metadata_path, str):
                self.metadata_path = Path(self.metadata_path)
            if (
                not self.metadata_path.is_absolute()
                and not self.metadata_path.is_relative_to(self.base_dir)
            ):
                self.metadata_path = self.base_dir / self.metadata_path


@dataclass
class PPACorpusConfig(CorpusConfig):
    # subclass for ppa corpus config; which has known filenames
    name: str = "ppa"
    _path_suffix: ClassVar[dict[str, str]] = {
        "text": "_pages.jsonl.gz",
        "metadata": "_metadata.csv",
    }


@dataclass
class ConfigOpts:
    base_dir: Path
    compiled_dataset_dir: Path
    ppa_corpus: Optional[PPACorpusConfig] = None
    reference_corpora: dict[str, CorpusConfig] = field(default_factory=dict)  # type: ignore[arg-type]
    excerpt_data_dir: Optional[Path] = None
    poem_clusters_path: Optional[str] = (
        None  # currently expect a url rather than local path
    )

    def __post_init__(self):
        # convert string to path to simplify optional config handling
        if self.excerpt_data_dir is not None:
            if isinstance(self.excerpt_data_dir, str):
                self.excerpt_data_dir = Path(self.excerpt_data_dir)
            if (
                not self.excerpt_data_dir.is_absolute()
                and not self.excerpt_data_dir.is_relative_to(self.base_dir)
            ):
                self.excerpt_data_dir = self.base_dir / self.excerpt_data_dir

        if self.compiled_dataset_dir is not None:
            if isinstance(self.compiled_dataset_dir, str):
                self.compiled_dataset_dir = Path(self.compiled_dataset_dir)
            if (
                not self.compiled_dataset_dir.is_absolute()
                and not self.compiled_dataset_dir.is_relative_to(self.base_dir)
            ):
                self.compiled_dataset_dir = self.base_dir / self.compiled_dataset_dir


# assume defaults
# - ppa data standard file names are known
# - config logic should be in one place
# - use path objects
# .  - make relative
# .  - validate
# - simple: list / dict defaults (ref corpora)
#   - easy to add in future (follows pattern)


def get_config():
    # if the config file is not in place
    if not CORPPA_CONFIG_PATH.exists():
        not_found_msg = (
            "Config file not found.\n"
            + f"Copy {SAMPLE_CONFIG_PATH} to {CORPPA_CONFIG_PATH} and configure for your environment."
        )
        raise SystemExit(not_found_msg)

    with CORPPA_CONFIG_PATH.open() as cfg_file:
        try:
            # configuration in the yaml file should override any defaults
            config_values = yaml.load(cfg_file, Loader=Loader)
        except yaml.parser.ParserError as err:
            raise SystemExit(f"Error parsing config file: {err}")

    try:
        base_dir = Path(config_values["base_dir"])
        ref_corpus_configs = {}
        # allow ref corpora config to be optional
        if "reference_corpora" in config_values:
            ref_corpus_base_dir = config_values["reference_corpora"].get("base_dir")
            if ref_corpus_base_dir is not None:
                ref_corpus_base_dir = Path(ref_corpus_base_dir)
                if (
                    not ref_corpus_base_dir.is_absolute()
                    and not ref_corpus_base_dir.is_relative_to(base_dir)
                ):
                    ref_corpus_base_dir = base_dir / ref_corpus_base_dir

                # remove base_dir from dict before iterating over sections
                del config_values["reference_corpora"]["base_dir"]
            else:
                # if now ref-corpus base dir is specified, use top-level as base dir
                ref_corpus_base_dir = base_dir

            for section, values in config_values["reference_corpora"].items():
                # when section is empty, values is None; convert to empty dict
                if values is None:
                    values = {}

                # if base dir is set for this corpus, use it;
                # make relative to ref corpus base dir when set, unless absolute
                section_base_dir = values.get("base_dir")
                if section_base_dir is not None:
                    section_base_dir = Path(section_base_dir)
                    if (
                        ref_corpus_base_dir is not None
                        and not section_base_dir.is_absolute()
                    ):
                        section_base_dir = ref_corpus_base_dir / section_base_dir

                # FIXME: default needs an optional relative to base path here
                ref_corpus_configs[section] = CorpusConfig(
                    name=section,
                    base_dir=section_base_dir,
                    text_path=values.get("text_path"),
                    metadata_path=values.get("metadata_path"),
                    relative_dir=ref_corpus_base_dir,
                )
        # allow ppa corpus to be optional
        ppa_corpus = None
        if "ppa_corpus" in config_values:
            ppa_corpus = PPACorpusConfig(
                base_dir=Path(config_values["ppa_corpus"]["base_dir"])
            )

        # use direct access for required values to trigger a KeyError

        return ConfigOpts(
            base_dir=base_dir,
            compiled_dataset_dir=Path(config_values["compiled_dataset_dir"]),
            ppa_corpus=ppa_corpus,
            reference_corpora=ref_corpus_configs,
            excerpt_data_dir=config_values.get("excerpt_data_dir"),
            poem_clusters_path=config_values.get("poem_clusters_path"),
        )
    except KeyError as err:
        raise SystemExit(
            f"Config file is missing required configuration: {err.args[0]}"
        )
