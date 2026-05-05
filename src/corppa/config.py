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


def resolve_path(path: str | Path | None, base_dir: Path) -> Path | str | None:
    """Convert to Path and make relative to base_dir if not absolute."""
    if path is None:
        return None
    if isinstance(path, str):
        # check for URL; don't convert to path but keep as-is
        if path.startswith("http"):
            return path
        # otherwise, convert string to path and make relative
        path = Path(path)
    if not path.is_absolute() and not path.is_relative_to(base_dir):
        path = base_dir / path
    return path


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
        if self.base_dir is None:
            self.base_dir = Path(self.name)
        # optionally make base dir relative to another dir
        # (most important for case where default base dir is inferred from name)
        if self.relative_dir is not None:
            self.base_dir = resolve_path(self.base_dir, self.relative_dir)

        # if paths are not set, use name and default suffix;
        # make paths relative to base dir
        if self.text_path is None:
            self.text_path = f"{self.name}{self._path_suffix['text']}"
        self.text_path = resolve_path(self.text_path, self.base_dir)

        if self.metadata_path is None:
            self.metadata_path = f"{self.name}{self._path_suffix['metadata']}"
        self.metadata_path = resolve_path(self.metadata_path, self.base_dir)

    def validate(self, text=True, metadata=True) -> bool:
        if text:
            if not self.text_path.exists():
                raise ValueError(
                    f"Configuration error: {self.name} path {self.text_path} does not exist"
                )
            # Currently supports directory and tar.gz file
            if not self.text_path.is_dir() and not (
                self.text_path.is_file() and self.text_path.name.endswith(".tar.gz")
            ):
                raise ValueError(
                    f"Configuration error: {self.name} path {self.text_path} is not a directory or a tar.gz"
                )
        if metadata:
            if (
                isinstance(self.metadata_path, Path)
                and not self.metadata_path.is_file()
            ):
                raise ValueError(
                    f"Configuration error: {self.name} metadata {self.metadata_path} does not exist"
                )

        return True


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
        self.excerpt_data_dir = resolve_path(self.excerpt_data_dir, self.base_dir)
        self.compiled_dataset_dir = resolve_path(
            self.compiled_dataset_dir, self.base_dir
        )


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
        # use direct access for required values to trigger a KeyError
        base_dir = Path(config_values["base_dir"])
        ref_corpus_configs = {}
        # allow ref corpora config to be optional
        if "reference_corpora" in config_values:
            ref_corpus_base_dir = (
                resolve_path(
                    config_values["reference_corpora"].get("base_dir"), base_dir
                )
                or base_dir
            )
            if "base_dir" in config_values["reference_corpora"]:
                # remove base_dir from dict before iterating over sections
                del config_values["reference_corpora"]["base_dir"]

            for section, values in config_values["reference_corpora"].items():
                # when section is empty, values is None; convert to empty dict
                if values is None:
                    values = {}

                section_base_dir = resolve_path(
                    values.get("base_dir"), ref_corpus_base_dir
                )
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
