"""
Load local configuration options
"""

import pathlib

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:  # pragma: no cover
    from yaml import Loader  # pragma: no cover

#: src dir relative to this file (assuming dev environment for now)
CORPPA_SRC_DIR = pathlib.Path(__file__).parent.parent.absolute()

#: expected path for local config file (non-versioned)
CORPPA_CONFIG_PATH = CORPPA_SRC_DIR.parent / "corppa_config.yml"
#: expected path for example config file
SAMPLE_CONFIG_PATH = CORPPA_SRC_DIR.parent / "sample_config.yml"


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
            return yaml.load(cfg_file, Loader=Loader)
        except yaml.parser.ParserError as err:
            raise SystemExit(f"Error parsing config file: {err}")
