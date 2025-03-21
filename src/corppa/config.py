"""
Load local configuration options
"""

import configparser
import pathlib

#: src dir relative to this file (assuming dev environment for now)
CORPPA_SRC_DIR = pathlib.Path(__file__).parent.parent.absolute()

#: expected path for local config file (non-versioned)
CORPPA_CONFIG_PATH = CORPPA_SRC_DIR.parent / "corppa.cfg"
#: expected path for example config file
SAMPLE_CONFIG_PATH = CORPPA_SRC_DIR.parent / "sample.cfg"


def get_config():
    # if the config file is not in place
    if not CORPPA_CONFIG_PATH.exists():
        not_found_msg = (
            "Config file not found.\n"
            + f"Copy {SAMPLE_CONFIG_PATH} to {CORPPA_CONFIG_PATH} and configure for your environment."
        )
        raise SystemExit(not_found_msg)

    config = configparser.ConfigParser()
    config.read(CORPPA_CONFIG_PATH)
    return config
