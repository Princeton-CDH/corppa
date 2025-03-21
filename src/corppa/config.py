"""
Load local configuration options
"""

import configparser
import pathlib

#: src dir relative to this file (assumeing dev environment for now)
CORPPA_SRC_DIR = pathlib.Path(__file__).parent.absolute()

#: expected patch for local config file (non-versioned)
CORPPA_CONFIG_PATH = CORPPA_SRC_DIR.parent / "corppa.cfg"
#: expected patch for example config file
SAMPLE_CONFIG_PATH = CORPPA_SRC_DIR.parent / "sample.cfg"

CFG_NOT_FOUND_MESSAGE = f""""Config file not found.
Copy {SAMPLE_CONFIG_PATH} to {CORPPA_CONFIG_PATH} and configure for your environment.
"""


def get_config():
    # if the config file is not in place
    if not CORPPA_CONFIG_PATH.exists():
        raise SystemExit(CFG_NOT_FOUND_MESSAGE)

    config = configparser.ConfigParser()
    config.read(CORPPA_CONFIG_PATH)
    return config
