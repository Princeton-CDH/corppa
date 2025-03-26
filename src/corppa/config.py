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

#: default configuration; anything in yaml config will override
DEFAULTS = {
    "reference_corpora": {
        # base_dir path is relative to top-level data_ingredients_dir
        "base_dir": "ref-corpora",
        # paths are relative to base_dir
        "internet_poems": {
            # tarball of directory of text files OR expanded directory
            "text_dir": "internet_poems/internet_poems_texts.tar.gz"
        },
        "chadwyck-healey": {
            "text_dir": "chadwyck-healey/chadwyck-healey_texts.tar.gz",
            "metadata_path": "chadwyck-healey/chadwyck-healey.csv",
        },
        # other poems metadata_path configuration required
    }
}


def merge_dicts(initial_values: dict, overrides: dict) -> dict:
    """Recursively merge two dictionaries. Anything values in second dict
    take precedence over the first. Nested dictionaries are merged
    by calling :meth:`merge_dicts`.
    """
    merged_dict = initial_values.copy()
    for key, value in overrides.items():
        # if the override value is a dict and key is present in both,
        # then merge the nested dicts
        if isinstance(value, dict) and key in merged_dict:
            # note: assumes they are both dicts; this should be safe
            # for our config file setup
            merged_dict[key] = merge_dicts(initial_values[key], overrides[key])
        # otherwise, set the value from the override dictionary
        else:
            merged_dict[key] = value

    return merged_dict


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
            config_overrides = yaml.load(cfg_file, Loader=Loader)
        except yaml.parser.ParserError as err:
            raise SystemExit(f"Error parsing config file: {err}")

    # use default config as the starting point; settings loaded from the config
    # file should override
    return merge_dicts(DEFAULTS, config_overrides)
