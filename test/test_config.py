# Copyright (c) 2024-2025, Center for Digital Humanities, Princeton University
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from corppa import config


def test_get_config_not_found(tmp_path):
    test_config = tmp_path / "test.cfg"
    # error should include directions about how to fix the problem
    expected_error_msg = (
        "Config file not found.\n"
        + f"Copy .*{config.SAMPLE_CONFIG_PATH.name} to .*{test_config.name} and configure for your environment."
    )
    with patch.object(config, "CORPPA_CONFIG_PATH", new=test_config):
        with pytest.raises(SystemExit, match=expected_error_msg):
            config.get_config()


def test_get_config_parse_error(tmp_path):
    test_config = tmp_path / "test.cfg"
    # config in non-yaml format
    test_config.write_text("""[poem_dataset]
data_dir=/tmp/p-p-poems/data
""")
    with patch.object(config, "CORPPA_CONFIG_PATH", new=test_config):
        with pytest.raises(SystemExit, match="Error parsing config file"):
            config.get_config()


def test_get_config(tmp_path):
    # create a test config file with one section and one value
    test_config = tmp_path / "test.cfg"
    test_config.write_text("""
# local path to compiled poem dataset files
poem_dataset:
  data_dir: "/tmp/p-p-poems/data"
""")
    # use patch to override the config path and load our test file
    with patch.object(config, "CORPPA_CONFIG_PATH", new=test_config):
        config_opts = config.get_config()
        assert "poem_dataset" in config_opts
        assert config_opts["poem_dataset"]["data_dir"] == "/tmp/p-p-poems/data"
