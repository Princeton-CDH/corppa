from unittest.mock import patch

import pytest

from corppa import config


def test_get_config_not_found(tmp_path):
    with pytest.raises(SystemExit, match="Config file not found") as err:
        config.get_config()
    # error should include directions about how to fix the problem
    help_msg = f"Copy {config.SAMPLE_CONFIG_PATH} to {config.CORPPA_CONFIG_PATH} and configure for your environment."
    assert help_msg in str(err)


def test_get_config(tmp_path):
    # create a test config file with one section and one value
    test_config = tmp_path / "test.cfg"
    test_config.write_text("""
# local path to compiled poem dataset files        
[poem_dataset]
data_dir=/tmp/p-p-poems/data
""")
    # use patch to override the config path and load our test file
    with patch.object(config, "CORPPA_CONFIG_PATH", new=test_config):
        config_opts = config.get_config()
        assert "poem_dataset" in config_opts.sections()
        assert config_opts["poem_dataset"]["data_dir"] == "/tmp/p-p-poems/data"
