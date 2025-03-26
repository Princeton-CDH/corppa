from unittest.mock import patch

import pytest

from corppa import config


def test_merge_dicts():
    # no nesting, no conflict
    assert config.merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    # no nesting; second overrides first
    assert config.merge_dicts({"a": 1}, {"a": 2}) == {"a": 2}
    # nesting, no conflict
    assert config.merge_dicts({"a": {"b": 1}}, {"a": {"c": 2}}) == {
        "a": {"b": 1, "c": 2}
    }
    # nesting, with override
    assert config.merge_dicts({"a": {"b": 1}}, {"a": {"b": 2}}) == {"a": {"b": 2}}


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
compiled_dataset:
  text_dir: "/tmp/p-p-poems/data"
""")
    # use patch to override the config path and load our test file
    with patch.object(config, "CORPPA_CONFIG_PATH", new=test_config):
        config_opts = config.get_config()
        assert "compiled_dataset" in config_opts
        assert config_opts["compiled_dataset"]["text_dir"] == "/tmp/p-p-poems/data"


def test_get_config_defaults(tmp_path):
    # create a test config file with one section and one value
    test_config = tmp_path / "test.cfg"
    # override one portion of a nested config
    override_text_dir = "/ch/text.tar.gz"
    test_config.write_text(f"""
# local path to compiled poem dataset files
reference_corpora:
    chadwyck-healey:
        text_dir: "{override_text_dir}"
""")
    # use patch to override the config path and load our test file
    with patch.object(config, "CORPPA_CONFIG_PATH", new=test_config):
        config_opts = config.get_config()
        # override value should be used
        chadwyck_healey_config = config_opts["reference_corpora"]["chadwyck-healey"]
        assert chadwyck_healey_config["text_dir"] == override_text_dir
        # default value in the same section that is not specified should be present
        assert (
            chadwyck_healey_config["metadata_path"]
            == config.DEFAULTS["reference_corpora"]["chadwyck-healey"]["metadata_path"]
        )
