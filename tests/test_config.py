# Copyright (c) 2024-2026, Center for Digital Humanities, Princeton University
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
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


def test_get_config_missing_required(tmp_path):
    test_config = tmp_path / "test.cfg"
    test_config.write_text("foo: bar")
    with patch.object(config, "CORPPA_CONFIG_PATH", new=test_config):
        with pytest.raises(
            SystemExit, match="missing required configuration: base_dir"
        ):
            config.get_config()

        # if first required field is present, should error on the next one
        test_config.write_text("base_dir: /tmp/data/")
        with pytest.raises(
            SystemExit, match="missing required configuration: compiled_dataset_dir"
        ):
            config.get_config()


def test_get_config(tmp_path):
    # create a test config file with one section and one value
    test_config = tmp_path / "test.cfg"
    test_config.write_text("""
base_dir: data/
# local path to compiled poem dataset files
compiled_dataset_dir: "/tmp/p-p-poems/data"
""")
    # use patch to override the config path and load our test file
    with patch.object(config, "CORPPA_CONFIG_PATH", new=test_config):
        config_opts = config.get_config()
        assert isinstance(config_opts, config.ConfigOpts)
        assert config_opts.base_dir == Path("data")
        assert config_opts.compiled_dataset_dir == Path("/tmp/p-p-poems/data")
        # other items are unset
        assert config_opts.ppa_corpus is None
        assert config_opts.reference_corpora == {}


def test_get_config_defaults(tmp_path):
    # create a test config file with one section and one value
    test_config = tmp_path / "test.cfg"
    # override one portion of a nested config
    override_text_dir = "/ch/text.tar.gz"
    test_config.write_text(f"""
base_dir: data/
compiled_dataset_dir: found-poems/
# local path to compiled poem dataset files
reference_corpora:
    chadwyck-healey:
        text_path: "{override_text_dir}"
""")
    # use patch to override the config path and load our test file
    with patch.object(config, "CORPPA_CONFIG_PATH", new=test_config):
        config_opts = config.get_config()
        assert len(config_opts.reference_corpora) == 1
        ch_config = config_opts.reference_corpora["chadwyck-healey"]
        assert ch_config.text_path == Path(override_text_dir)


def test_get_config_maximal(tmp_path):
    # create a test config file with one section and one value
    test_config = tmp_path / "test.cfg"
    test_config.write_text("""
base_dir: data/
compiled_dataset_dir: found-poems/
excerpt_data_dir: excerpt-data/
poem_clusters_path: http://example.com/poem_groups.csv
reference_corpora:
    chadwyck-healey:
    internet_poems:
    other_poems:
""")
    # use patch to override the config path and load our test file
    with patch.object(config, "CORPPA_CONFIG_PATH", new=test_config):
        config_opts = config.get_config()
        assert config_opts.compiled_dataset_dir == Path("data/found-poems")
        assert config_opts.poem_clusters_path == "http://example.com/poem_groups.csv"
        assert len(config_opts.reference_corpora) == 3
        ref_corpora_names = [
            "chadwyck-healey",
            "internet_poems",
            "other_poems",
        ]
        assert list(config_opts.reference_corpora.keys()) == ref_corpora_names
        ref_corpus_configs = config_opts.reference_corpora.values()
        assert [rc.name for rc in ref_corpus_configs] == ref_corpora_names
        # all ref-corpus base directories should be relative to base dir
        assert all(
            rc.base_dir.is_relative_to(config_opts.base_dir)
            for rc in ref_corpus_configs
        )


## test module-level helpers


class TestResolvePath:
    base = Path("/data/corpora")

    def test_none_returns_none(self):
        assert config.resolve_path(None, self.base) is None

    def test_string_converted_to_path(self):
        result = config.resolve_path("subdir/file.txt", self.base)
        assert isinstance(result, Path)
        assert result == self.base / "subdir/file.txt"

    def test_absolute_path_unchanged(self):
        abs_path = Path("/other/location/file.csv")
        assert config.resolve_path(abs_path, self.base) == abs_path

    def test_relative_path_prefixed_with_base(self):
        result = config.resolve_path(Path("corpus/texts.tar.gz"), self.base)
        assert result == self.base / "corpus/texts.tar.gz"

    def test_already_relative_to_base_unchanged(self):
        already_relative = self.base / "corpus/texts.tar.gz"
        assert config.resolve_path(already_relative, self.base) == already_relative


## test dataclass config objects


class TestCorpusConfig:
    def test_init_defaults(self):
        corpus_name = "internet_poems"
        # specifying corpus name is enough to set defaults for the rest
        ref_corpus = config.CorpusConfig(name=corpus_name)
        assert ref_corpus.name == corpus_name
        assert ref_corpus.base_dir == Path(corpus_name)
        # default text file suffix
        assert (
            ref_corpus.text_path == ref_corpus.base_dir / f"{corpus_name}_texts.tar.gz"
        )
        # default metadata suffix
        assert ref_corpus.metadata_path == ref_corpus.base_dir / f"{corpus_name}.csv"

    def test_init_override(self):
        corpus_name = "other"
        # specifying corpus name is enough to set defaults for the rest
        ref_corpus = config.CorpusConfig(
            name=corpus_name,
            base_dir=Path("data/foo"),
            text_path=Path("my_texts.tar.gz"),
            metadata_path=Path("data.csv"),
        )
        assert ref_corpus.name == corpus_name
        assert ref_corpus.base_dir == Path("data/foo")
        # provided paths are now relative to base dir
        assert ref_corpus.text_path == ref_corpus.base_dir / "my_texts.tar.gz"
        assert ref_corpus.metadata_path == ref_corpus.base_dir / "data.csv"

        # test absolute path is not changed
        abs_path = Path("/path/to/my_texts.tar.gz")
        ref_corpus = config.CorpusConfig(name=corpus_name, text_path=abs_path)
        assert ref_corpus.text_path == abs_path

    def test_subclass(self):
        base_dir = Path("ppa_corpus-2026-01-03")
        ppa_corpus = config.PPACorpusConfig(base_dir=base_dir)
        assert ppa_corpus.name == "ppa"
        assert ppa_corpus.base_dir == base_dir
        # default file names
        assert ppa_corpus.text_path == base_dir / "ppa_pages.jsonl.gz"
        assert ppa_corpus.metadata_path == base_dir / "ppa_metadata.csv"

    def test_vaidate(self, tmp_path):
        # specifying corpus name is enough to set defaults for the rest
        ref_corpus = config.CorpusConfig(
            name="test",
            base_dir=tmp_path / "foo",
        )
        # neither text nor metadata exists
        with pytest.raises(ValueError):
            ref_corpus.validate()

        # text file only
        ref_corpus.base_dir.mkdir()
        ref_corpus.text_path.touch()
        # text-only validation should pass
        assert ref_corpus.validate(metadata=False)
        # text and metadata should fail
        with pytest.raises(ValueError):
            ref_corpus.validate()

        # both text and metadata files present
        ref_corpus.metadata_path.touch()
        assert ref_corpus.validate()

        # metadata only
        ref_corpus.text_path.unlink()
        # metadata only should pass
        assert ref_corpus.validate(text=False)
        # text and metadata should fail
        with pytest.raises(ValueError):
            ref_corpus.validate()

        # wrong kind of text file
        ref_corpus.text_path = ref_corpus.base_dir / "text_files.zip"
        ref_corpus.text_path.touch()
        with pytest.raises(ValueError):
            ref_corpus.validate()
