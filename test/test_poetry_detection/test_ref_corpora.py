import pathlib
from collections.abc import Generator
from unittest.mock import patch

import polars as pl
import pytest

from corppa import config
from corppa.poetry_detection.ref_corpora import BaseReferenceCorpus, InternetPoems


@pytest.fixture
def corppa_test_config(tmp_path):
    test_config = tmp_path / "test_config.yml"
    test_config.write_text(f"""
    # local path to compiled poem dataset files
    reference_corpora:
        dir: {tmp_path / "ref-corpora"}
        internet_poems:
            data_path: {tmp_path / "ref-corpora" / "internet_poems"}
        chadwyck-healey:
            data_path: {tmp_path / "ref-corpora" / "chadwyck-healey"}
            metadata_path: {tmp_path / "ref-corpora" / "chadwyck-healey" / "chadwyck-healey.csv"}
        other:
            metadata_url: http://example.com/other-poems.csv
    """)
    with patch.object(config, "CORPPA_CONFIG_PATH", new=test_config):
        yield test_config


class TestBaseReferenceCorpus:
    def test_not_implemented(self):
        with pytest.raises(NotImplementedError):
            BaseReferenceCorpus().get_metadata_df()

        with pytest.raises(NotImplementedError):
            BaseReferenceCorpus().get_text_corpus()


class TestInternetPoems:
    sample_texts = [
        {
            "id": "King-James-Bible_Psalms",
            "text": "He hath made his wonderful works to be remembered",
        },
        {
            "id": "Robert-Burns_Mary",
            "text": "Powers celestial! whose protection Ever guards the virtuous fair,",
        },
    ]

    def test_init(self, tmp_path, corppa_test_config):
        # path in test config doesn't exist
        with pytest.raises(ValueError, match="not configured correctly"):
            InternetPoems()

        config_opts = config.get_config()

        # create expected data_dir
        expected_data_dir = pathlib.Path(
            config_opts["reference_corpora"]["internet_poems"]["data_path"]
        )
        expected_data_dir.mkdir(parents=True)

        internet_poems = InternetPoems()
        assert isinstance(internet_poems.data_path, pathlib.Path)
        assert internet_poems.data_path == expected_data_dir

    @patch.object(InternetPoems, "get_config_opts")
    def test_get_metadata_df(self, mock_get_config_opts, tmp_path):
        data_dir = tmp_path / "internet_poems"
        data_dir.mkdir()

        for sample in self.sample_texts:
            text_file = data_dir / f"{sample['id']}.txt"
            text_file.write_text(sample["text"])

        mock_get_config_opts.return_value = {"data_path": str(data_dir)}
        internet_poems = InternetPoems()
        meta_df = internet_poems.get_metadata_df()
        assert isinstance(meta_df, pl.DataFrame)
        assert meta_df.height == len(self.sample_texts)
        # get the first row as a dict; sort by id so order matches input
        meta_row = meta_df.sort("poem_id").row(0, named=True)
        assert meta_row["poem_id"] == self.sample_texts[0]["id"]
        assert meta_row["author"] == "King James Bible"
        assert meta_row["title"] == "Psalms"
        assert meta_row["ref_corpus"] == internet_poems.corpus_id

    @patch.object(InternetPoems, "get_config_opts")
    def test_get_text_corpus(self, mock_get_config_opts, tmp_path):
        data_dir = tmp_path / "internet_poems"
        data_dir.mkdir()

        for sample in self.sample_texts:
            text_file = data_dir / f"{sample['id']}.txt"
            text_file.write_text(sample["text"])

        mock_get_config_opts.return_value = {"data_path": str(data_dir)}
        internet_poems = InternetPoems()
        text_data = internet_poems.get_text_corpus()
        assert isinstance(text_data, Generator)
        # turn the generator into a list; sort by id so order matches input
        text_data = sorted(text_data, key=lambda x: x["poem_id"])
        assert len(text_data) == len(self.sample_texts)
        assert text_data[0]["poem_id"] == self.sample_texts[0]["id"]
        assert text_data[0]["text"] == self.sample_texts[0]["text"]
