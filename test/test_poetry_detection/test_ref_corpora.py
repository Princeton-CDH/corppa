import pathlib
from collections.abc import Generator
from unittest.mock import patch

import polars as pl
import pytest

from corppa import config
from corppa.poetry_detection.ref_corpora import (
    METADATA_SCHEMA,
    BaseReferenceCorpus,
    ChadwyckHealey,
    InternetPoems,
    OtherPoems,
    all_corpora,
    compile_metadata_df,
    fulltext_corpora,
)


@pytest.fixture
def corppa_test_config(tmp_path):
    # test fixture to create and use a temporary config file
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


# fixture data for internet poems
INTERNETPOEMS_TEXTS = [
    {
        "id": "King-James-Bible_Psalms",
        "text": "He hath made his wonderful works to be remembered",
    },
    {
        "id": "Robert-Burns_Mary",
        "text": "Powers celestial! whose protection Ever guards the virtuous fair,",
    },
]


@pytest.fixture
def internetpoems_data_dir(tmp_path):
    # test fixture to create internet poems data directory with sample text files
    data_dir = tmp_path / "ref-corpora" / "internet_poems"
    data_dir.mkdir(parents=True, exist_ok=True)
    for sample in INTERNETPOEMS_TEXTS:
        text_file = data_dir / f"{sample['id']}.txt"
        text_file.write_text(sample["text"])
    return data_dir


class TestInternetPoems:
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
    def test_get_metadata_df(
        self, mock_get_config_opts, tmp_path, internetpoems_data_dir
    ):
        mock_get_config_opts.return_value = {"data_path": str(internetpoems_data_dir)}
        internet_poems = InternetPoems()
        meta_df = internet_poems.get_metadata_df()
        assert isinstance(meta_df, pl.DataFrame)
        assert meta_df.schema == METADATA_SCHEMA
        assert meta_df.height == len(INTERNETPOEMS_TEXTS)
        # get the first row as a dict; sort by id so order matches input
        meta_row = meta_df.sort("poem_id").row(0, named=True)
        assert meta_row["poem_id"] == INTERNETPOEMS_TEXTS[0]["id"]
        assert meta_row["author"] == "King James Bible"
        assert meta_row["title"] == "Psalms"
        assert meta_row["ref_corpus"] == internet_poems.corpus_id

    @patch.object(InternetPoems, "get_config_opts")
    def test_get_text_corpus(
        self, mock_get_config_opts, tmp_path, internetpoems_data_dir
    ):
        mock_get_config_opts.return_value = {"data_path": str(internetpoems_data_dir)}
        internet_poems = InternetPoems()
        text_data = internet_poems.get_text_corpus()
        assert isinstance(text_data, Generator)
        # turn the generator into a list; sort by id so order matches input
        text_data = sorted(text_data, key=lambda x: x["poem_id"])
        assert len(text_data) == len(INTERNETPOEMS_TEXTS)
        assert text_data[0]["poem_id"] == INTERNETPOEMS_TEXTS[0]["id"]
        assert text_data[0]["text"] == INTERNETPOEMS_TEXTS[0]["text"]


@pytest.fixture
def chadwyck_healey_csv(tmp_path):
    "fixture to create a test version of the chadwyck-healey metadata csv file"
    # test fixture to create internet poems data directory with sample text files
    # TODO: move these defaults to the class
    data_dir = tmp_path / "ref-corpora" / "chadwyck-healey"
    data_dir.mkdir(parents=True, exist_ok=True)
    ch_meta_csv = data_dir / "chadwyck-healey.csv"
    ch_meta_csv.write_text("""id,author_lastname,author_firstname,author_birth,author_death,author_period,transl_lastname,transl_firstname,transl_birth,transl_death,title_id,title_main,title_sub,edition_id,edition_text,period,genre,rhymes
Z300475611,Robinson,Mary,1758,1800,,,,,,Z300475611,THE CAVERN OF WOE.,,Z000475579,The Poetical Works (1806),Later Eighteenth-Century 1750-1799,,y""")
    return ch_meta_csv


class TestChadwyckHealey:
    @patch.object(ChadwyckHealey, "get_config_opts")
    def test_get_metadata_df(self, mock_get_config_opts, tmp_path, chadwyck_healey_csv):
        # data path is currently required even though not used in this test
        data_dir = tmp_path / "chadwyck-healey_texts"
        data_dir.mkdir()
        mock_get_config_opts.return_value = {
            "metadata_path": str(chadwyck_healey_csv),
            "data_path": str(data_dir),
        }
        chadwyck_healey = ChadwyckHealey()
        meta_df = chadwyck_healey.get_metadata_df()
        assert isinstance(meta_df, pl.DataFrame)
        assert meta_df.schema == METADATA_SCHEMA
        # csv fixture data currently has one row
        assert meta_df.height == 1
        # get the first row as a dict and check values
        meta_row = meta_df.row(0, named=True)
        assert meta_row["poem_id"] == "Z300475611"
        assert meta_row["author"] == "Mary Robinson"
        assert meta_row["title"] == "THE CAVERN OF WOE."
        assert meta_row["ref_corpus"] == chadwyck_healey.corpus_id

    # get_text_corpus method is not tested here because it is inherited;
    # logic is shared with InternetPoems and tested there


# text fixture data for other poems corpus
OTHERPOEM_METADATA = [
    # poem ids
    ["Joseph-Addison_Cato", "John-Ogilvie_Ode-to-Time", "John-Dryden_Amphitryon"],
    # authors
    ["Joseph Addison", "John Ogilvie", "John Dryden"],
    # titles
    ["Cato", "Ode to Time", "Amphitryon"],
]


@pytest.fixture
def otherpoems_metadata_df():
    # create and return polars dataframe from fixture data above
    # does NOT include ref_corpus field, to simulate other poem spreadsheet
    return pl.from_records(OTHERPOEM_METADATA, schema=["poem_id", "author", "title"])


class TestOtherPoems:
    @patch("corppa.poetry_detection.ref_corpora.pl.read_csv")
    def test_get_metadata_df(
        self, mock_pl_read_csv, corppa_test_config, otherpoems_metadata_df
    ):
        mock_pl_read_csv.return_value = otherpoems_metadata_df
        opoems = OtherPoems()
        meta_df = opoems.get_metadata_df()
        assert isinstance(meta_df, pl.DataFrame)
        assert meta_df.schema == METADATA_SCHEMA
        assert meta_df.height == len(OTHERPOEM_METADATA)
        # check values on the first row
        meta_row = meta_df.row(0, named=True)
        assert meta_row["poem_id"] == OTHERPOEM_METADATA[0][0]
        assert meta_row["author"] == OTHERPOEM_METADATA[1][0]
        assert meta_row["title"] == OTHERPOEM_METADATA[2][0]
        assert meta_row["ref_corpus"] == opoems.corpus_id

        mock_pl_read_csv.assert_called_with(opoems.metadata_url, schema=METADATA_SCHEMA)


def test_all_corpora():
    all_ref_corpora = all_corpora()
    assert all(
        isinstance(ref_corpus, BaseReferenceCorpus) for ref_corpus in all_ref_corpora
    )
    corpus_classes = [ref_corpus.__class__ for ref_corpus in all_ref_corpora]
    # order indicates priority, so check both presence and order
    assert corpus_classes == [InternetPoems, ChadwyckHealey, OtherPoems]


def test_fulltext_corpora():
    fulltext_ref_corpora = fulltext_corpora()
    assert all(
        isinstance(ref_corpus, BaseReferenceCorpus)
        for ref_corpus in fulltext_ref_corpora
    )
    corpus_classes = [ref_corpus.__class__ for ref_corpus in fulltext_ref_corpora]
    # other poems is currently our only metadata-only reference corpus
    assert OtherPoems not in corpus_classes


def test_compile_metadata_df(
    tmp_path,
    corppa_test_config,
    internetpoems_data_dir,
    chadwyck_healey_csv,
    otherpoems_metadata_df,
):
    # data fixtures should ensure that all the expected directories exist

    # add corpus id to other poems data frame and patch it to be returned
    otherpoems_metadata_df = otherpoems_metadata_df.with_columns(
        ref_corpus=pl.lit(OtherPoems.corpus_id)
    )
    with patch.object(
        OtherPoems, "get_metadata_df", return_value=otherpoems_metadata_df
    ):
        compiled_metadata = compile_metadata_df()

    assert isinstance(compiled_metadata, pl.DataFrame)
    assert compiled_metadata.schema == METADATA_SCHEMA
    assert (
        compiled_metadata.height
        == len(INTERNETPOEMS_TEXTS) + len(OTHERPOEM_METADATA) + 1
    )
    assert set(compiled_metadata["ref_corpus"].unique().to_list()) == {
        InternetPoems.corpus_id,
        ChadwyckHealey.corpus_id,
        OtherPoems.corpus_id,
    }
