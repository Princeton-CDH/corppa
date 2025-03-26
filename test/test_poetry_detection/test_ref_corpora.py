import pathlib
import tarfile
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
    save_poem_metadata,
)


@pytest.fixture
def corppa_test_config(tmp_path):
    # test fixture to create and use a temporary config file
    # uses explicit, non-default paths
    compiled_dataset_dir = tmp_path / "found-poems-data"
    test_config = tmp_path / "test_config.yml"
    base_dir = tmp_path / "ref-corpora"
    test_config.write_text(f"""
    # local path to compiled poem dataset files
    compiled_dataset:    
        output_data_dir : {compiled_dataset_dir}
    reference_corpora:
        base_dir: {base_dir}
        internet_poems:
            text_dir: {base_dir / "internet_poems2"}
        chadwyck-healey:
            text_dir: "ch"
            metadata_path: "ch/chadwyck-healey.csv"
        other:
            metadata_path: http://example.com/other-poems.csv
    """)
    with patch.object(config, "CORPPA_CONFIG_PATH", new=test_config):
        yield test_config


@pytest.fixture
def corppa_test_config_defaults(tmp_path):
    # test fixture with a minimal reference corpus config file
    test_config = tmp_path / "test_config.yml"
    compiled_dataset_dir = tmp_path / "found-poems-data"
    base_dir = tmp_path / "ref-corpora"
    test_config.write_text(f"""
    # local path to compiled poem dataset files
    compiled_dataset:    
        output_data_dir : {compiled_dataset_dir}
    reference_corpora:
        base_dir: {base_dir}
        other:
            metadata_path: http://example.com/other-poems.csv
    """)
    with patch.object(config, "CORPPA_CONFIG_PATH", new=test_config):
        yield test_config


class TestBaseReferenceCorpus:
    def test_not_implemented(self):
        with pytest.raises(NotImplementedError):
            BaseReferenceCorpus().get_metadata_df()

        with pytest.raises(NotImplementedError):
            BaseReferenceCorpus().get_text_corpus()

    @patch("corppa.poetry_detection.ref_corpora.get_config")
    def test_get_config_error(self, mock_get_config):
        # reference_corpora not set
        mock_get_config.return_value = {}
        with pytest.raises(
            ValueError,
            match="Configuration error: required section 'reference_corpora' not found",
        ):
            BaseReferenceCorpus().get_config_opts()

        # reference_corpora section but no base dir
        mock_get_config.return_value = {"reference_corpora": {}}
        with pytest.raises(
            ValueError,
            match="Configuration error: required 'reference_corpora.base_dir' not found",
        ):
            BaseReferenceCorpus().get_config_opts()


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
    config_opts = config.get_config()
    # use the configured text data dir
    data_dir = pathlib.Path(
        config_opts["reference_corpora"]["internet_poems"]["text_dir"]
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    for sample in INTERNETPOEMS_TEXTS:
        text_file = data_dir / f"{sample['id']}.txt"
        text_file.write_text(sample["text"])
    return data_dir


@pytest.fixture
def internetpoems_tarball(tmp_path):
    # test fixture to create tar.gzip of internet poems data directory with sample text files
    # should be used with default config
    config_opts = config.get_config()
    internetpoems_data_dir = tmp_path / "internet_poems_texts"
    internetpoems_data_dir.mkdir(parents=True, exist_ok=True)
    for sample in INTERNETPOEMS_TEXTS:
        text_file = internetpoems_data_dir / f"{sample['id']}.txt"
        text_file.write_text(sample["text"])

    tarfile_name = config_opts["reference_corpora"]["internet_poems"]["text_dir"]
    base_dir = pathlib.Path(config_opts["reference_corpora"]["base_dir"])
    tarfile_path = base_dir / tarfile_name
    tarfile_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tarfile_path, "w:gz") as tar:
        for text_file in internetpoems_data_dir.glob("*.txt"):
            tar.add(text_file)

    return tarfile_path


class TestInternetPoems:
    def test_init(self, tmp_path, corppa_test_config):
        # path in test config doesn't exist
        with pytest.raises(ValueError, match="Configuration error:.* does not exist"):
            InternetPoems()

        config_opts = config.get_config()
        # expected data_dir
        expected_data_dir = pathlib.Path(
            config_opts["reference_corpora"]["internet_poems"]["text_dir"]
        )

        # init should succeed when directory exists
        expected_data_dir.mkdir(parents=True)
        internet_poems = InternetPoems()
        assert isinstance(internet_poems.text_dir, pathlib.Path)
        assert internet_poems.text_dir == expected_data_dir

        # error if it is not a directory : remove dir and create a regular file
        expected_data_dir.rmdir()
        expected_data_dir.touch()
        with pytest.raises(
            ValueError, match="Configuration error:.* is not a directory"
        ):
            InternetPoems()

    @patch("corppa.poetry_detection.ref_corpora.pathlib")
    def test_get_config(self, mock_pathlib, tmp_path, corppa_test_config):
        config_opts = InternetPoems().get_config_opts()
        # should pass in reference corpus base directory
        assert "base_dir" in config_opts
        # should include ref_corpus specific options, where are in the test config
        assert "text_dir" in config_opts

    @patch.object(InternetPoems, "get_config_opts")
    def test_get_metadata_df(
        self, mock_get_config_opts, tmp_path, corppa_test_config, internetpoems_data_dir
    ):
        mock_get_config_opts.return_value = {"text_dir": str(internetpoems_data_dir)}
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

    def test_get_metadata_df_tarball(
        self,
        tmp_path,
        corppa_test_config_defaults,
        internetpoems_tarball,
    ):
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

    def test_get_text_corpus_tarball(
        self,
        tmp_path,
        corppa_test_config_defaults,
        internetpoems_tarball,
    ):
        internet_poems = InternetPoems()
        with pytest.raises(NotImplementedError, match="not supported for tar.gz"):
            # returns a generator; use list to get to actually run
            list(internet_poems.get_text_corpus())

    @patch.object(InternetPoems, "get_config_opts")
    def test_get_text_corpus(
        self,
        mock_get_config_opts,
        tmp_path,
        corppa_test_config,
        internetpoems_data_dir,
    ):
        mock_get_config_opts.return_value = {
            "text_dir": str(internetpoems_data_dir),
            "base_dir": tmp_path / "ref-corpora",
        }
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

    config_opts = config.get_config()
    # use the configured data paths or configured ref_corpus base_dir and defaults
    base_dir = pathlib.Path(config_opts["reference_corpora"]["base_dir"])
    # if configured, text_dir overrides default path
    if ChadwyckHealey.corpus_id in config_opts["reference_corpora"]:
        override_opts = config_opts["reference_corpora"][ChadwyckHealey.corpus_id]
        if "text_dir" in override_opts:
            data_dir = pathlib.Path(override_opts["text_dir"])
        if "metadata_path" in override_opts:
            ch_meta_csv = pathlib.Path(override_opts["metadata_path"])
    else:
        data_dir = ChadwyckHealey.text_dir
        ch_meta_csv = ChadwyckHealey.metadata_path

    # in either case, make relative to base dir if not absolute
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir
    if not ch_meta_csv.is_absolute():
        ch_meta_csv = base_dir / ch_meta_csv

    data_dir.mkdir(parents=True, exist_ok=True)
    ch_meta_csv.write_text("""id,author_lastname,author_firstname,author_birth,author_death,author_period,transl_lastname,transl_firstname,transl_birth,transl_death,title_id,title_main,title_sub,edition_id,edition_text,period,genre,rhymes
Z300475611,Robinson,Mary,1758,1800,,,,,,Z300475611,THE CAVERN OF WOE.,,Z000475579,The Poetical Works (1806),Later Eighteenth-Century 1750-1799,,y""")
    return ch_meta_csv


class TestChadwyckHealey:
    def test_init(self, tmp_path, corppa_test_config, chadwyck_healey_csv):
        # configured metadata file doesn't exist
        chadwyck_healey_csv.unlink()
        with pytest.raises(
            ValueError, match="Configuration error:.* metadata .* does not exist"
        ):
            ChadwyckHealey()

    def test_get_metadata_df(self, tmp_path, corppa_test_config, chadwyck_healey_csv):
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

        mock_pl_read_csv.assert_called_with(
            opoems.metadata_path, schema=METADATA_SCHEMA
        )

    @patch.object(OtherPoems, "get_config_opts")
    def test_config_error(self, mock_get_config_opts):
        mock_get_config_opts.return_value = {}
        with pytest.raises(
            ValueError, match="Configuration error:.* 'metadata_path' is not set"
        ):
            OtherPoems()


# because this method instantiates the ref_corpus objects,
# data directories must pass validation checks


def test_all_corpora(
    corppa_test_config,
    internetpoems_data_dir,
    chadwyck_healey_csv,
    otherpoems_metadata_df,
):
    all_ref_corpora = all_corpora()
    assert all(
        isinstance(ref_corpus, BaseReferenceCorpus) for ref_corpus in all_ref_corpora
    )
    corpus_classes = [ref_corpus.__class__ for ref_corpus in all_ref_corpora]
    # order indicates priority, so check both presence and order
    assert corpus_classes == [InternetPoems, ChadwyckHealey, OtherPoems]


def test_fulltext_corpora(
    corppa_test_config,
    internetpoems_data_dir,
    chadwyck_healey_csv,
    otherpoems_metadata_df,
):
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


def test_save_poem_metadata(
    tmp_path,
    capsys,
    corppa_test_config,
    internetpoems_data_dir,
    chadwyck_healey_csv,
    otherpoems_metadata_df,
):
    # data fixtures should ensure that all the expected directories exist
    config_opts = config.get_config()
    expected_data_dir = pathlib.Path(config_opts["compiled_dataset"]["output_data_dir"])

    # add corpus id to other poems data frame and patch it to be returned
    otherpoems_metadata_df = otherpoems_metadata_df.with_columns(
        ref_corpus=pl.lit(OtherPoems.corpus_id)
    )
    with patch.object(
        OtherPoems, "get_metadata_df", return_value=otherpoems_metadata_df
    ):
        # data dir does not exist
        with pytest.raises(
            ValueError,
            match="Configuration error: compiled dataset path .* does not exist",
        ):
            save_poem_metadata()

        # exists but not a directory
        expected_data_dir.touch()
        with pytest.raises(
            ValueError,
            match="Configuration error: compiled dataset path .* is not a directory",
        ):
            save_poem_metadata()

        # valid directory - should create a csv file
        expected_data_dir.unlink()
        expected_data_dir.mkdir(parents=True)
        save_poem_metadata()
        expected_meta_file = expected_data_dir / "poem_meta.csv"
        assert expected_meta_file.exists()
        # check output
        captured = capsys.readouterr()
        # create vs replace
        assert "Creating" in captured.out
        # output currently includes summary numbers
        assert "6 poem metadata entries" in captured.out

        # run again when the file already exists
        save_poem_metadata()
        captured = capsys.readouterr()
        assert "Replacing" in captured.out
