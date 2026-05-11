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
    ref_base_dir = tmp_path / "ref-corpora"
    ref_base_dir.mkdir()

    ref_corpus_names = ["internet_poems", "chadwyck-healey", "other_poems"]

    ref_corpus_configs = {
        name: config.CorpusConfig(name=name, relative_dir=ref_base_dir)
        for name in ref_corpus_names
    }

    config_opts = config.ConfigOpts(
        base_dir=tmp_path,
        compiled_dataset_dir=compiled_dataset_dir,
        # ppa_corpus=None,
        reference_corpora=ref_corpus_configs,
        # excerpt_data_dir=config_values.get("excerpt_data_dir"),
        # poem_clusters_path=config_values.get("poem_clusters_path"),
    )

    # validation requires the files to exist, so create them
    config_opts.reference_corpora["internet_poems"].base_dir.mkdir()
    config_opts.reference_corpora["internet_poems"].text_path.touch()
    config_opts.reference_corpora["chadwyck-healey"].base_dir.mkdir()
    config_opts.reference_corpora["chadwyck-healey"].text_path.touch()
    config_opts.reference_corpora["chadwyck-healey"].metadata_path.touch()
    config_opts.reference_corpora["other_poems"].base_dir.mkdir()
    config_opts.reference_corpora["other_poems"].metadata_path.touch()

    with patch.object(config, "get_config") as mock_get_config:
        # this patches calls in the current test file
        mock_get_config.return_value = config_opts
        # this patches calls in ref_corpora
        with patch(
            "corppa.poetry_detection.ref_corpora.get_config"
        ) as ref_corppa_config:
            ref_corppa_config.return_value = config_opts
            yield config_opts


class TestBaseReferenceCorpus:
    def test_not_implemented(self):
        with pytest.raises(NotImplementedError):
            BaseReferenceCorpus().get_metadata_df()

        with pytest.raises(NotImplementedError):
            BaseReferenceCorpus().get_text_corpus()

    def test_calculate_poem_length(self):
        # Test single line text
        result = BaseReferenceCorpus.calculate_poem_length("Hello world test")
        assert result == {"num_lines": 1, "num_words": 3, "char_len": 16}

        # Test multi-line text with blank lines
        text = "Line one here\nLine two here\n\nLine three"
        result = BaseReferenceCorpus.calculate_poem_length(text)
        assert result == {"num_lines": 3, "num_words": 8, "char_len": len(text)}

        # Test empty text
        result = BaseReferenceCorpus.calculate_poem_length("")
        assert result == {"num_lines": 0, "num_words": 0, "char_len": 0}

        # Test text with only blank lines
        result = BaseReferenceCorpus.calculate_poem_length("   \n\n   ")
        assert result == {"num_lines": 0, "num_words": 0, "char_len": 8}


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
def internetpoems_data_dir(tmp_path, corppa_test_config):
    # test fixture to create internet poems data directory with sample text files
    config_opts = config.get_config()
    # update the configured text path in fixture config to be a directory
    data_dir = config_opts.reference_corpora["internet_poems"].base_dir / "text_files"
    data_dir.mkdir(exist_ok=True)
    config_opts.reference_corpora["internet_poems"].text_path = data_dir
    for sample in INTERNETPOEMS_TEXTS:
        text_file = data_dir / f"{sample['id']}.txt"
        text_file.write_text(sample["text"])
    return data_dir


@pytest.fixture
def internetpoems_tarball(tmp_path, corppa_test_config):
    # test fixture to create tar.gzip of internet poems data directory with sample text files
    config_opts = config.get_config()
    internetpoems_data_dir = tmp_path / "internet_poems_texts"
    internetpoems_data_dir.mkdir(exist_ok=True)
    for sample in INTERNETPOEMS_TEXTS:
        text_file = internetpoems_data_dir / f"{sample['id']}.txt"
        text_file.write_text(sample["text"])

    tarfile_path = config_opts.reference_corpora["internet_poems"].text_path

    with tarfile.open(tarfile_path, "w:gz") as tar:
        for text_file in internetpoems_data_dir.glob("*.txt"):
            tar.add(text_file)

    return tarfile_path


class TestInternetPoems:
    def test_init(self, corppa_test_config):
        config_opts = config.get_config()
        ip_config = config_opts.reference_corpora["internet_poems"]
        internet_poems = InternetPoems(ip_config)
        assert internet_poems.config == ip_config
        assert isinstance(internet_poems.config, config.CorpusConfig)

    def test_get_metadata_df(
        self, tmp_path, corppa_test_config, internetpoems_data_dir
    ):
        config_opts = config.get_config()
        internet_poems = InternetPoems(config_opts.reference_corpora["internet_poems"])
        meta_df = internet_poems.get_metadata_df(poem_length=True)
        assert isinstance(meta_df, pl.DataFrame)
        assert meta_df.schema == METADATA_SCHEMA
        assert meta_df.height == len(INTERNETPOEMS_TEXTS)
        # get the first row as a dict; sort by id so order matches input
        meta_row = meta_df.sort("poem_id").row(0, named=True)
        assert meta_row["poem_id"] == INTERNETPOEMS_TEXTS[0]["id"]
        assert meta_row["author"] == "King James Bible"
        assert meta_row["title"] == "Psalms"
        assert meta_row["ref_corpus"] == internet_poems.corpus_id
        # check poem length calculations (non-blank lines, word count, char length)
        assert meta_row["num_lines"] == 1
        assert meta_row["num_words"] == 9
        assert meta_row["char_len"] == len(INTERNETPOEMS_TEXTS[0]["text"])

    def test_get_metadata_df_no_poem_length(
        self, tmp_path, corppa_test_config, internetpoems_data_dir
    ):
        # Test that poem_length=False sets length fields to null
        config_opts = config.get_config()
        internet_poems = InternetPoems(config_opts.reference_corpora["internet_poems"])
        meta_df = internet_poems.get_metadata_df(poem_length=False)
        assert isinstance(meta_df, pl.DataFrame)
        # Length fields should be present but null
        assert "num_lines" in meta_df.columns
        assert "num_words" in meta_df.columns
        assert "char_len" in meta_df.columns
        # All length values should be null
        assert (
            meta_df.select("num_lines", "num_words", "char_len").drop_nulls().height
            == 0
        )

    def test_get_metadata_df_csv_and_poem_length(
        self, tmp_path, corppa_test_config, internetpoems_data_dir, capsys
    ):
        # test case where we load metadata from CSV and add poem length info

        config_opts = config.get_config()
        ipoem_cfg = config_opts.reference_corpora["internet_poems"]
        # create a very simple metadata file based on fixture dictionary ids
        ip_meta_csv = ipoem_cfg.metadata_path
        # split id into poem id, author, title
        csv_rows = [
            (poem["id"], poem["id"].split("_")[0], poem["id"].split("_")[1])
            for poem in INTERNETPOEMS_TEXTS
        ]
        # combine header row and one row for each poem, then write out to the meta path
        csv_text = "poem_id,author,title\n" + "\n".join(
            ",".join(row) for row in csv_rows
        )
        ip_meta_csv.write_text(csv_text)

        internet_poems = InternetPoems(ipoem_cfg)
        meta_df = internet_poems.get_metadata_df(poem_length=True)
        assert isinstance(meta_df, pl.DataFrame)
        # Length fields should be present but null
        assert "num_lines" in meta_df.columns
        assert "num_words" in meta_df.columns
        assert "char_len" in meta_df.columns

        # get the first row as a dict; sort by id so order matches input
        meta_row = meta_df.sort("poem_id").row(0, named=True)
        assert meta_row["poem_id"] == INTERNETPOEMS_TEXTS[0]["id"]
        assert meta_row["ref_corpus"] == internet_poems.corpus_id
        # check poem length calculations (non-blank lines, word count, char length)
        assert meta_row["num_lines"] == 1
        assert meta_row["num_words"] == 9
        assert meta_row["char_len"] == len(INTERNETPOEMS_TEXTS[0]["text"])

    def test_get_metadata_df_tarball(
        self,
        tmp_path,
        corppa_test_config,
        internetpoems_tarball,
    ):
        config_opts = config.get_config()
        internet_poems = InternetPoems(config_opts.reference_corpora["internet_poems"])
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
        corppa_test_config,
        internetpoems_tarball,
    ):
        config_opts = config.get_config()
        internet_poems = InternetPoems(config_opts.reference_corpora["internet_poems"])
        # returns a generator; use list to get to actually run
        # convert to list, sort to ensure order matches fixture data
        text_data = sorted(
            list(internet_poems.get_text_corpus()), key=lambda x: x["poem_id"]
        )
        assert len(text_data) == len(INTERNETPOEMS_TEXTS)
        assert text_data[0]["poem_id"] == INTERNETPOEMS_TEXTS[0]["id"]
        assert text_data[0]["text"] == INTERNETPOEMS_TEXTS[0]["text"]

    def test_get_text_corpus(
        self,
        tmp_path,
        corppa_test_config,
        internetpoems_data_dir,
    ):
        config_opts = config.get_config()
        internet_poems = InternetPoems(config_opts.reference_corpora["internet_poems"])
        text_data = internet_poems.get_text_corpus()
        assert isinstance(text_data, Generator)
        # turn the generator into a list; sort by id so order matches input
        text_data = sorted(text_data, key=lambda x: x["poem_id"])
        assert len(text_data) == len(INTERNETPOEMS_TEXTS)
        assert text_data[0]["poem_id"] == INTERNETPOEMS_TEXTS[0]["id"]
        assert text_data[0]["text"] == INTERNETPOEMS_TEXTS[0]["text"]


@pytest.fixture
def chadwyck_healey_csv(tmp_path, corppa_test_config):
    "fixture to create a test version of the chadwyck-healey metadata csv file"
    config_opts = config.get_config()
    ch_config = config_opts.reference_corpora[ChadwyckHealey.corpus_id]
    # unlink fixture file and make text path a directory
    ch_config.text_path.unlink()
    data_dir = ch_config.base_dir / "text_files"
    data_dir.mkdir(exist_ok=True)
    ch_config.text_path = data_dir
    ch_meta_csv = ch_config.metadata_path

    ch_meta_csv.write_text("""id,author_lastname,author_firstname,author_birth,author_death,author_period,transl_lastname,transl_firstname,transl_birth,transl_death,title_id,title_main,title_sub,edition_id,edition_text,period,genre,rhymes
Z300475611,Robinson,Mary,1758,1800,,,,,,Z300475611,THE CAVERN OF WOE.,,Z000475579,The Poetical Works (1806),Later Eighteenth-Century 1750-1799,,y""")
    return ch_meta_csv


class TestChadwyckHealey:
    def test_init(self, corppa_test_config, chadwyck_healey_csv):
        config_opts = config.get_config()
        ch = ChadwyckHealey(config_opts.reference_corpora[ChadwyckHealey.corpus_id])
        assert isinstance(ch.config, config.CorpusConfig)
        assert ch.config.metadata_path == chadwyck_healey_csv

    def test_get_metadata_df(self, tmp_path, corppa_test_config, chadwyck_healey_csv):
        config_opts = config.get_config()
        chadwyck_healey = ChadwyckHealey(
            config_opts.reference_corpora[ChadwyckHealey.corpus_id]
        )
        meta_df = chadwyck_healey.get_metadata_df()
        assert isinstance(meta_df, pl.DataFrame)
        # schema is a subset because we don't include poem lengths
        assert all(key in METADATA_SCHEMA for key in meta_df.schema.keys())
        # csv fixture data currently has one row
        assert meta_df.height == 1
        # get the first row as a dict and check values
        meta_row = meta_df.row(0, named=True)
        assert meta_row["poem_id"] == "Z300475611"
        assert meta_row["author"] == "Mary Robinson"
        assert meta_row["title"] == "THE CAVERN OF WOE."
        assert meta_row["ref_corpus"] == chadwyck_healey.corpus_id

    def test_get_metadata_df_with_poem_length(
        self, tmp_path, corppa_test_config, chadwyck_healey_csv
    ):
        # Create a text file for the poem to test poem length calculation
        config_opts = config.get_config()
        chadwyck_healey = ChadwyckHealey(
            config_opts.reference_corpora[ChadwyckHealey.corpus_id]
        )
        text_dir = chadwyck_healey.config.text_path
        # three lines, eight words
        text_content = "Line one here\nLine two here\nLine three"
        text_file = text_dir / "Z300475611.txt"
        text_file.write_text(text_content)

        meta_df = chadwyck_healey.get_metadata_df(poem_length=True)
        assert isinstance(meta_df, pl.DataFrame)
        # Should include length fields
        assert "num_lines" in meta_df.columns
        assert "num_words" in meta_df.columns
        assert "char_len" in meta_df.columns

        meta_row = meta_df.row(0, named=True)
        # 3 non-blank lines
        assert meta_row["num_lines"] == 3
        # 8 words total
        assert meta_row["num_words"] == 8
        # character length (including newlines)
        assert meta_row["char_len"] == len(text_content)

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
        config_opts = config.get_config()
        opoems = OtherPoems(config_opts.reference_corpora["other_poems"])
        meta_df = opoems.get_metadata_df()
        assert isinstance(meta_df, pl.DataFrame)
        # schema is a subset because we don't include poem lengths
        assert all(key in METADATA_SCHEMA for key in meta_df.schema.keys())
        assert meta_df.height == len(OTHERPOEM_METADATA)
        # check values on the first row
        meta_row = meta_df.row(0, named=True)
        assert meta_row["poem_id"] == OTHERPOEM_METADATA[0][0]
        assert meta_row["author"] == OTHERPOEM_METADATA[1][0]
        assert meta_row["title"] == OTHERPOEM_METADATA[2][0]
        assert meta_row["ref_corpus"] == opoems.corpus_id

        mock_pl_read_csv.assert_called_with(
            opoems.config.metadata_path, schema=METADATA_SCHEMA
        )


# because this method instantiates the ref_corpus objects,
# data directories must pass validation checks


def test_all_corpora(corppa_test_config):
    all_ref_corpora = all_corpora()
    assert all(
        isinstance(ref_corpus, BaseReferenceCorpus) for ref_corpus in all_ref_corpora
    )
    corpus_classes = [ref_corpus.__class__ for ref_corpus in all_ref_corpora]
    # order indicates priority, so check both presence and order
    assert corpus_classes == [InternetPoems, ChadwyckHealey, OtherPoems]


def test_fulltext_corpora(corppa_test_config):
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

    # add corpus id to other poems data frame and patch it to be returned
    otherpoems_metadata_df = otherpoems_metadata_df.with_columns(
        ref_corpus=pl.lit(OtherPoems.corpus_id)
    )
    with patch.object(
        OtherPoems, "get_metadata_df", return_value=otherpoems_metadata_df
    ):
        # create a path reference for the file we want to create
        output_file = tmp_path / "poem_meta.csv"
        save_poem_metadata(output_file)
        assert output_file.exists()
        # check output
        captured = capsys.readouterr()
        # create vs replace
        assert "Creating" in captured.out
        # output currently includes summary numbers
        assert "6 poem metadata entries" in captured.out

        # run again when the file already exists
        save_poem_metadata(output_file)
        captured = capsys.readouterr()
        assert "Replacing" in captured.out


def test_save_poem_metadata_with_cluster_ids(
    tmp_path,
    capsys,
    corppa_test_config,
    internetpoems_data_dir,
    otherpoems_metadata_df,
    chadwyck_healey_csv,
):
    # test compiling in poem cluster ids;
    # confirm left join works as desired, adding clusters or nulls
    # for poems that do not have a cluster id

    # made up cluster for testing purposes, with ids from fixtures
    cluster_id_df = pl.DataFrame(
        data={
            "poem_id": ["Robert-Burns_Mary", "Z300475611"],
            "cluster_id": ["mary", "mary"],
        }
    )
    # add corpus id to other poems data frame and patch it to be returned
    otherpoems_metadata_df = otherpoems_metadata_df.with_columns(
        ref_corpus=pl.lit(OtherPoems.corpus_id)
    )
    with patch.object(
        OtherPoems, "get_metadata_df", return_value=otherpoems_metadata_df
    ):
        # create a path reference for the file we want to create
        output_file = tmp_path / "poem_meta.csv"
        save_poem_metadata(output_file, poem_clusters_df=cluster_id_df)
        assert output_file.is_file()
        # check output
        df = pl.read_csv(output_file)
        # should still have all rows
        assert df.height == 6
        # 2 poems should have cluster id
        assert "cluster_id" in df.columns
        assert df.filter(pl.col.cluster_id.is_not_null()).height == 2
        # others should be null
        assert df.filter(pl.col.cluster_id.is_null()).height == 4

        captured = capsys.readouterr()
        # check output reporting on cluster ids
        assert "2 poems with cluster ids" in captured.out
        assert "(1 unique cluster)" in captured.out


def test_save_poem_metadata_with_excerpts(
    tmp_path,
    capsys,
    corppa_test_config,
    internetpoems_data_dir,
    chadwyck_healey_csv,
    otherpoems_metadata_df,
):
    # Test the case where excerpts_df is provided - tests aggregation logic

    # add corpus id to other poems data frame and patch it to be returned
    otherpoems_metadata_df = otherpoems_metadata_df.with_columns(
        ref_corpus=pl.lit(OtherPoems.corpus_id)
    )

    # Create sample excerpts dataframe with poem data
    # Use poem IDs from the INTERNETPOEMS_TEXTS global variable
    excerpts_df = pl.from_dicts(
        [
            # two excerpts for poem 0 from the same work, two different pages
            {
                "poem_id": INTERNETPOEMS_TEXTS[0]["id"],
                "excerpt_id": "p@1:10",
                "ppa_work_id": "work1",
                "page_id": "page1",
            },
            {
                "poem_id": INTERNETPOEMS_TEXTS[0]["id"],
                "excerpt_id": "p@3:30",
                "ppa_work_id": "work1",
                "page_id": "page2",
            },
            # one excerpt for poem 2
            {
                "poem_id": INTERNETPOEMS_TEXTS[1]["id"],
                "excerpt_id": "ex3",
                "ppa_work_id": "work2",
                "page_id": "page3",
            },
        ]
    )

    aggregation_fields = ["num_excerpts", "num_ppa_works", "num_ppa_pages"]

    with patch.object(
        OtherPoems, "get_metadata_df", return_value=otherpoems_metadata_df
    ):
        output_file = tmp_path / "poem_meta.csv"
        save_poem_metadata(output_file, excerpts_df=excerpts_df)
        assert output_file.exists()

        # Read the output CSV and check for aggregate columns
        result_df = pl.read_csv(output_file)
        # all fields should be present
        for field in aggregation_fields:
            assert field in result_df.columns

    # Check that poem with 2 excerpts has correct counts
    psalms_row = result_df.filter(
        pl.col("poem_id") == INTERNETPOEMS_TEXTS[0]["id"]
    ).row(0, named=True)
    # two excerpts from one work, different pages
    assert psalms_row["num_excerpts"] == 2
    assert psalms_row["num_ppa_works"] == 1
    assert psalms_row["num_ppa_pages"] == 2

    # Check that poem with 1 excerpt has correct counts
    mary_row = result_df.filter(pl.col("poem_id") == INTERNETPOEMS_TEXTS[1]["id"]).row(
        0, named=True
    )
    # one excerpt, all counts are 1
    assert all(mary_row[value] == 1 for value in aggregation_fields)

    # Check that poems without excerpts (from otherpoems) have zero counts
    for poem_info in result_df.filter(
        pl.col("poem_id").is_in(OTHERPOEM_METADATA[0])
    ).iter_rows(named=True):
        assert all(poem_info[value] == 0 for value in aggregation_fields)
