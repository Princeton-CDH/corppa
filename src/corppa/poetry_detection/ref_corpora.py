import pathlib
from collections.abc import Generator

import polars as pl

from corppa.config import get_config
from corppa.utils.build_text_corpus import build_text_corpus


class BaseReferenceCorpus:
    """
    Base class for reference poetry corpora, with corpus identifier and
    methods to access metadata and text content.
    """

    corpus_id: str
    data_path: pathlib.Path
    metadata_path: pathlib.Path

    def get_config_opts(self):
        # load data path from config file and check that path exists
        config_opts = get_config()
        # TODO: handle missing config opts
        return config_opts["reference_corpora"][self.corpus_id]

    def get_metadata(self) -> Generator[dict[str, str]]:
        """Minimal common poetry metadata for use across reference corpora.
        Should yield a dictionary with id, author, and title for each poem
        in this corpus."""
        raise NotImplementedError

    def get_text_corpus(self) -> Generator[dict[str, str]]:
        """Minimal text record for reference corpora.
        Should yield a dictionary with id and text for each poem in this
        corpus."""
        raise NotImplementedError


class InternetPoemsCorpus(BaseReferenceCorpus):
    corpus_id: str = "internet_poems"
    data_path: pathlib.Path

    def __init__(self):
        # get configuration for this corpus
        config_opts = self.get_config_opts()
        # set data path from config file and check that path exists
        self.data_path = pathlib.Path(config_opts["data_path"])
        if not (self.data_path.exists() and self.data_path.is_dir()):
            raise ValueError(
                "Internet Poems Reference Corpus is not configured correctly"
            )

    def get_metadata(self) -> Generator[dict[str, str]]:
        for text_file in self.data_path.glob("*.txt"):
            # filename format:
            #   Firstname-Lastname_Poem-Title.txt
            # use filename without extension as poem identifier
            poem_id = text_file.stem
            #   Replace - with spaces and split on - to separate author/title
            author, title = poem_id.replace("-", " ").split("_", 1)
            yield {"id": text_file.stem, "author": author, "title": title}

    def get_text(self, disable_progress: bool = True) -> Generator[dict[str, str]]:
        yield from build_text_corpus(self.data_path, disable_progress=disable_progress)


class ChadwyckHealeyCorpus(BaseReferenceCorpus):
    corpus_id: str = "chadwyck-healey"
    data_path: pathlib.Path
    metadata_path: pathlib.Path

    def __init__(self):
        # get configuration for this corpus
        config_opts = self.get_config_opts()
        # set data path from config file and check that path exists
        self.data_path = pathlib.Path(config_opts["data_path"])
        if not (self.data_path.exists() and self.data_path.is_dir()):
            raise ValueError(
                "Chadwyck-Healey Reference Corpus is not configured correctly"
            )
        self.metadata_path = pathlib.Path(config_opts["metadata_path"])
        if not (self.metadata_path.exists() and self.metadata_path.is_file()):
            raise ValueError(
                "Chadwyck-Healey Reference Corpus is not configured correctly"
            )

    def get_metadata(self) -> Generator[dict[str, str]]:
        df = (
            # ignore parse errors in fields we don't care about (author_dob)
            pl.read_csv(self.metadata_path, ignore_errors=True)
            .rename({"title_main": "title"})
            .with_columns(
                pl.concat_str(
                    [pl.col("author_firstname"), pl.col("author_lastname")],
                    separator=" ",
                ).alias("author")
            )
            .select(["id", "author", "title"])
        )
        yield from df.iter_rows(named=True)
