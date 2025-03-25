import pathlib
from collections.abc import Generator

import polars as pl

from corppa.config import get_config
from corppa.utils.build_text_corpus import build_text_corpus

#: schema for reference corpora metadata :class:`pl.DataFrame`
METADATA_SCHEMA = {
    "poem_id": pl.String,
    "author": pl.String,
    "title": pl.String,
    "ref_corpus": pl.String,
}


class BaseReferenceCorpus:
    """
    Base class for reference poetry corpora, with corpus identifier and
    methods to access metadata and text content.
    """

    corpus_id: str
    corpus_name: str
    data_path: pathlib.Path
    metadata_path: pathlib.Path

    def get_config_opts(self) -> dict:
        # load reference section for this corpus from config file
        config_opts = get_config()
        if "reference_corpora" not in config_opts:
            raise ValueError(
                "Reference corpora configuration section `reference_corpora` not found in config file"
            )
        try:
            return config_opts["reference_corpora"][self.corpus_id]
        except KeyError:
            raise ValueError(
                f"Configuration for `{self.corpus_id}` reference corpus not found"
            )

    def get_metadata_df(self) -> pl.DataFrame:
        """Minimal common poetry metadata for use across reference corpora.
        Should return a :class:`pl.DataFrame` with poem_id, author, title, and
        ref_corpus for each poem in this corpus."""
        raise NotImplementedError

    def get_text_corpus(self) -> Generator[dict[str, str]]:
        """Minimal text record for reference corpora.
        Should yield a dictionary with id and text for each poem in this
        corpus."""
        raise NotImplementedError


class LocalTextCorpus(BaseReferenceCorpus):
    """Base class for reference corpus where text content is
    provided as a set of text files in a directory.
    Provides :meth:`get_text_corpus` for generating text corpus from
    the file system."""

    def get_text_corpus(
        self, disable_progress: bool = True
    ) -> Generator[dict[str, str]]:
        # NOTE: the build_text_corpus method returns id,
        # so we have to rename to poem_id
        yield from (
            {"poem_id": p["id"], "text": p["text"]}
            for p in build_text_corpus(
                self.data_path, disable_progress=disable_progress
            )
        )


class InternetPoems(LocalTextCorpus):
    """Curated corpus of poems with plain text content sourced from
    the internet, for high priority sources known to occur in excerpts,
    including full text of Shakespeare's plays. Metadata is inferred based on
    filename, with a naming convention of `Firstname-Lastname_Poem-Title.txt`.
    The filename without extension is used as the `poem_id`.
    """

    corpus_id: str = "internet_poems"
    corpus_name: str = "Internet Poems"
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

    def get_metadata_df(self) -> pl.DataFrame:
        metadata = []
        for text_file in self.data_path.glob("*.txt"):
            # filename format:
            #   Firstname-Lastname_Poem-Title.txt
            # use filename without extension as poem identifier
            poem_id = text_file.stem
            #   Replace - with spaces and split on - to separate author/title
            author, title = poem_id.replace("-", " ").split("_", 1)
            metadata.append(
                {
                    "poem_id": poem_id,
                    "author": author,
                    "title": title,
                    "ref_corpus": self.corpus_id,
                }
            )
        return pl.from_dicts(metadata, schema=METADATA_SCHEMA)


class ChadwyckHealey(LocalTextCorpus):
    """Reference corpus based on a filtered subset of Chadwyck-Healey
    poetry collection. Requires a directory of plain text files and a
    metadata csv file. Uses Chadwyck-Healey identifiers for `poem_id`.
    """

    corpus_id: str = "chadwyck-healey"
    corpus_name: str = "Chadwyck-Healey"  # should we note that it is filtered?
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

    def get_metadata_df(self) -> pl.DataFrame:
        df = (
            # ignore parse errors in fields we don't care about (author_dob)
            pl.read_csv(self.metadata_path, ignore_errors=True)
            # rename fields
            .rename({"title_main": "title", "id": "poem_id"})
            # construct author name from separate fields in the metadata
            .with_columns(
                author=pl.concat_str(
                    [pl.col("author_firstname"), pl.col("author_lastname")],
                    separator=" ",
                ),
                # set corpus id
                ref_corpus=pl.lit(self.corpus_id),
            )
            .select(["poem_id", "author", "title", "ref_corpus"])
        )
        # construct a new dataframe with our schema and add the content
        return pl.DataFrame([], schema=METADATA_SCHEMA).extend(df)


class OtherPoems(BaseReferenceCorpus):
    """A metadata-only reference corpus with metadata for poems that have
    been identified but for which we do not have full text.
    Poem identifiers are constructed from author and title using the same
    convention as :class:`InternetPoems`.

    Does not provide an implementation for :meth:`get_text`.
    """

    corpus_id: str = "other"
    corpus_name: str = "Other Poems"
    metadata_url: str

    def __init__(self):
        # get configuration for this corpus
        config_opts = self.get_config_opts()
        # set data path from config file and check that path exists
        self.metadata_url = config_opts["metadata_url"]
        # TODO: handle key error

    def get_metadata_df(self) -> pl.DataFrame:
        # polars can load csv from a url;
        df = pl.read_csv(self.metadata_url, schema=METADATA_SCHEMA)
        df = df.with_columns(ref_corpus=pl.lit(self.corpus_id))
        meta_df = pl.DataFrame([], schema=METADATA_SCHEMA)
        meta_df.extend(df)
        return meta_df

    # this is a metadata-only corpus, so leave get_text as not implemented


def all_corpora() -> list[BaseReferenceCorpus]:
    """Convenience access to all reference corpora, for generating
    compiled versions of reference data."""
    return [InternetPoems(), ChadwyckHealey(), OtherPoems()]


def fulltext_corpora() -> list[BaseReferenceCorpus]:
    """Convenience access to all full-text reference corpora, for generating
    compiled metadata and text."""
    return [InternetPoems(), ChadwyckHealey()]


def compile_metadata_df() -> pl.DataFrame:
    """Compile poetry metadata from all reference corpora into a single
    polars DataFrame with reference corpus ids."""
    # create an empty dataframe with the intended fields
    poem_metadata = pl.DataFrame([], schema=METADATA_SCHEMA)

    # for each corpus, load poem metadata into a polars dataframe,
    # rename id to poem_id, and add a column with the corpus id
    for ref_corpus in all_corpora():
        meta_df = ref_corpus.get_metadata_df()
        poem_metadata.extend(meta_df.head())
    return poem_metadata


def save_poem_metadata():
    """Generate and save compiled poetry metadata as a data file in the
    poem dataset.
    """
    config_opts = get_config()
    output_data_dir = pathlib.Path(config_opts["poem_dataset"]["data_dir"])
    if not (output_data_dir.exists() and output_data_dir.is_dir()):
        raise ValueError("Poem dataset path is not configured correctly")

    output_file = output_data_dir / "poem_meta.csv"

    # check & report if the file already exists
    output_verb = "Creating"
    if output_file.exists():
        output_verb = "Replacing"
    print(f"{output_verb} {output_file}")

    df = compile_metadata_df()
    ref_corpus_names = {
        ref_corpus.corpus_id: ref_corpus.corpus_name for ref_corpus in all_corpora()
    }

    total_by_corpus = df["ref_corpus"].value_counts()
    totals = []
    for value, count in total_by_corpus.iter_rows():
        # row is a tuple of value, count;  convert reference corpus id to name
        totals.append(f"{ref_corpus_names[value]}: {count:,}")

    print(f"{df.height:,} poem metadata entries ({'; '.join(totals)})")
    df.write_csv(output_file, include_bom=True)
