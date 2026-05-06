import logging
import pathlib
from collections.abc import Generator
from typing import Optional

import polars as pl

from corppa.config import CorpusConfig, get_config
from corppa.utils.build_text_corpus import build_text_corpus, text_corpus_from_tarfile

logger = logging.getLogger(__name__)


#: schema for reference corpora metadata :class:`pl.DataFrame`
METADATA_SCHEMA = {
    "poem_id": pl.String,
    "author": pl.String,
    "title": pl.String,
    "ref_corpus": pl.String,
    "num_lines": pl.Int64,
    "num_words": pl.Int64,
    "char_len": pl.Int64,
}


class BaseReferenceCorpus:
    """
    Base class for reference poetry corpora, with corpus identifier and
    methods to access metadata and text content.
    """

    corpus_id: str
    corpus_name: str
    config: CorpusConfig

    @staticmethod
    def calculate_poem_length(text: str) -> dict[str, int]:
        """Calculate poem length metrics from text content.  Takes the
        text of the poem and returns a dictionary num_lines (non-blank lines),
        num_words, and char_len.
        """
        return {
            "num_lines": len([line for line in text.splitlines() if line.strip()]),
            "num_words": len(text.split()),
            "char_len": len(text),
        }

    def get_metadata_df(self, poem_length=False) -> pl.DataFrame:
        """Minimal common poetry metadata for use across reference corpora.
        Should return a :class:`pl.DataFrame` with poem_id, author, title, and
        ref_corpus for each poem in this corpus.  Optionally, should
        return information about poem length (number of characters and lines
        in the text)."""
        raise NotImplementedError

    def get_text_corpus(self) -> Generator[dict[str, str]]:
        """Minimal text record for reference corpora.
        Should yield a dictionary with id and text for each poem in this
        corpus."""
        raise NotImplementedError


class LocalTextCorpus(BaseReferenceCorpus):
    """Base class for reference corpus where text content is
    provided as a set of text files in a directory or tar.gz.
    On initialization, configures data path based on
    configured base dir and corpus default or any overrides, and validates
    that the path exists and is a directory.
    Provides :meth:`get_text_corpus` for generating text corpus from
    the file system."""

    def __init__(self, config_opts: CorpusConfig):
        # get text directory for this reference corpus from corpus configuration
        self.config = config_opts
        # validate config file; for text-only corpus, metadata is optional
        self.config.validate(metadata=False)

    def get_text_corpus(
        self, disable_progress: bool = True
    ) -> Generator[dict[str, str]]:
        # validation is now handled by CorpusConfig.validate
        if self.config.text_path.is_dir():
            corpus_method = build_text_corpus
        elif self.config.text_path.name.endswith(".tar.gz"):
            corpus_method = text_corpus_from_tarfile

        # build_text_corpus method returns id, so rename id to poem_id
        yield from (
            {"poem_id": p["id"], "text": p["text"]}
            for p in corpus_method(
                self.config.text_path, disable_progress=disable_progress
            )
        )


class InternetPoems(LocalTextCorpus):
    """Curated corpus of poems with plain text content sourced from
    the internet, for high priority sources known to occur in excerpts,
    including full text of Shakespeare's plays. Metadata was originally based on
    filename (naming convention of `Firstname-Lastname_Poem-Title.txt`),
    but has since been converted to a CSV file for correction and augmentation.
    The text filename without extension is used as the `poem_id`.
    """

    #: id for this reference corpus: internet_poems
    corpus_id: str = "internet_poems"
    corpus_name: str = "Internet Poems"
    # inherits config with text_path

    # no init/validation needed beyond that provided by LocalTextCorpus

    def get_metadata_df(self, poem_length=False) -> pl.DataFrame:
        if (
            self.config.metadata_path is not None
            and self.config.metadata_path.is_file()
        ):
            # load metadata and add reference corpus id
            df = pl.read_csv(
                self.config.metadata_path, schema_overrides=METADATA_SCHEMA
            ).with_columns(ref_corpus=pl.lit(self.corpus_id))
            # if poem length is requested, get from the files and add to df
            if poem_length:
                length_df = self.get_metadata_from_files(poem_length=poem_length)
                df = df.join(
                    length_df.select("poem_id", "num_lines", "num_words", "char_len"),
                    on="poem_id",
                )
        else:
            # fallback metadata: generate from filenames
            df = self.get_metadata_from_files(poem_length=poem_length)

        return df

    def get_metadata_from_files(self, poem_length=False) -> pl.DataFrame:
        metadata = []
        # returns a generator of dicts with id and text string
        # NOTE: when called from compile script, might be nice to show progress bar
        for poem in self.get_text_corpus():
            # filename format:
            #   Firstname-Lastname_Poem-Title.txt
            #   Replace - with spaces and split on - to separate author/title
            author, title = poem["poem_id"].replace("-", " ").split("_", 1)
            poem_metadata: dict[str, str | int] = {
                "poem_id": poem["poem_id"],
                "author": author,
                "title": title,
                "ref_corpus": self.corpus_id,
            }
            if poem_length:
                poem_metadata.update(self.calculate_poem_length(poem["text"]))

            metadata.append(poem_metadata)

        return pl.from_dicts(metadata, schema=METADATA_SCHEMA)


class ChadwyckHealey(LocalTextCorpus):
    """Reference corpus based on a filtered subset of Chadwyck-Healey
    poetry collection. Requires a directory of plain text files and a
    metadata csv file. Uses Chadwyck-Healey identifiers for `poem_id`.
    """

    #: id for this reference corpus: chadwyck-healey
    corpus_id: str = "chadwyck-healey"
    corpus_name: str = "Chadwyck-Healey"
    # inherits config with text_path &  metadata path

    def get_metadata_df(self, poem_length=False) -> pl.DataFrame:
        # disable schema inference; the fields we care about are all strings
        # TODO: check / update for revised metadata
        df = (
            pl.read_csv(self.config.metadata_path, infer_schema=False)
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

        if poem_length:
            poem_lengths = []
            # text corpus returns a generator of dicts with id and text string
            # NOTE: when called from compile script, might be nice to show progress bar
            for poem in self.get_text_corpus():
                poem_lengths.append(
                    {
                        "poem_id": poem["poem_id"],
                        **self.calculate_poem_length(poem["text"]),
                    }
                )
            if poem_lengths:
                poem_length_df = pl.from_dicts(poem_lengths)
                df = df.join(poem_length_df, on="poem_id")
            else:
                logger.warning(
                    "Poem length requested but none calculated (no text files found?)"
                )

        return df


class OtherPoems(BaseReferenceCorpus):
    """A metadata-only reference corpus with metadata for poems that have
    been identified but for which we do not have full text.
    Poem identifiers are constructed from author and title using the same
    convention as :class:`InternetPoems`.

    Does not provide an implementation for :meth:`get_text_corpus`.
    """

    #: id for this reference corpus (currently "other")
    corpus_id: str = "other"
    corpus_name: str = "Other Poems"
    config: CorpusConfig
    #: URL or local path for metadata (can pull from Google Sheets published csv)
    # metadata_path: str

    def __init__(self, config_opts: CorpusConfig):
        # get configuration for this corpus
        self.config = config_opts
        # validate configuration - metadata only
        self.config.validate(text=False)

    def get_metadata_df(self, poem_length=False) -> pl.DataFrame:
        # polars can load csv directly from a url
        return pl.read_csv(
            self.config.metadata_path, schema=METADATA_SCHEMA
        ).with_columns(ref_corpus=pl.lit(self.corpus_id))

    # this is a metadata-only corpus; get_text_corpus is intentionally not implemented


def all_corpora() -> list[BaseReferenceCorpus]:
    """Convenience access to all reference corpora, for generating
    compiled versions of reference data."""
    config = get_config()
    return [
        InternetPoems(config.reference_corpora["internet_poems"]),
        ChadwyckHealey(config.reference_corpora["chadwyck-healey"]),
        OtherPoems(config.reference_corpora["other_poems"]),
    ]


def fulltext_corpora() -> list[BaseReferenceCorpus]:
    """Convenience access to all full-text reference corpora, for generating
    compiled metadata and text."""
    config = get_config()
    return [
        InternetPoems(config.reference_corpora["internet_poems"]),
        ChadwyckHealey(config.reference_corpora["chadwyck-healey"]),
    ]


def compile_metadata_df(poem_length=False) -> pl.DataFrame:
    """Compile poetry metadata from all reference corpora into a single
    polars DataFrame with reference corpus ids."""
    # Combine poem metadata from all reference corpora

    # use a diagonal concat instead of vstack/extend
    ref_corpora_dfs = [
        ref_corpus.get_metadata_df(poem_length=poem_length)
        for ref_corpus in all_corpora()
    ]
    return pl.concat(ref_corpora_dfs, how="diagonal")


def save_poem_metadata(
    output_file: pathlib.Path,
    excerpts_df: Optional[pl.DataFrame] = None,
    poem_clusters_df: Optional[pl.DataFrame] = None,
):
    """Generate and save compiled poetry metadata as a data file in the
    poem dataset. Loads and compiles metadata for all reference corpora,
    including poem length calculations (:meth:`compile_metadata_df`)
    and saves the result to the specified `output_file`.  When the optional
    `excerpts_df` is present, calculates work-level excerpt total for poems
    based on primary poem id (number of excerpts, number of PPA works,
    number of PPA pages).  When the optional `poem_clusters_df` is provided,
    adds a `cluster_id` field to poems known to be duplicates, near-duplicates
    or subsets.
    """
    # check & report if the file already exists
    output_verb = "Creating"
    if output_file.exists():
        output_verb = "Replacing"
    print(f"{output_verb} {output_file}")

    df = compile_metadata_df(poem_length=True)
    ref_corpus_names = {
        ref_corpus.corpus_id: ref_corpus.corpus_name for ref_corpus in all_corpora()
    }

    total_by_corpus = df["ref_corpus"].value_counts()
    totals = []
    for value, count in total_by_corpus.iter_rows():
        # row is a tuple of value, count;  convert reference corpus id to name
        totals.append(f"{ref_corpus_names[value]}: {count:,}")

    # when excerpt data is present, calculate & include aggregate totals
    if excerpts_df is not None:
        # get work-level aggregate excerpt totals
        # (only includes primary poem ids, not alt poem ids)
        excerpt_totals_df = excerpts_df.group_by("poem_id").agg(
            pl.col("excerpt_id").n_unique().alias("num_excerpts"),
            pl.col("ppa_work_id").n_unique().alias("num_ppa_works"),
            pl.col("page_id").n_unique().alias("num_ppa_pages"),
            # number of unique ppa authors would be nice, but requires joining ppa metadata
        )
        # combine the totals with poem metadata
        df = df.join(excerpt_totals_df, on="poem_id", how="left").with_columns(
            # fill any missing values with zeroes
            pl.col("num_excerpts").fill_null(pl.lit(0)),
            pl.col("num_ppa_works").fill_null(pl.lit(0)),
            pl.col("num_ppa_pages").fill_null(pl.lit(0)),
        )
    if poem_clusters_df is not None:
        df = df.join(
            poem_clusters_df.select("poem_id", "cluster_id"), on="poem_id", how="left"
        )
        # report on cluster ids and number of unique clusters (don't include nulls)
        df_with_cluster_ids = df.filter(pl.col.cluster_id.is_not_null())
        n_with_cluster_ids = df_with_cluster_ids.height
        n_uniq_clusters = df_with_cluster_ids["cluster_id"].n_unique()
        print(
            f"{n_with_cluster_ids:,} poems with cluster ids ({n_uniq_clusters:,} unique cluster{'s' if n_uniq_clusters != 1 else ''})"
        )

    print(f"{df.height:,} poem metadata entries ({'; '.join(totals)})")
    df.write_csv(output_file, include_bom=True)
