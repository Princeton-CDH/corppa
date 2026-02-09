import os.path
import pathlib
import tarfile
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
    text_dir: pathlib.Path
    metadata_path: pathlib.Path | str

    def get_config_opts(self) -> dict:
        """Load reference corpus-specific configuration options from
        corppa config file. Must include the reference corpora base directory;
        may include overrides for non-default paths.
        """
        config_opts = get_config()
        if "reference_corpora" not in config_opts:
            raise ValueError(
                "Configuration error: required section 'reference_corpora' not found"
            )
        try:
            base_dir = pathlib.Path(config_opts["reference_corpora"]["base_dir"])
        except KeyError:
            raise ValueError(
                "Configuration error: required 'reference_corpora.base_dir' not found"
            )
        # always include the reference_corpora base directory as a path
        # if not absolute, path is relative to data ingredients dir
        if not base_dir.is_absolute():
            try:
                ingredients_dir = pathlib.Path(config_opts["data_ingredients_dir"])
            except KeyError:
                raise ValueError(
                    "Configuration error: 'reference_corpora.base_dir' is relative but 'data_ingredients_dir' is not configured"
                )
            base_dir = ingredients_dir / base_dir

        corpus_opts = {"base_dir": base_dir}
        # include any options for this specific corpus
        corpus_opts.update(config_opts["reference_corpora"].get(self.corpus_id, {}))
        return corpus_opts

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
    provided as a set of text files in a directory or tar.gz.
    On initialization, configures data path based on
    configured based dir and corpus default or any overrides, and validates
    that the path exists and is a directory.
    Provides :meth:`get_text_corpus` for generating text corpus from
    the file system."""

    def __init__(self):
        # get configuration for this corpus
        config_opts = self.get_config_opts()

        # get text directory for this reference corpus from app configuration
        if "text_dir" in config_opts:
            self.text_dir = pathlib.Path(config_opts["text_dir"])
        # if text dir is not absolute, assume relative to ref_corpus base dir
        if not self.text_dir.is_absolute():
            self.text_dir = config_opts["base_dir"] / self.text_dir

        if not self.text_dir.exists():
            raise ValueError(
                f"Configuration error: {self.corpus_name} path {self.text_dir} does not exist"
            )
        # TODO: allow tar.gz here; determine which and set a flag?
        if not self.text_dir.is_dir() and not (
            self.text_dir.is_file() and self.text_dir.name.endswith(".tar.gz")
        ):
            raise ValueError(
                f"Configuration error: {self.corpus_name} path {self.text_dir} is not a directory or a tar.gz"
            )

    def get_text_corpus(
        self, disable_progress: bool = True
    ) -> Generator[dict[str, str]]:
        # if text_dir is tarball, raise not implemented error
        if not self.text_dir.is_dir():
            raise NotImplementedError(
                "text corpus generation is not supported for tar.gz; configure a directory"
            )
        # build_text_corpus method returns id, so rename id to poem_id
        yield from (
            {"poem_id": p["id"], "text": p["text"]}
            for p in build_text_corpus(self.text_dir, disable_progress=disable_progress)
        )


class InternetPoems(LocalTextCorpus):
    """Curated corpus of poems with plain text content sourced from
    the internet, for high priority sources known to occur in excerpts,
    including full text of Shakespeare's plays. Metadata is inferred based on
    filename, with a naming convention of `Firstname-Lastname_Poem-Title.txt`.
    The filename without extension is used as the `poem_id`.
    """

    #: id for this reference corpus: internet_poems
    corpus_id: str = "internet_poems"
    corpus_name: str = "Internet Poems"
    # inherits text_dir path

    # no init/validation needed beyond that provided by LocalTextCorpus

    def get_metadata_df(self) -> pl.DataFrame:
        metadata = []

        # if configured text_dir is a directory, get list of names
        # from the filesystem
        if self.text_dir.is_dir():
            text_ids = [file.stem for file in self.text_dir.glob("*.txt")]
        # otherwise, get from tar archive list
        else:
            with tarfile.open(str(self.text_dir), "r:gz") as text_archive:
                text_ids = [
                    os.path.splitext(os.path.basename(name))[0]
                    for name in text_archive.getnames()
                    if name.endswith(".txt")
                ]

        # filename without extension is poem identifier
        for poem_id in text_ids:
            # filename format:
            #   Firstname-Lastname_Poem-Title.txt
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

    #: id for this reference corpus: chadwyck-healey
    corpus_id: str = "chadwyck-healey"
    corpus_name: str = "Chadwyck-Healey"
    # inherits text_dir path

    def __init__(self):
        # use LocalTextCorpus init to configure and validate text_dir
        super().__init__()
        # get configuration to set metadata path
        config_opts = self.get_config_opts()

        self.metadata_path = pathlib.Path(config_opts["metadata_path"])
        # if metadata path is not absolute, assume relative to ref_corpus base dir
        if not self.metadata_path.is_absolute():
            self.metadata_path = config_opts["base_dir"] / self.metadata_path
        if not (self.metadata_path.exists() and self.metadata_path.is_file()):
            raise ValueError(
                f"Configuration error: {self.corpus_name} metadata {self.metadata_path} does not exist"
            )

    def get_metadata_df(self) -> pl.DataFrame:
        # disable schema inference; the fields we care about are all strings
        return (
            pl.read_csv(self.metadata_path, infer_schema=False)
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
    #: URL or local path for metadata (can pull from Google Sheets published csv)
    metadata_path: str

    def __init__(self):
        # get configuration for this corpus
        config_opts = self.get_config_opts()
        # set data path from config file and check that path exists
        try:
            self.metadata_path = config_opts["metadata_path"]
        except KeyError:
            raise ValueError(
                f"Configuration error: {self.corpus_name} 'metadata_path' is not set"
            )

    def get_metadata_df(self) -> pl.DataFrame:
        # polars can load csv directly from a url
        return pl.read_csv(self.metadata_path, schema=METADATA_SCHEMA).with_columns(
            ref_corpus=pl.lit(self.corpus_id)
        )

    # this is a metadata-only corpus; get_text_corpus is intentionally not implemented


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
        poem_metadata.extend(ref_corpus.get_metadata_df())
    return poem_metadata


def save_poem_metadata(output_file: pathlib.Path):
    """Generate and save compiled poetry metadata as a data file in the
    poem dataset.
    """
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
