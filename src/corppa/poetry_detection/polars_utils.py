"""
Polars methods for working with excerpt data
"""

import pathlib

import polars as pl

from corppa.poetry_detection.core import MULTIVAL_DELIMITER, Excerpt, LabeledExcerpt

#: List of required fields for excerpt data
REQ_EXCERPT_FIELDS = set(Excerpt.fieldnames(required_only=True))
#: List required fields for labeled excerpt data
REQ_LABELED_EXCERPT_FIELDS = set(LabeledExcerpt.fieldnames(required_only=True))
#: All fields for labeled excerpts, in the expected order
LABELED_EXCERPT_FIELDS = LabeledExcerpt.fieldnames()
#: All fields for excerpts, in the expected order
EXCERPT_FIELDS = Excerpt.fieldnames()

#: dictionary of excerpt field names with associated data types
FIELD_TYPES = LabeledExcerpt.field_types()
# override set types with list, since Polars does not have a set type
FIELD_TYPES["detection_methods"] = pl.List
FIELD_TYPES["identification_methods"] = pl.List

#: Table of included PPA work-level field names and their names for use downstream
PPA_FIELDS = {
    "work_id": "ppa_work_id",
    "source_id": "ppa_source_id",
    "cluster_id": "ppa_cluster_id",
    "title": "ppa_work_title",
    "author": "ppa_work_author",
    "pub_year": "ppa_work_year",
    "source": "ppa_source",
    "collections": "ppa_collections",
}
#: Table of included reference poem field names and their names for use downstream
POEM_FIELDS = {
    "poem_id": "poem_id",
    "author": "poem_author",
    "title": "poem_title",
}


def has_poem_ids(df: pl.DataFrame) -> bool:
    """
    Check if a polars DataFrame has poem_id values. Returns true if 'poem_id'
    is present in the list of columns and there is at least one non-null value.
    """
    return bool("poem_id" in df.columns and df["poem_id"].count())


def fix_data_types(df):
    """
    Return a modified polars DataFrame with Labeled/Excerpt data with the appropriate
    data types. In particular, this handles converting multivalue method fields into
    polars Lists, since polars does not directly support sets.
    """
    # Get expected field types for columns that match Labeled/Excerpt fields
    df_types = {column: FIELD_TYPES.get(column) for column in df.columns}
    for c, ctype in df_types.items():
        # For list (set) types, split strings on multival delimiter to convert to list
        if ctype is pl.List:
            # if excerpt content is loaded from csv, it will have an inferred
            # type of string; split string content on our delimiter and
            # convert to list of string
            if df.schema[c] == pl.String:
                df = df.with_columns(pl.col(c).str.split(MULTIVAL_DELIMITER))
            # if a list column is loaded with no data (i.e., identification_methods
            # is present but has no content), it will have an inferred type of null
            # convert to list of string
            elif df.schema[c] == pl.Null:
                df = df.with_columns(pl.col(c).cast(pl.List(pl.String)))
        # For any other field type, cast the column to the expected type
        elif ctype is not None:
            df = df.with_columns(pl.col(c).cast(ctype))

    return df


def standardize_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Standardizes an excerpts dataframe so that it contains exactly the columns
    corresponding to the fields of :class:`corppa.poetry_detection.core.LabeledExcerpt`,
    in a standard order. This allows us to combine dataframes of excerpts consistently.
    Any fields not present will be added as a series of null values with the appropriate
    type. Any additional fields will be dropped.
    """
    df_columns = set(df.columns)
    expected_columns = set(LABELED_EXCERPT_FIELDS)
    missing_columns = expected_columns - df_columns
    # if any columns are missing, add them and make sure types are correct
    if missing_columns:
        df = df.with_columns([pl.lit(None).alias(field) for field in missing_columns])
        df = fix_data_types(df)

    # set consistent order to allow extending/appending
    return df.select(LABELED_EXCERPT_FIELDS)


def extract_page_meta(excerpts_df: pl.DataFrame) -> pl.DataFrame:
    """
    Extracts PPA page metadata (i.e., PPA work ID and page number) from each excerpt's
    ``page_id`` and combines it with the input excerpts ``DataFrame``.
    """
    out_df = excerpts_df.with_columns(
        ppa_work_id=pl.col("page_id").str.extract(r"^(.*)\.\d+$", 1),
        page_num=pl.col("page_id").str.extract(r"(\d+)$").cast(pl.Int64),
    )
    return out_df


def load_meta_df(file: pathlib.Path, fields_table: dict[str, str]) -> pl.DataFrame:
    """
    Loads specified metadata file (``CSV``) as a polars DataFrame. The columns of the
    resulting DataFrame are dictacted by the fields_table whose keys specify the
    metadata fields to be selected and whose values indicate what they should be
    renamed to.
    """
    # Check that file exists
    if not file.is_file():
        raise ValueError(f"Input file {file} does not exist")
    # Load in CSV
    df = pl.read_csv(file, infer_schema=False)
    # Optionally, select & rename fields
    if fields_table:
        # Check that all required fields exist
        missing_fields = fields_table.keys() - set(df.columns)
        if missing_fields:
            missing_str = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"Input CSV is missing the following required fields: {missing_str}"
            )
        # Select and rename fields
        df = df.select(fields_table.keys()).rename(fields_table)
    return df


def add_ppa_works_meta(
    excerpts_df: pl.DataFrame,
    ppa_works_csv: pathlib.Path,
) -> pl.DataFrame:
    """
    Combines found poem excerpt data (:class:`polars.DataFrame`) with PPA
    work-level metadata (``CSV``) and returns the resulting ``DataFrame``.
    """
    # Check for ppa_work_id field
    if "ppa_work_id" not in excerpts_df.columns:
        raise ValueError(
            "Missing ppa_work_id field; use extract_page_meta to extract it."
        )
    ppa_works_meta = load_meta_df(ppa_works_csv, PPA_FIELDS)
    return excerpts_df.join(ppa_works_meta, on="ppa_work_id", how="left")


def add_ref_poems_meta(
    excerpts_df: pl.DataFrame,
    ref_poem_meta: pathlib.Path,
) -> pl.DataFrame:
    """
    Combines found poem excerpt data (:class:`polars.DataFrame`) with reference
    poem metadata (``CSV``, possibly compressed) and returns the resulting
    ``DataFrame``.
    """
    join_fields = ["poem_id", "ref_corpus"]
    # Check for required fields
    missing_fields = set(join_fields) - set(excerpts_df.columns)
    if missing_fields:
        missing_str = ", ".join(sorted(missing_fields))
        raise ValueError(
            f"Input DataFrame missing the following required fields: {missing_str}"
        )
    poems_meta_df = load_meta_df(ref_poem_meta, POEM_FIELDS)
    return excerpts_df.join(poems_meta_df, on=join_fields, how="left")


def load_excerpts_df(
    excerpts_file: pathlib.Path,
    ppa_works_meta: None | pathlib.Path = None,
    ref_poems_meta: None | pathlib.Path = None,
) -> pl.DataFrame:
    """
    Load the specified excerpts file as a polars DataFrame, with column names
    based on fields in :class:`~corppa.poetry_detection.core.LabeledExcerpt`.

    Optionally, combine PPA work-level and reference poem metadata to the
    returned DataFrame.

    Currently, assume input file is a (possible compresed) `CSV` file.
    """
    # Load input file as a polars dataframe
    df = pl.read_csv(excerpts_file)

    # Check that we have the required fields for either Labeled/Excerpt data
    columns = set(df.columns)
    ## Treat presence of poem ids as indication of LabeledExcerpt
    if has_poem_ids(df):
        expected_type = "labeled excerpt"
        missing_columns = REQ_LABELED_EXCERPT_FIELDS - columns
    else:
        expected_type = "excerpt"
        missing_columns = REQ_EXCERPT_FIELDS - columns

    if missing_columns:
        raise ValueError(
            f"Input file {excerpts_file} is missing required {expected_type} fields: {', '.join(missing_columns)}"
        )

    # Set the correct data types for excerpt fields before returning
    df = fix_data_types(df)

    # Optionally, add PPA work-level metadata
    if ppa_works_meta:
        df = add_ppa_works_meta(df, ppa_works_meta)

    # Optionally, add reference poem metadata
    if ref_poems_meta:
        df = add_ref_poems_meta(df, ref_poems_meta)

    return df
