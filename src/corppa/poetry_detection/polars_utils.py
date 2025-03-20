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


def load_excerpts_df(input_file: pathlib.Path) -> pl.DataFrame:
    """
    Load the specified input file as a polars DataFrame, with column names
    based on fields in :class:`~corppa.poetry_detection.core.LabeledExcerpt`.

    Currently, assume input file is a `CSV` file.
    """
    # Load input file as a polars dataframe
    df = pl.read_csv(input_file)

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
            f"Input file {input_file} is missing required {expected_type} fields: {', '.join(missing_columns)}"
        )

    # Set the correct data types for excerpt fields before returning
    return fix_data_types(df)


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
