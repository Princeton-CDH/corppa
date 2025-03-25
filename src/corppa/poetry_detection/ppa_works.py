"""
Module containing PPA work-level metadata utilities
"""

from pathlib import Path

import polars as pl

# Table of included PPA work-level field names and their fieldnames for use downstream
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


def load_ppa_works_df(file: Path) -> pl.DataFrame:
    """
    Loads PPA work-level metadata (``CSV``) as a polars DataFrame containing only the
    fields in :data:`corppa.poetry_detection.ppa_works.PPA_FIELDS` that have been
    renamed to its corresponding values.
    """
    # Check that file exists
    if not file.is_file():
        raise ValueError(f"Input file {file} does not exist")
    # Load in CSV
    ppa_works_df = pl.read_csv(file)
    # Check that all required fields exist
    missing_fields = PPA_FIELDS.keys() - set(ppa_works_df.columns)
    if missing_fields:
        missing_str = ", ".join(sorted(missing_fields))
        raise ValueError(
            f"Input CSV is missing the following required fields: {missing_str}"
        )
    # Select and rename fields
    ppa_works_df = ppa_works_df.select(PPA_FIELDS.keys()).rename(PPA_FIELDS)
    return ppa_works_df


def add_ppa_work_meta(
    excerpts_df: pl.DataFrame,
    ppa_works_csv: Path,
) -> pl.DataFrame:
    """
    Combines found poem excerpt data (:class:`polars.DataFrame`) with PPA
    work-level metadata (``CSV``) and returns the resulting ``DataFrame``.
    """
    ppa_works_meta = load_ppa_works_df(ppa_works_csv)
    return excerpts_df.join(ppa_works_meta, on=["ppa_work_id"])
