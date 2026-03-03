"""
PPA work-level metadata utilities
"""

import pathlib

import polars as pl


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


def load_ppa_works_df(file: pathlib.Path) -> pl.DataFrame:
    """
    Loads PPA work-level metadata (``CSV``) as a polars DataFrame;
    must include `work_id` field.
    """
    # Check that file exists
    if not file.is_file():
        raise ValueError(f"Input file {file} does not exist")
    # Load in CSV
    ppa_works_df = pl.read_csv(file)
    # We could check for expected fields, but the only
    # field *required* for joining with excerpts is work_id
    if "work_id" not in ppa_works_df.columns:
        raise ValueError("Input CSV is missing required `work_id` field")
    # Rename all fields to prefix with ppa_
    return ppa_works_df.rename(lambda column_name: f"ppa_{column_name}")


def add_ppa_works_meta(
    excerpts_df: pl.DataFrame,
    ppa_works_csv: pathlib.Path,
) -> pl.DataFrame:
    """
    Combine found poem excerpt data (:class:`polars.DataFrame`) with PPA
    work-level metadata (``CSV``) and returns the resulting ``DataFrame``.
    """
    # Check for ppa_work_id field; if not present, extract it
    if "ppa_work_id" not in excerpts_df.columns:
        excerpts_df = extract_page_meta(excerpts_df)
    ppa_works_meta = load_ppa_works_df(ppa_works_csv)
    return excerpts_df.join(ppa_works_meta, on="ppa_work_id", how="left")
