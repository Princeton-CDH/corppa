import marimo

__generated_with = "0.13.7"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Exploring Poetry Excerpt Data

    This notebook is for exploring the poetry excerpt data to aid data play at Ends of Prosody.
    """
    )
    return


@app.cell
def _():
    import pathlib

    import marimo as mo
    import polars as pl

    from corppa.config import get_config
    from corppa.poetry_detection.polars_utils import (
        add_ppa_works_meta,
        add_ref_poems_meta,
        extract_page_meta,
        load_excerpts_df,
    )

    return (
        add_ppa_works_meta,
        add_ref_poems_meta,
        extract_page_meta,
        get_config,
        load_excerpts_df,
        mo,
        pathlib,
        pl,
    )


@app.cell
def _(get_config, pathlib):
    # load local configuration options to get path to data
    config_opts = get_config()

    data_dir = pathlib.Path(config_opts["poem_dataset"]["data_dir"])
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(
            f"Data directory {data_dir} not found. "
            + "\nCheck your configuration file, and remember to use an absolute path for the poem dataset data directory."
        )
    else:
        print(f"Data will be loaded from {data_dir}")

    # Create a dictionary of data files for lookup based on file base name without any extension
    # so that excerpts data can be .csv or compressed .csv.gz
    data_paths = {
        data_file.stem.split(".", 1)[0]: data_file for data_file in data_dir.iterdir()
    }
    return (data_paths,)


@app.cell
def _(
    add_ppa_works_meta,
    add_ref_poems_meta,
    data_paths,
    extract_page_meta,
    load_excerpts_df,
):
    # load the excerpts into a polars dataframe
    excerpts_df = load_excerpts_df(data_paths["excerpts"])
    # add poem metadata
    excerpts_df = add_ref_poems_meta(excerpts_df, data_paths["poem_meta"])
    # add PPA metadata
    excerpts_df = add_ppa_works_meta(
        extract_page_meta(excerpts_df), data_paths["ppa_work_metadata"]
    )
    excerpts_df
    return (excerpts_df,)


@app.cell
def _(excerpts_df, pl):
    # summarize the data
    total_excerpts = excerpts_df.height
    labeled_excerpts = excerpts_df.filter(pl.col("poem_id").is_not_null())
    total_labeled_excerpts = labeled_excerpts.height
    print(f"""{total_excerpts:,} total excerpts in combined data
    {total_labeled_excerpts:,} labeled excerpts; {total_excerpts - total_labeled_excerpts:,} unlabeled ({((total_excerpts - total_labeled_excerpts) / total_excerpts) * 100:.1f}%)""")
    return


@app.cell
def _(excerpts_df, pl):
    detectmethod_counts = excerpts_df["detection_methods"].value_counts()
    idmethod_counts = excerpts_df.filter(pl.col("poem_id").is_not_null())[
        "identification_methods"
    ].value_counts()
    print("Total by detection method:")
    for value, count in detectmethod_counts.iter_rows():
        # row is a tuple of value, count; vlaue is a list
        print(f"\t{','.join(value)}: {count:,}")
    print("Total by identification method:")
    for value, count in idmethod_counts.iter_rows():
        # row is a tuple of value, count
        print(f"\t{','.join(value)}: {count:,}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## View page-level statistics""")
    return


@app.cell
def _(excerpts_df, pl):
    excerpts_by_page = (
        excerpts_df.group_by("page_id")
        .agg(
            pl.count("excerpt_id").alias("num_excerpts"),
            pl.n_unique("excerpt_id").alias("num_distinct_excerpts"),
            pl.n_unique("poem_id").alias("num_poems"),
            pl.first("ppa_work_title"),
            pl.first("ppa_work_author"),
            pl.first("ppa_work_year"),
        )
        .sort("num_distinct_excerpts", descending=True)
    )

    excerpts_by_page
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## View work-level statistics""")
    return


@app.cell
def _(excerpts_df, pl):
    excerpts_by_work = (
        excerpts_df.group_by("ppa_work_id")
        .agg(
            pl.count("excerpt_id").alias("num_excerpts"),
            pl.n_unique("excerpt_id").alias("num_distinct_excerpts"),
            pl.n_unique("poem_id").alias("num_poems"),
            pl.n_unique("page_id").alias("num_pages_with_poems"),
            pl.first("ppa_work_title"),
            pl.first("ppa_work_author"),
            pl.first("ppa_work_year"),
        )
        .sort("num_distinct_excerpts", descending=True)
    )

    excerpts_by_work
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## View poetry-level statistics""")
    return


@app.cell
def _(excerpts_df, pl):
    excerpts_by_poem = (
        excerpts_df.group_by("poem_id")
        .agg(
            pl.count("excerpt_id").alias("num_excerpts"),
            pl.n_unique("excerpt_id").alias("num_distinct_excerpts"),
            pl.n_unique("page_id").alias("n_pages"),
            pl.n_unique("ppa_work_id").alias("n_works"),
            pl.first("poem_title"),
            pl.first("poem_author"),
        )
        .sort("num_distinct_excerpts", descending=True)
    )

    excerpts_by_poem
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## View poem author statistics""")
    return


@app.cell
def _(excerpts_df, pl):
    excerpts_by_poem_author = (
        excerpts_df.group_by("poem_author")
        .agg(
            pl.count("excerpt_id").alias("num_excerpts"),
            pl.n_unique("excerpt_id").alias("num_distinct_excerpts"),
            pl.n_unique("poem_id").alias("n_poems"),
            pl.n_unique("page_id").alias("n_pages"),
            pl.n_unique("ppa_work_id").alias("n_works"),
        )
        .sort("num_distinct_excerpts", descending=True)
    )

    excerpts_by_poem_author
    return


if __name__ == "__main__":
    app.run()
