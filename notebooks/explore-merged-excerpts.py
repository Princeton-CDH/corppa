import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Merged excerpts

    Preliminary notebook for reviewing merged excerpts.
    """)
    return


@app.cell
def _():
    import pathlib

    import marimo as mo
    import polars as pl

    from corppa.config import get_config
    from corppa.poetry_detection.polars_utils import load_excerpts_df

    return get_config, load_excerpts_df, mo, pathlib, pl


@app.cell
def _(get_config, pathlib):
    config_opts = get_config()

    data_dir = pathlib.Path(config_opts["compiled_dataset"]["data_dir"])
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
def _(data_paths, load_excerpts_df, pl):
    excerpts_df = load_excerpts_df(
        data_paths["excerpts"],
        ppa_works_meta=data_paths["ppa_work_metadata"],
        ref_poems_meta=data_paths["poem_meta"],
    )

    # identify merged excerpts by presence of merge note
    merged_ex_df = excerpts_df.filter(pl.col("notes").str.contains("merge"))
    merged_ex_df
    return excerpts_df, merged_ex_df


@app.cell
def _():
    return


@app.cell
def _(excerpts_df, pl):
    # check for excerpts with multiple poem identifications
    multi_poem_id_df = excerpts_df.filter(pl.col("alt_poem_ids").list.len().gt(0))
    multi_poem_id_df
    return (multi_poem_id_df,)


@app.cell
def _(excerpts_df, merged_ex_df, mo, multi_poem_id_df):
    # summarize
    mo.md(
        f"""
        {merged_ex_df.height:,} out of {excerpts_df.height:,} total excerpts are merges ({merged_ex_df.height/excerpts_df.height * 100:.2f}%).
    
        {multi_poem_id_df.height:,} out of {merged_ex_df.height:,} merged excerpts have multiple poem ids ({multi_poem_id_df.height/merged_ex_df.height * 100:.2f}%)"""
    )
    return


if __name__ == "__main__":
    app.run()
