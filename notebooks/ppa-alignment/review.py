import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Review image/text shifted alignment

    This notebook allows you to review and visualize alignment results for specific volumes.  It runs the same shifted alignment method used by the dataset_prep script, and then generates an altair visualization displaying the sequential alignment between the two versions of the text, with a hover preview of the text content for comparison.

    To run it, you must have a local copy of a handful of HathiTrust image/text dataset folders and zip files and a copy of the `ppa_pages.jsonl` data before alignment has been run.  The `work_id` select list has been pre-populated with a list of works with known page shift.
    """)
    return


@app.cell
def _(mo):
    image_dir_ui = mo.ui.file_browser(
        initial_path=".",
        selection_mode="directory",
        multiple=False,
        label="Select directory with HathiTrust image dataset folders and zip files.",
        restrict_navigation=False,
    )

    ppa_pages_ui = mo.ui.file_browser(
        initial_path=".",
        selection_mode="file",
        multiple=False,
        label="Select PPA pages.jsonl BEFORE image alignment",
        restrict_navigation=False,
        filetypes=[".jsonl", ".gz"],
    )

    mo.vstack([image_dir_ui, ppa_pages_ui])
    return image_dir_ui, ppa_pages_ui


@app.cell
def _():
    import marimo as mo

    # list of works with known alignment - must have zipfile present locally to run this
    works = [
        "hvd.hn1dlr",
        "hvd.32044048962955-p145",
        "hvd.32044050831999-p50",
        "hvd.32044092711431-p469",
        "mdp.39015030933512",
        "mdp.39015030934866",
        "mdp.39015031107785",
        "mdp.39015058693683",
        "mdp.39015059390321",
        "mdp.39015059409386",
        "mdp.39015059409410",
        "nyp.33433066585435",
        "yale.39002004065844-p76",
    ]

    select_work = mo.ui.multiselect(
        # preselect first one so we don't have to worry about it being unset
        options=works,
        max_selections=1,
        value=[works[0]],
        label="Select a work",
    )
    select_work
    return mo, select_work


@app.cell
def _(image_dir_ui, mo, ppa_pages_ui, select_work):
    import polars as pl

    from corppa.utils.dataset_prep import (
        get_ht_zipfile_path,
        plot_alignment,
        review_alignment,
    )

    mo.stop(
        not image_dir_ui.value,
        mo.md("Select an image directory in the file browser."),
    )

    mo.stop(
        not ppa_pages_ui.value,
        mo.md("Select ppa_pages.jsonl file in the file browser."),
    )

    work_id = select_work.value[0]

    data_dir = image_dir_ui.value[0].path

    all_pages_path = ppa_pages_ui.value[0].path

    print(
        f"Accessing HathiTrust images in {data_dir}.\nLoading page data from {all_pages_path}"
    )
    # load original pages
    orig_pages_df = (
        pl.scan_ndjson(all_pages_path).filter(work_id=work_id).collect().sort("order")
    )
    orig_pages_df

    zip_path = get_ht_zipfile_path(work_id, data_dir)
    mo.stop(
        not zip_path.exists(),
        mo.md(f"Zip file for selected work was not found: {zip_path}"),
    )

    df = review_alignment(work_id, orig_pages_df, zip_path)
    plot_alignment(df)
    return df, pl


@app.cell
def _(df, pl):
    df.filter(~pl.col.is_matched)  # inspect gaps
    return


if __name__ == "__main__":
    app.run()
