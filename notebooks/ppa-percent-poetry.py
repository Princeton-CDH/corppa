import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # What percent of PPA is poetry?

    According to the poetry we have detected,\* what percentage of PPA is poetry, and how does that change over time?

    \* We know we have not detected all of the poetry; data may include some false positives, but this is likely undercounting to some extent.
    """)
    return


@app.cell
def _():
    import pathlib

    import altair as alt
    import marimo as mo
    import polars as pl

    from corppa.config import get_config
    from corppa.poetry_detection.polars_utils import load_excerpts_df
    from corppa.poetry_detection.ppa_works import (
        extract_page_meta,
        load_ppa_works_df,
    )

    return (
        alt,
        extract_page_meta,
        get_config,
        load_excerpts_df,
        load_ppa_works_df,
        mo,
        pathlib,
        pl,
    )


@app.cell
def _(get_config, load_ppa_works_df, pathlib, pl):
    config_opts = get_config()
    data_dir = pathlib.Path(config_opts["compiled_dataset"]["data_dir"])

    # Create a dictionary of data files for lookup based on file base name without any extension
    # so that excerpts data can be .csv or compressed .csv.gz
    data_paths = {
        data_file.stem.split(".", 1)[0]: data_file for data_file in data_dir.iterdir()
    }

    ppa_meta_df = load_ppa_works_df(data_paths["ppa_work_metadata"]).with_columns(
        # add a boolean field for has poetry
        has_poetry=pl.col("ppa_num_excerpts").ne(0),
        # round years to decade
        ppa_pub_decade=pl.col("ppa_pub_year").floordiv(10).mul(10),
    )
    return data_paths, ppa_meta_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Works: what proportion of PPA works have poetry detected?

    Use the toggle to control whether the bar charts are normalized or not (show as raw counts or as the percent for that date).

    Mouse over to get counts; zoom in to see more detailed dates; double click to reset zoom.
    """)
    return


@app.cell
def _(mo):
    normalize_works_toggle = mo.ui.switch(label="Normalize", value=True)
    normalize_works_toggle
    return (normalize_works_toggle,)


@app.cell
def _(alt, mo, normalize_works_toggle, pl, ppa_meta_df):
    ppa_works_year_df = ppa_meta_df.group_by("ppa_pub_year", "has_poetry").agg(
        count=pl.len(),
    )

    mo.ui.altair_chart(
        alt.Chart(ppa_works_year_df)
        .mark_bar()
        .encode(
            x=alt.X("ppa_pub_year", title="Publication year").axis(
                format="r"
            ),  # no commas in years
            y=alt.Y("count", title="Number of works").stack(
                "normalize" if normalize_works_toggle.value else "zero"
            ),
            color=alt.Color("has_poetry", title="Has poetry"),
            tooltip=["count", "has_poetry"],
        )
        .properties(title="PPA works with detected poetry, by publication year")
        .interactive(bind_y=False)
    )
    return


@app.cell
def _(alt, mo, normalize_works_toggle, pl, ppa_meta_df):
    ppa_works_decade_df = ppa_meta_df.group_by("ppa_pub_decade", "has_poetry").agg(
        count=pl.len(),
    )

    mo.ui.altair_chart(
        alt.Chart(ppa_works_decade_df)
        .mark_bar(width=18)
        .encode(
            x=alt.X("ppa_pub_decade", title="Publication decade").axis(format="r"),
            y=alt.Y("count", title="Number of works").stack(
                "normalize" if normalize_works_toggle.value else "zero"
            ),
            color=alt.Color("has_poetry", title="Has poetry"),
            tooltip="count",
        )
        .properties(title="PPA works with detected poetry, by publication decade")
        .interactive(bind_y=False)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pages: what proportion of PPA pages have poetry detected?
    """)
    return


@app.cell
def _(mo):
    normalize_pages_toggle = mo.ui.switch(label="Normalize", value=True)
    normalize_pages_toggle
    return (normalize_pages_toggle,)


@app.cell
def _(data_paths, extract_page_meta, load_excerpts_df, pl, ppa_meta_df):
    # load the excerpts into a polars dataframe and extract ppa work/page ids
    excerpts_df = extract_page_meta(
        load_excerpts_df(
            data_paths["excerpts"]  # , ppa_works_meta=
        )
    )

    # aggregate by work id and count pages to determine number of pages with excerpts
    work_excerpt_pages_df = (
        excerpts_df.group_by("ppa_work_id")
        .agg(poetry_pages=pl.n_unique("page_id"))
        .join(
            # join with ppa metadata to get total page count
            ppa_meta_df.select(
                "ppa_work_id", "ppa_page_count", "ppa_pub_year", "ppa_pub_decade"
            ),
            on="ppa_work_id",
        )
        .with_columns(nonpoetry_pages=pl.col.ppa_page_count.sub(pl.col.poetry_pages))
    )
    return excerpts_df, work_excerpt_pages_df


@app.cell
def _(alt, mo, normalize_pages_toggle, pl, work_excerpt_pages_df):
    # aggregate before graphing with altair
    ppa_pages_year_df = (
        work_excerpt_pages_df.group_by("ppa_pub_year")
        .agg(
            # sum all the poetry and non-poetry pages
            pl.sum("poetry_pages", "nonpoetry_pages"),
        )
        # unpivot so we can stack and color the two different sets
        .unpivot(index="ppa_pub_year", value_name="num_pages")
        # add a boolean for more readability in the graph
        .with_columns(has_poetry=pl.col.variable.eq("poetry_pages"))
    )

    mo.ui.altair_chart(
        alt.Chart(ppa_pages_year_df)
        .mark_bar()
        .encode(
            x=alt.X("ppa_pub_year", title="Publication year").axis(
                format="r"
            ),  # no commas in years
            y=alt.Y("num_pages", title="Number of pages").stack(
                "normalize" if normalize_pages_toggle.value else "zero"
            ),
            color=alt.Color("has_poetry", title="Has poetry"),
            tooltip=["num_pages", "has_poetry"],
        )
        .properties(title="PPA pages with detected poetry, by publication year")
        .interactive(bind_y=False)
    )
    return


@app.cell
def _(alt, mo, normalize_pages_toggle, pl, work_excerpt_pages_df):
    # same as above, but for decade instead of year
    ppa_pages_decade_df = (
        work_excerpt_pages_df.group_by("ppa_pub_decade")
        .agg(
            # sum all the poetry and non-poetry pages
            pl.sum("poetry_pages", "nonpoetry_pages"),
        )
        # unpivot so we can stack and color the two different sets
        .unpivot(index="ppa_pub_decade", value_name="num_pages")
        # add a boolean for more readability in the graph
        .with_columns(has_poetry=pl.col.variable.eq("poetry_pages"))
    )

    mo.ui.altair_chart(
        alt.Chart(ppa_pages_decade_df)
        .mark_bar(width=18)
        .encode(
            x=alt.X("ppa_pub_decade", title="Publication decade").axis(
                format="r"
            ),  # no commas in years
            y=alt.Y("num_pages", title="Number of pages").stack(
                "normalize" if normalize_pages_toggle.value else "zero"
            ),
            color=alt.Color("has_poetry", title="Has poetry"),
            tooltip=["num_pages", "has_poetry"],
        )
        .properties(title="PPA pages with detected poetry, by publication decade")
        .interactive(bind_y=False)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Text: what proportion of PPA text has been detected as poetry?

    If we look at page text at the character level, what portion of the text has been included in any of our detected excerpt spans?
    """)
    return


@app.cell
def _(mo):
    normalize_text_toggle = mo.ui.switch(label="Normalize", value=True)
    normalize_text_toggle
    return (normalize_text_toggle,)


@app.cell
def _(pl):
    # load the PPA page data that was used to generate this found poems dataset
    ppa_pages_df = (
        pl.scan_ndjson(
            # TODO: use a config path?
            "../ppa-django/fulltext-production/ppa_corpus_2026-01-07_091133/ppa_pages.jsonl.gz"
        )
        # calculate the length of the text
        .with_columns(text_len=pl.col("text").str.len_chars())
        .rename({"id": "page_id"})  # rename for joining with excerpt data later
        .select("page_id", "work_id", "text_len")  # limit to the fields we need
        .collect()
    )
    return (ppa_pages_df,)


@app.cell
def _(excerpts_df, pl):
    # collapse excerpts with any overlap to a single span so we can calculate the total number of characters
    # covered by any of the merged spans

    excerpt_page_chars_df = (
        # sort by page and span start
        excerpts_df.sort("page_id", "ppa_span_start")
        .select("page_id", "ppa_span_start", "ppa_span_end", "detection_methods")
        .with_columns(
            # Use shift and cumulative max to determine if current span
            # has any overlap with previous spans or is the beginning of a new group.
            # shift(1) gets previous row; fill null for first row (which has no previous row),
            # and calculate current max span end for this page.
            new_group=(
                pl.col("ppa_span_start")
                > pl.col("ppa_span_end").shift(1).fill_null(-1).cum_max()
            )
            .cast(pl.Int32)  # cast to int gives 1 or 0 to indicate new group
            .over("page_id")  # limit to spans on a single page
        )
        .with_columns(
            # because new_group is 1 or 0, cumulative sum gives each group on a page a unique groep id
            pl.col("new_group").cum_sum().alias("group_id").over("page_id")
        )
        .group_by("page_id", "group_id")
        .agg(
            # group by page id and group id and get the smallest start and largest end
            # to get the outer bounds of the overlapping spans
            pl.col("ppa_span_start").min(),
            pl.col("ppa_span_end").max(),
        )
        .group_by("page_id")
        .agg(
            # calculate the number of characters covered by all merged spans for each page
            poetry_chars=(pl.col("ppa_span_end") - pl.col("ppa_span_start")).sum()
        )
    )
    return (excerpt_page_chars_df,)


@app.cell
def _(excerpt_page_chars_df, pl, ppa_meta_df, ppa_pages_df):
    # join merged span char length data with page data to determine poetry/nonpoetry chars
    text_poetrylen_df = (
        ppa_pages_df.join(excerpt_page_chars_df, on="page_id", how="left")
        .with_columns(pl.col("poetry_chars").fill_null(0))
        .with_columns(nonpoetry_chars=pl.col("text_len").sub(pl.col("poetry_chars")))
        .join(
            ppa_meta_df.select("ppa_work_id", "ppa_pub_year", "ppa_pub_decade"),
            left_on="work_id",
            right_on="ppa_work_id",
        )
    )
    return (text_poetrylen_df,)


@app.cell
def _(alt, mo, normalize_text_toggle, pl, text_poetrylen_df):
    # aggregate before graphing with altair
    text_poetrylen_year_df = (
        text_poetrylen_df.group_by("ppa_pub_year")
        .agg(
            # sum all the poetry and non-poetry characters
            pl.sum("poetry_chars", "nonpoetry_chars"),
        )
        # unpivot so we can stack and color the two different sets
        .unpivot(index="ppa_pub_year", value_name="text_len")
        # # add a boolean for more readability in the graph
        .with_columns(has_poetry=pl.col.variable.eq("poetry_chars"))
    )

    mo.ui.altair_chart(
        alt.Chart(text_poetrylen_year_df)
        .mark_bar()
        .encode(
            x=alt.X("ppa_pub_year", title="Publication year").axis(
                format="r"
            ),  # no commas in years
            y=alt.Y("text_len", title="Number of characters").stack(
                "normalize" if normalize_text_toggle.value else "zero"
            ),
            color=alt.Color("has_poetry", title="Has poetry"),
            tooltip=["text_len", "has_poetry"],
        )
        .properties(title="PPA text detected as poetry, by publication year")
        .interactive(bind_y=False)
    )
    return


@app.cell
def _(alt, mo, normalize_text_toggle, pl, text_poetrylen_df):
    # aggregate before graphing with altair
    text_poetrylen_decade_df = (
        text_poetrylen_df.group_by("ppa_pub_decade")
        .agg(
            # sum all the poetry and non-poetry characters
            pl.sum("poetry_chars", "nonpoetry_chars"),
        )
        # unpivot so we can stack and color the two different sets
        .unpivot(index="ppa_pub_decade", value_name="text_len")
        # # add a boolean for more readability in the graph
        .with_columns(has_poetry=pl.col.variable.eq("poetry_chars"))
    )

    mo.ui.altair_chart(
        alt.Chart(text_poetrylen_decade_df)
        .mark_bar(width=18)
        .encode(
            x=alt.X("ppa_pub_decade", title="Publication decade").axis(
                format="r"
            ),  # no commas in years
            y=alt.Y("text_len", title="Number of characters").stack(
                "normalize" if normalize_text_toggle.value else "zero"
            ),
            color=alt.Color("has_poetry", title="Has poetry"),
            tooltip=["text_len", "has_poetry"],
        )
        .properties(title="PPA text detected as poetry, by publication decade")
        .interactive(bind_y=False)
    )
    return


if __name__ == "__main__":
    app.run()
