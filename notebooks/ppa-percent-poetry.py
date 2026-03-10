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

    Broad overview: this is just looking at works with any excerpts detected and those zero excerpts.

    ---
    Use the toggle to control whether the bar charts are normalized or not (show as raw counts or as the percent for that date).

    Mouse over to get counts; zoom in to see more detailed dates; double click to reset zoom.
    """)
    return


@app.cell
def _(mo, pl, ppa_meta_df):
    # output some numbers
    total_works = ppa_meta_df.height
    total_with_poetry = ppa_meta_df.filter(pl.col.has_poetry).height
    # check numbers for single poem / single poet / single excerpt
    with_1excerpt = ppa_meta_df.filter(pl.col.ppa_num_excerpts.eq(1)).height
    single_poem_works = ppa_meta_df.filter(
        pl.col.ppa_num_excerpts.gt(10), pl.col.ppa_num_poems.eq(1)
    ).height
    single_poet_works = ppa_meta_df.filter(
        pl.col.ppa_num_excerpts.gt(10), pl.col.ppa_num_poets.eq(1)
    ).height

    max_num_excerpts = ppa_meta_df["ppa_num_excerpts"].max()
    max_num_poems = ppa_meta_df["ppa_num_poems"].max()
    max_num_poets = ppa_meta_df["ppa_num_poets"].max()

    mo.md(f"""Of {total_works:,} total PPA works:

    - {total_with_poetry:,} have poetry detected ({total_with_poetry/total_works * 100:.1f}%)
    - {with_1excerpt:,} have just one excerpt detected ({with_1excerpt/total_works * 100:.1f}%)
    - {single_poem_works:,} have at least 10 excerpts, all from only one poem ({single_poem_works/total_works * 100:.1f}%)
    - {single_poet_works:,} have at least 10 excerpts, all from only one poet ({single_poet_works/total_works * 100:.1f}%)


    Maximum numbers for a single PPA work * (likely includes duplicates)

    - {max_num_excerpts:,} excerpts
    - {max_num_poems:,} poems 
    - {max_num_poets:,} poets  

    """)
    return


@app.cell
def _(alt, mo, pl, ppa_meta_df):
    ppa_works_year_df = ppa_meta_df.group_by("ppa_pub_year", "has_poetry").agg(
        count=pl.len()
    )

    def chart_has_poetry(
        df,
        field,
        field_title,
        y_axis_title,
        normalize=True,
        mark_opts=None,
    ):
        if mark_opts is None:
            mark_opts = {}
        return (
            alt.Chart(df)
            .mark_bar(**mark_opts)
            .encode(
                x=alt.X(field, title=field_title).axis(
                    format="r"
                ),  # no commas in years
                y=alt.Y("count", title=y_axis_title).stack(
                    "normalize" if normalize else "zero"
                ),
                color=alt.Color("has_poetry", title="Has poetry"),
                tooltip=["count", "has_poetry"],
            )
            .properties(
                height=150,
            )
        )

    works_chart_normalized = chart_has_poetry(
        ppa_works_year_df, "ppa_pub_year", "Publication year", "Works"
    )
    works_chart_count = chart_has_poetry(
        ppa_works_year_df,
        "ppa_pub_year",
        "Publication year",
        "Works",
        normalize=False,
    )

    mo.ui.altair_chart(
        (works_chart_normalized & works_chart_count).properties(
            title="PPA works with detected poetry, by year"
        )
    )
    return (chart_has_poetry,)


@app.cell
def _(chart_has_poetry, mo, pl, ppa_meta_df):
    ppa_works_decade_df = ppa_meta_df.group_by("ppa_pub_decade", "has_poetry").agg(
        count=pl.len(),
    )

    decade_works_chart_normalized = chart_has_poetry(
        ppa_works_decade_df,
        "ppa_pub_decade",
        field_title=None,
        y_axis_title="Works",
        mark_opts={"width": 18},
    )
    decade_works_chart_count = chart_has_poetry(
        ppa_works_decade_df,
        "ppa_pub_decade",
        "Publication decade",
        y_axis_title="Works",
        normalize=False,
        mark_opts={"width": 18},
    )

    mo.ui.altair_chart(
        (decade_works_chart_normalized & decade_works_chart_count).properties(
            title="PPA works with detected poetry, by decade"
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pages: what proportion of PPA pages have poetry detected?

    If we look at the page level, how many pages in PPA have any poetry detected?
    """)
    return


@app.cell
def _(mo):
    normalize_pages_toggle = mo.ui.switch(label="Normalize", value=True)
    normalize_pages_toggle
    return


@app.cell
def _(data_paths, extract_page_meta, load_excerpts_df, pl, ppa_meta_df):
    # load the excerpts into a polars dataframe and extract ppa work/page ids
    excerpts_df = extract_page_meta(
        load_excerpts_df(
            data_paths["excerpts"]  # , ppa_works_meta=
        ).with_columns(
            # calculate span length for later
            ppa_span_len=pl.col.ppa_span_end - pl.col.ppa_span_start
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
def _(excerpts_df, mo, pl, ppa_meta_df):
    # output some numbers for page
    excerpt_pages_df = excerpts_df.group_by("page_id").agg(
        num_excerpts=pl.len(), num_poems=pl.n_unique("poem_id")
    )

    total_pages = ppa_meta_df["ppa_page_count"].sum()
    num_pages_with_poetry = excerpt_pages_df.height
    # check numbers for single poem / single poet / single excerpt
    pages_with_1excerpt = excerpt_pages_df.filter(pl.col.num_excerpts.eq(1)).height
    single_poem_pages = excerpt_pages_df.filter(pl.col.num_poems.eq(1)).height
    # we don't have poet count because we haven't joined poem metadata yet
    # single_poet_pages = excerpt_pages_df.filter(pl.col.ppa_num_poets.eq(1)).height

    page_max_num_excerpts = excerpt_pages_df["num_excerpts"].max()
    page_max_num_poems = excerpt_pages_df["num_poems"].max()
    # max_num_poets = excerpt_pages_df["ppa_num_poets"].max()

    mo.md(f"""Of {total_pages:,} total pages in PPA:

    - {num_pages_with_poetry:,} pages have poetry detected ({num_pages_with_poetry/total_pages * 100:.1f}%)
    - {pages_with_1excerpt:,} pages have just one excerpt detected ({pages_with_1excerpt/total_pages * 100:.1f}%)
    - {single_poem_pages:,} pages have excerpts from a single poem ({single_poem_pages/total_pages * 100:.1f}%)

    Maximum numbers for a single PPA page * (likely includes duplicates)

    - {page_max_num_excerpts:,} excerpts
    - {page_max_num_poems:,} poems
    """)
    return (total_pages,)


@app.cell
def _(chart_has_poetry, mo, pl, work_excerpt_pages_df):
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
        .rename({"num_pages": "count"})
    )

    pages_chart_normalized = chart_has_poetry(
        ppa_pages_year_df,
        "ppa_pub_year",
        field_title=None,
        y_axis_title="Pages",
    )
    pages_chart_count = chart_has_poetry(
        ppa_pages_year_df,
        "ppa_pub_year",
        "Publication year",
        normalize=False,
        y_axis_title="Pages",
    )

    mo.ui.altair_chart(
        (pages_chart_normalized & pages_chart_count).properties(
            title="PPA pages with detected poetry, by work publication year"
        )
    )
    return


@app.cell
def _(chart_has_poetry, mo, pl, work_excerpt_pages_df):
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
        .rename({"num_pages": "count"})
    )

    pages_decade_chart_normalized = chart_has_poetry(
        ppa_pages_decade_df,
        "ppa_pub_decade",
        field_title=None,
        y_axis_title="Pages",
        mark_opts={"width": 18},
    )
    pages_decade_chart_count = chart_has_poetry(
        ppa_pages_decade_df,
        "ppa_pub_decade",
        "Publication year",
        y_axis_title="Pages",
        normalize=False,
        mark_opts={"width": 18},
    )

    mo.ui.altair_chart(
        (pages_decade_chart_normalized & pages_decade_chart_count).properties(
            title="PPA pages with detected poetry, by work publication decade"
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Text: what proportion of PPA text has been detected as poetry?

    If we look at page text at the character level, what portion of the text is been included in any of the detected poetry excerpt spans?
    """)
    return


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
def _(excerpts_df):
    excerpts_df
    return


@app.cell
def _(excerpts_df, pl):
    # collapse excerpts with any overlap to a single span so we can calculate the total number of characters
    # covered by any of the merged spans

    merged_excerpts_df = (
        # sort by page and span start
        excerpts_df.sort("page_id", "ppa_span_start")
        .select("page_id", "ppa_span_start", "ppa_span_end", "detection_methods")
        .with_columns(
            # Use shift and cumulative max to determine if current span
            # has any overlap with previous spans or is the beginning of a new group.
            # shift(1) gets previous row; fill null for first row (which has no previous row),
            # and calculate current max span end for this page.
            # NOTE: we use >= because span end is exclusive (i.e., is not included in the range)
            new_group=(
                pl.col("ppa_span_start")
                >= pl.col("ppa_span_end").shift(1).fill_null(-1).cum_max()
            )
            .cast(pl.Int32)  # cast to int gives 1 or 0 to indicate new group
            .over("page_id")  # limit to spans on a single page
        )
        .with_columns(
            # because new_group is 1 or 0, cumulative sum gives each group on a page a unique group id
            pl.col("new_group").cum_sum().alias("group_id").over("page_id")
        )
        .group_by("page_id", "group_id")
        .agg(
            # group by page id and group id and get the smallest start and largest end
            # to get the outer bounds of the overlapping spans
            pl.col("ppa_span_start").min(),
            pl.col("ppa_span_end").max(),
        )
        .with_columns(ppa_span_len=pl.col.ppa_span_end - pl.col.ppa_span_start)
    )

    excerpt_page_chars_df = merged_excerpts_df.group_by("page_id").agg(
        # calculate the number of characters covered by all merged spans for each page
        poetry_chars=(pl.col("ppa_span_end") - pl.col("ppa_span_start")).sum()
    )
    return excerpt_page_chars_df, merged_excerpts_df


@app.cell
def _(excerpt_page_chars_df, pl, ppa_meta_df, ppa_pages_df):
    # join merged span char length data with page data to determine poetry/nonpoetry chars
    text_poetrylen_df = (
        # We are starting with pages, so left join will include all pages, whether or not they have excerpts
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
def _(
    excerpts_df,
    merged_excerpts_df,
    mo,
    ppa_pages_df,
    text_poetrylen_df,
    total_pages,
):
    # output some numbers for text characters

    total_characters = ppa_pages_df["text_len"].sum()
    poetry_characters = text_poetrylen_df["poetry_chars"].sum()

    # longest / shortest excerpt
    longest_excerpt = excerpts_df["ppa_span_len"].max()
    shortest_excerpt = excerpts_df["ppa_span_len"].min()
    average_excerpt_len = excerpts_df["ppa_span_len"].mean()

    # same for merged excerpts
    longest_merged_excerpt = merged_excerpts_df["ppa_span_len"].max()
    shortest_merged_excerpt = merged_excerpts_df["ppa_span_len"].min()
    average_merged_excerpt_len = merged_excerpts_df["ppa_span_len"].mean()

    mo.md(f"""Across all {total_pages:,} PPA pages there are a total of {total_characters:,} characters of text.

    - {poetry_characters:,} characters detected as poetry ({poetry_characters/total_characters * 100:.1f}%)

    Excerpt length in characters (unmerged, {excerpts_df.height:,} total excerpts):
    - Longest: {longest_excerpt:,} 
    - Shortest: {shortest_excerpt:,} 
    - Average: {average_excerpt_len:.1f}

    Excerpt length in characters (merged all overlapping spans, {merged_excerpts_df.height:,} total excerpts):
    - Longest: {longest_merged_excerpt:,} 
    - Shortest: {shortest_merged_excerpt:,} 
    - Average: {average_merged_excerpt_len:.1f}


    """)
    return


@app.cell
def _(chart_has_poetry, mo, pl, text_poetrylen_df):
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
        .rename({"text_len": "count"})
    )

    text_chart_normalized = chart_has_poetry(
        text_poetrylen_year_df,
        "ppa_pub_year",
        field_title=None,
        y_axis_title="Characters",
        # mark_opts={"width": 18},
    )
    text_chart_count = chart_has_poetry(
        text_poetrylen_year_df,
        "ppa_pub_year",
        "Publication year",
        y_axis_title="Characters",
        normalize=False,
        # mark_opts={"width": 18},
    )

    mo.ui.altair_chart(
        (text_chart_normalized & text_chart_count).properties(
            title="PPA text detected as poetry, by work publication year"
        )
    )
    return


@app.cell
def _(chart_has_poetry, mo, pl, text_poetrylen_df):
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
        .rename({"text_len": "count"})
    )

    text_decade_chart_normalized = chart_has_poetry(
        text_poetrylen_decade_df,
        "ppa_pub_decade",
        field_title=None,
        y_axis_title="Characters",
        mark_opts={"width": 18},
    )
    text_decade_chart_count = chart_has_poetry(
        text_poetrylen_decade_df,
        "ppa_pub_decade",
        "Publication decade",
        y_axis_title="Characters",
        normalize=False,
        mark_opts={"width": 18},
    )

    mo.ui.altair_chart(
        (text_decade_chart_normalized & text_decade_chart_count).properties(
            title="PPA text detected as poetry, by work publication decade"
        )
    )
    return


if __name__ == "__main__":
    app.run()
