import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PPA found poems — poem/excerpt length over time

    MM expects that excerpts get shorter over time as the books get smaller.  We also know that poems are also getting shorter over this time. What evidence of that can we find our found poem excerpt data?
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

    return alt, get_config, load_excerpts_df, mo, pathlib, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Poems cited in PPA

    We don't currently have dates in our poem metadata, but as a starting point we can look at the lengths of being poems quoted in PPA over time.
    """)
    return


@app.cell
def _(get_config, load_excerpts_df, pathlib, pl):
    config_opts = get_config()
    data_dir = pathlib.Path(config_opts["compiled_dataset"]["data_dir"])

    # Create a dictionary of data files for lookup based on file base name without any extension
    # so that excerpts data can be .csv or compressed .csv.gz
    data_paths = {
        data_file.stem.split(".", 1)[0]: data_file for data_file in data_dir.iterdir()
    }

    # use the existing method to do the maximal load and join excerpts with ppa and poem metadata, and then subset/combine
    excerpts_df = (
        load_excerpts_df(
            data_paths["excerpts"],
            ppa_works_meta=data_paths["ppa_work_metadata"],
            ref_poems_meta=data_paths["poem_meta"],
        )
        .with_columns(
            # round years to decade
            ppa_pub_decade=pl.col("ppa_pub_year").floordiv(10).mul(10),
        )
        .cast(
            # convert all the length measures to numeric so we can calculate stats
            {
                "poem_num_lines": pl.Int32,
                "poem_num_words": pl.Int32,
                "poem_char_len": pl.Int32,
            }
        )
    )

    excerpts_df
    return (excerpts_df,)


@app.cell
def _(excerpts_df):
    # filter down to unique pairs of works + poems with decade and poem length field
    works_poems_df = excerpts_df.select(
        "ppa_work_id",
        "ppa_pub_decade",
        "poem_id",
        "poem_num_lines",
        "poem_num_words",
        "poem_char_len",
    ).unique()
    works_poems_df
    return (works_poems_df,)


@app.cell
def _(pl, works_poems_df):
    # aggregate by decade and calculate min/max/average for all poem length measurements
    work_poem_decade_stats_df = works_poems_df.group_by("ppa_pub_decade").agg(
        count=pl.len(),
        # number of lines
        min_lines=pl.col("poem_num_lines").min(),
        max_lines=pl.col("poem_num_lines").max(),
        mean_lines=pl.col("poem_num_lines").mean(),
        lines_Q1=pl.col("poem_num_lines").quantile(0.25),
        lines_Q2=pl.col("poem_num_lines").quantile(0.5),
        lines_Q3=pl.col("poem_num_lines").quantile(0.75),
        # number of words
        min_words=pl.col("poem_num_words").min(),
        max_words=pl.col("poem_num_words").max(),
        mean_words=pl.col("poem_num_words").mean(),
        words_Q1=pl.col("poem_num_words").quantile(0.25),
        words_Q2=pl.col("poem_num_words").quantile(0.5),
        words_Q3=pl.col("poem_num_words").quantile(0.75),
        # number of characters poem_char_len
        min_chars=pl.col("poem_char_len").min(),
        max_chars=pl.col("poem_char_len").max(),
        mean_chars=pl.col("poem_char_len").mean(),
        chars_Q1=pl.col("poem_char_len").quantile(0.25),
        chars_Q2=pl.col("poem_char_len").quantile(0.5),
        chars_Q3=pl.col("poem_char_len").quantile(0.75),
    )
    work_poem_decade_stats_df
    return (work_poem_decade_stats_df,)


@app.cell
def _(alt, mo, work_poem_decade_stats_df):
    mo.ui.altair_chart(
        alt.Chart(work_poem_decade_stats_df)
        .mark_area(
            opacity=0.4,
            color="#f05b69",
        )
        .encode(
            x=alt.X("ppa_pub_decade", title="PPA Publication decade").axis(format="r"),
            y=alt.Y("lines_Q3", title="Number of lines (mean, Q1, Q3)"),
            y2="lines_Q1",
            tooltip=["lines_Q1", "mean_lines", "lines_Q2", "lines_Q3"],
        )
        + alt.Chart(work_poem_decade_stats_df)
        .mark_line()
        .encode(x="ppa_pub_decade", y="lines_Q2")
        .properties(
            title="Average and Quartile poem length (by lines) for poems cited in PPA works by PPA publication decade"
        )
    )
    return


@app.cell
def _(alt, mo, works_poems_df):
    # plot the same distribution as a series of box plots
    # requires vegafusion because we're letting altair calculate the quartiles
    alt.data_transformers.enable("vegafusion")

    mo.ui.altair_chart(
        alt.Chart(works_poems_df)
        .mark_boxplot()
        .encode(
            x=alt.X("ppa_pub_decade", title="PPA Publication decade")
            .axis(format="r")
            .scale(zero=False),
            y=alt.Y("poem_num_lines", title="Poem length by number of lines"),
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can plot poem length by number of words or number of characters - but the general trend looks the same across those measurements.
    """)
    return


@app.cell
def _(alt, mo, work_poem_decade_stats_df):
    mo.ui.altair_chart(
        alt.Chart(work_poem_decade_stats_df)
        .mark_area(
            opacity=0.4,
            color="#f05b69",
        )
        .encode(
            x=alt.X("ppa_pub_decade", title="PPA Publication decade").axis(format="r"),
            y=alt.Y("words_Q3", title="Number of words (mean, Q1, Q3)"),
            y2="words_Q1",
            tooltip=["words_Q1", "words_Q2", "words_Q3", "mean_words"],
        )
        + alt.Chart(work_poem_decade_stats_df)
        .mark_line()
        .encode(x="ppa_pub_decade", y="words_Q2")
        .properties(
            title="Average and Quartile poem length (by number of words) for poems cited in PPA works by PPA publication decade"
        )
    )
    return


@app.cell
def _(alt, mo, work_poem_decade_stats_df):
    mo.ui.altair_chart(
        alt.Chart(work_poem_decade_stats_df)
        .mark_area(
            opacity=0.4,
            color="#f05b69",
        )
        .encode(
            x=alt.X("ppa_pub_decade", title="PPA Publication decade").axis(format="r"),
            y=alt.Y("chars_Q3", title="Number of characters (mean, Q1, Q3)"),
            y2="chars_Q1",
            tooltip=["chars_Q1", "chars_Q2", "chars_Q3", "mean_chars"],
        )
        + alt.Chart(work_poem_decade_stats_df)
        .mark_line()
        .encode(x="ppa_pub_decade", y="chars_Q2")
        .properties(
            title="Average and Quartile poem length (by number of characters) for poems cited in PPA works by PPA publication decade"
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Poem excerpt length

    How much of a poem is cited in a PPA work, and how does that change over time?

    To simplify our measurement and avoid counting duplicate excerpts or excerpts split by page range, we aggregate excerpts
    and collapse the reference spans by PPA work, to determine the total length of each poem cited in each work.
    """)
    return


@app.cell
def _(excerpts_df, pl):
    # in the percent ppa poetry notebook, we collapsed overlapping spans for the ppa text, for each page of ppa
    # here, we want to do collapse reference spans for each ppa work

    # collapse excerpts with any overlap to a single span so we can calculate the total number of characters
    # covered by any of the merged spans

    ref_merged_excerpts_df = (
        # sort by work, poem, and reference span start
        excerpts_df.sort("ppa_work_id", "poem_id", "ref_span_start")
        .select(
            "ppa_work_id",
            "ppa_pub_decade",
            "poem_id",
            "ref_span_start",
            "ref_span_end",
            "poem_author",
            "poem_title",
            "poem_char_len",
        )
        .with_columns(
            # Use shift and cumulative max to determine if current span
            # has any overlap with previous spans or is the beginning of a new group.
            # shift(1) gets previous row; fill null for first row (which has no previous row),
            # and calculate current max span end for this page.
            # NOTE: we use >= because span end is exclusive (i.e., is not included in the range)
            new_group=(
                pl.col("ref_span_start")
                >= pl.col("ref_span_end").shift(1).fill_null(-1).cum_max()
            )
            .cast(pl.Int32)  # cast to int gives 1 or 0 to indicate new group
            .over(
                "ppa_work_id", "poem_id"
            )  # limit to spans to a single poem quoted in a single work
        )
        .with_columns(
            # because new_group is 1 or 0, cumulative sum gives each group on a page a unique group id
            pl.col("new_group")
            .cum_sum()
            .alias("group_id")
            .over("ppa_work_id", "poem_id")
        )
        .group_by("ppa_work_id", "poem_id", "group_id")
        .agg(
            # group by page id and group id and get the smallest start and largest end
            # to get the outer bounds of the overlapping spans
            pl.col("ref_span_start").min(),
            pl.col("ref_span_end").max(),
            pl.col("ppa_pub_decade").first(),
            pl.col("poem_title").first(),
            pl.col("poem_author").first(),
            pl.col("poem_char_len").first(),
        )
        # calculate length of the consolidated reference
        .with_columns(ref_span_len=pl.col.ref_span_end - pl.col.ref_span_start)
        # calculate percentage of the poem that is quoted
        .with_columns(ref_percent=pl.col.ref_span_len.truediv(pl.col.poem_char_len))
        .drop("group_id")
    )

    # based on the merged reference spans, calculate how much of each poem is quoted in each work

    excerpt_poem_chars_df = ref_merged_excerpts_df.group_by(
        "ppa_work_id", "poem_id"
    ).agg(
        # calculate the number of characters covered by all merged spans for each poem
        ref_char_len=(pl.col("ref_span_end") - pl.col("ref_span_start")).sum(),
        ppa_pub_decade=pl.col.ppa_pub_decade.first(),
    )

    excerpt_poem_chars_df
    return excerpt_poem_chars_df, ref_merged_excerpts_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Which poems are quoted from the most?
    """)
    return


@app.cell
def _(ref_merged_excerpts_df):
    ref_merged_excerpts_df.sort(
        "ref_span_len", descending=True, nulls_last=True
    ).select(
        "ppa_work_id",
        "poem_id",
        "poem_author",
        "poem_title",
        "poem_char_len",
        "ref_span_len",
        "ref_percent",
    )
    return


@app.cell
def _(alt, excerpt_poem_chars_df, mo, pl):
    # aggregrate reference spans to get statistics over PPA works by decade

    ref_excerpts_stats_df = excerpt_poem_chars_df.group_by("ppa_pub_decade").agg(
        count=pl.len(),
        # number of characters quoted from a poem, based on combined reference span length
        min_chars=pl.col("ref_char_len").min(),
        max_chars=pl.col("ref_char_len").max(),
        mean_chars=pl.col("ref_char_len").mean(),
        Q1_chars=pl.col("ref_char_len").quantile(0.25),
        median_chars=pl.col("ref_char_len").quantile(0.5),
        Q3_chars=pl.col("ref_char_len").quantile(0.75),
    )

    # unpivot mean/median to graph together with color legend
    mean_median_ref_stats_df = ref_excerpts_stats_df.unpivot(
        on=["mean_chars", "median_chars"], index="ppa_pub_decade"
    )

    mean_median_reflength_chart = (
        alt.Chart(mean_median_ref_stats_df)
        .mark_line()
        .encode(x="ppa_pub_decade", y="value", color="variable")
        .properties(
            title="Average and quantiles for poem excerpt length included per PPA work, by PPA publication decade"
        )
    )

    mo.ui.altair_chart(
        alt.Chart(ref_excerpts_stats_df)
        .mark_area(
            opacity=0.4,
            color="#f05b69",
        )
        .encode(
            x=alt.X("ppa_pub_decade", title="PPA Publication decade").axis(format="r"),
            y=alt.Y("Q3_chars", title="Poem characters quoted (mean, Q1, Q3)"),
            y2="Q1_chars",
            tooltip=["Q1_chars", "mean_chars", "median_chars", "Q3_chars"],
        )
        + mean_median_reflength_chart
    )
    return (ref_excerpts_stats_df,)


@app.cell
def _(alt, mo, ref_excerpts_stats_df):
    mo.ui.altair_chart(
        (
            alt.Chart(ref_excerpts_stats_df)
            .mark_area(
                opacity=0.4,
                color="#6252a0",
            )
            .encode(
                x=alt.X("ppa_pub_decade", title="PPA Publication decade").axis(
                    format="r"
                ),
                y=alt.Y("min_chars", title="Poem characters quoted (min/max length)"),
                y2="max_chars",
            )
        )
    )
    return


@app.cell
def _():
    # still to do - does the percent of the poem that is quoted change over time?
    return


if __name__ == "__main__":
    app.run()
