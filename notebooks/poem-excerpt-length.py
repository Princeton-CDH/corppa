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
        median_lines=pl.col("poem_num_lines").quantile(0.5),  #  Q2 = median
        lines_Q3=pl.col("poem_num_lines").quantile(0.75),
        # number of words
        min_words=pl.col("poem_num_words").min(),
        max_words=pl.col("poem_num_words").max(),
        mean_words=pl.col("poem_num_words").mean(),
        words_Q1=pl.col("poem_num_words").quantile(0.25),
        median_words=pl.col("poem_num_words").quantile(0.5),
        words_Q3=pl.col("poem_num_words").quantile(0.75),
        # number of characters poem_char_len
        min_chars=pl.col("poem_char_len").min(),
        max_chars=pl.col("poem_char_len").max(),
        mean_chars=pl.col("poem_char_len").mean(),
        chars_Q1=pl.col("poem_char_len").quantile(0.25),
        median_chars=pl.col("poem_char_len").quantile(0.5),
        chars_Q3=pl.col("poem_char_len").quantile(0.75),
    )
    work_poem_decade_stats_df
    return (work_poem_decade_stats_df,)


@app.cell
def _(alt):
    def plot_quartiles(df, x_field, x_field_title, stat_field, stat_noun):
        # generate a layered chart of area between Q1/Q3 and lines for quartiles, means, median

        # unpivot mean/median to graph together with color legend
        stats_fields = [
            f"{stat_field}_Q1",
            f"mean_{stat_field}",
            f"median_{stat_field}",
            f"{stat_field}_Q3",
        ]
        stats_df = df.unpivot(on=stats_fields, index=x_field)

        # return a layered chart with area and lines
        return alt.layer(
            alt.Chart(df)
            .mark_area(
                opacity=0.4,
                color="#f05b69",
            )
            .encode(
                x=alt.X(x_field, title=x_field_title)
                .axis(format="r")
                .scale(zero=False),
                y=alt.Y(
                    f"{stat_field}_Q3",
                    title=f"{stat_noun} (Q1, Q2, Q3, mean, max)",
                ),
                y2=f"{stat_field}_Q1",
                tooltip=stats_fields,
            ),
            alt.Chart(stats_df)
            .mark_line()
            .encode(x=x_field, y="value", color="variable"),
        )

    return (plot_quartiles,)


@app.cell
def _(mo, plot_quartiles, work_poem_decade_stats_df):
    mo.ui.altair_chart(
        plot_quartiles(
            work_poem_decade_stats_df,
            "ppa_pub_decade",
            "PPA Publication decade",
            "lines",
            "Number of lines",
        ).properties(
            title="Mean and quartile poem length in lines for poems cited in PPA works by decade"
        )
    )
    return


@app.cell
def _(custom_boxplot, mo, work_poem_decade_stats_df):
    mo.ui.altair_chart(
        custom_boxplot(
            work_poem_decade_stats_df,
            "ppa_pub_decade",
            "PPA Publication decade",
            "lines",
            "Number of lines",
        )
        .properties(
            title="Distribution of poem length for all poems quoted in PPA by decade"
        )
        .interactive()
    )
    return


@app.cell
def _(alt):
    # define a custom box plot method using layered plots,
    # so that we can quickly generate plots from statistics generated by polars

    def custom_boxplot(df, x_field, x_field_title, stat_field, stat_noun):
        stats_fields = [
            f"min_{stat_field}",
            f"{stat_field}_Q1",
            f"mean_{stat_field}",
            f"median_{stat_field}",
            f"{stat_field}_Q3",
            f"max_{stat_field}",
        ]

        # create base chart to use across layers
        base_chart = alt.Chart(df)

        # area chart for Q1 to Q3
        area_chart = base_chart.mark_rect(width=15).encode(
            y=alt.Y(f"{stat_field}_Q1").axis(
                offset=12
            ),  # add offset so axis does not crowd rectangle
            y2=f"{stat_field}_Q3",
            x=alt.X(x_field, title=x_field_title),
            tooltip=stats_fields,
        )
        stroke_color = "orange"
        # line chart for min-max spread
        # specifying a stroke for point on the line only adds the min point
        minmax_line_chart = base_chart.mark_line(
            point=alt.OverlayMarkDef(
                filled=False, shape="stroke", color=stroke_color, strokeWidth=2
            ),
            color=stroke_color,
        ).encode(alt.Y(f"min_{stat_field}"), alt.Y2(f"max_{stat_field}"), x=x_field)
        # add a stroke for the max
        max_marks = base_chart.mark_point(
            shape="stroke", size=55, color=stroke_color
        ).encode(
            y=alt.Y(f"max_{stat_field}"),
            x=x_field,
        )
        # add a stroke for the min
        median_marks = base_chart.mark_point(
            shape="stroke", size=100, strokeWidth=1, color=stroke_color
        ).encode(y=f"median_{stat_field}", x=x_field)

        # mean line ?
        mean_line_chart = base_chart.mark_line(
            interpolate="monotone", color="yellow", opacity=0.5
        ).encode(
            x=alt.X(x_field),
            y=alt.Y(f"mean_{stat_field}", title=f"{stat_noun} (mean)").scale(
                zero=False
            ),
        )

        return alt.layer(
            mean_line_chart, minmax_line_chart, area_chart, median_marks, max_marks
        ).resolve_axis("shared")

    return (custom_boxplot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can plot poem length by number of words or number of characters - but the general trend looks the same across those measurements.
    """)
    return


@app.cell
def _(mo, plot_quartiles, work_poem_decade_stats_df):
    mo.ui.altair_chart(
        plot_quartiles(
            work_poem_decade_stats_df,
            "ppa_pub_decade",
            "PPA Publication decade",
            "words",
            "Number of words",
        ).properties(
            title="Mean and quartile poem length by number of words for poems cited in PPA works by decade"
        )
    )
    return


@app.cell
def _(mo, plot_quartiles, work_poem_decade_stats_df):
    mo.ui.altair_chart(
        plot_quartiles(
            work_poem_decade_stats_df,
            "ppa_pub_decade",
            "PPA Publication decade",
            "chars",
            "Number of characters",
        ).properties(
            title="Mean and quartile poem length by number of characters for poems cited in PPA works by decade"
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Poem length by first appearance in PPA
    """)
    return


@app.cell
def _(excerpts_df, pl):
    # instead of filtering to unique pairs of works + poems with decade and poem length field,
    # aggregate by poem id and get the earliest date it is quoted in the PPA
    poems_firstquoted_df = excerpts_df.group_by("poem_id").agg(
        pl.first("ppa_pub_decade"),
        pl.first("poem_num_lines"),
        pl.first("poem_num_words"),
        pl.first("poem_char_len"),
        pl.first("ppa_work_id"),
        pl.first("poem_title"),
        pl.first("poem_author"),
    )
    poems_firstquoted_df
    return (poems_firstquoted_df,)


@app.cell
def _(pl, poems_firstquoted_df):
    # now generate stats
    # aggregate by decade and calculate min/max/average for all poem length measurements
    poems_firstquoted_stats_df = poems_firstquoted_df.group_by("ppa_pub_decade").agg(
        count=pl.len(),  # number of poems
        # number of lines
        min_lines=pl.col("poem_num_lines").min(),
        max_lines=pl.col("poem_num_lines").max(),
        mean_lines=pl.col("poem_num_lines").mean(),
        lines_Q1=pl.col("poem_num_lines").quantile(0.25),
        median_lines=pl.col("poem_num_lines").quantile(0.5),
        lines_Q3=pl.col("poem_num_lines").quantile(0.75),
        # number of words
        min_words=pl.col("poem_num_words").min(),
        max_words=pl.col("poem_num_words").max(),
        mean_words=pl.col("poem_num_words").mean(),
        words_Q1=pl.col("poem_num_words").quantile(0.25),
        median_words=pl.col("poem_num_words").quantile(0.5),
        words_Q3=pl.col("poem_num_words").quantile(0.75),
        # number of characters poem_char_len
        min_chars=pl.col("poem_char_len").min(),
        max_chars=pl.col("poem_char_len").max(),
        mean_chars=pl.col("poem_char_len").mean(),
        chars_Q1=pl.col("poem_char_len").quantile(0.25),
        median_chars=pl.col("poem_char_len").quantile(0.5),
        chars_Q3=pl.col("poem_char_len").quantile(0.75),
    )
    poems_firstquoted_stats_df
    return (poems_firstquoted_stats_df,)


@app.cell
def _(pl, poems_firstquoted_df):
    # what is that early outlier skewing the graphs?
    poems_firstquoted_df.filter(pl.col.ppa_pub_decade.lt(1600)).sort(
        "poem_num_lines", descending=True
    )
    return


@app.cell
def _(pl, poems_firstquoted_stats_df):
    poems_firstquoted_stats_df.filter(pl.col.median_lines.gt(100))
    return


@app.cell
def _(mo, plot_quartiles, poems_firstquoted_stats_df):
    mo.ui.altair_chart(
        plot_quartiles(
            poems_firstquoted_stats_df,
            "ppa_pub_decade",
            "PPA Publication decade",
            "lines",
            "Number of lines",
        ).properties(
            title="Mean and quartile poem length by number of lines for poems first appearance in PPA"
        )
    )
    return


@app.cell
def _(custom_boxplot, mo, poems_firstquoted_stats_df):
    mo.ui.altair_chart(
        custom_boxplot(
            poems_firstquoted_stats_df,
            "ppa_pub_decade",
            "PPA Publication decade",
            "lines",
            "Number of lines",
        )
        .properties(
            title="Distribution of poem length based on poem first appearance in PPA by decade"
        )
        .interactive()
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
    return (ref_merged_excerpts_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Which poems are quoted from the most?   (Sorting by sum of reference span lengths)
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
def _(mo, pl, ref_merged_excerpts_df):
    fully_quoted_poems = ref_merged_excerpts_df.filter(pl.col.ref_percent.ge(1)).height

    mo.md(
        f"""{fully_quoted_poems:,} poems are quoted in full 

    (based on total reference span length and percentage of poem length, which may not match exactly)"""
    )
    return


@app.cell
def _(ref_merged_excerpts_df):
    # which ones are quoted most?
    # we have numbers of 100% here - guessing this is due to lack of alignment / different ways of counting characters

    ref_merged_excerpts_df.sort("ref_percent", descending=True, nulls_last=True).select(
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
def _(alt, mo, pl, plot_quartiles, ref_merged_excerpts_df):
    # aggregrate reference spans to get statistics over PPA works by decade

    ref_excerpts_stats_df = ref_merged_excerpts_df.group_by("ppa_pub_decade").agg(
        count=pl.len(),
        # number of characters quoted from a poem, based on combined reference span length
        min_chars=pl.col("ref_span_len").min(),
        max_chars=pl.col("ref_span_len").max(),
        mean_chars=pl.col("ref_span_len").mean(),
        chars_Q1=pl.col("ref_span_len").quantile(0.25),
        median_chars=pl.col("ref_span_len").quantile(0.5),
        chars_Q3=pl.col("ref_span_len").quantile(0.75),
        # percent of poem by character length
        mean_percent=pl.col("ref_percent").mean(),
        percent_Q1=pl.col("ref_percent").quantile(0.25),
        median_percent=pl.col("ref_percent").quantile(0.5),
        percent_Q3=pl.col("ref_percent").quantile(0.75),
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
        plot_quartiles(
            ref_excerpts_stats_df,
            "ppa_pub_decade",
            "PPA Publication decade",
            "chars",
            "Number of characters",
        ).properties(
            title="Mean and quartile poem quotation length by number of characters for poems found in PPA"
        )
    )
    return (ref_excerpts_stats_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can graph the min/max, but the maximum length is quite large and changes the scale substantially.
    """)
    return


@app.cell
def _(alt, mo, ref_excerpts_stats_df):
    mo.ui.altair_chart(
        alt.Chart(ref_excerpts_stats_df)
        .mark_area(
            opacity=0.4,
            color="#6252a0",
        )
        .encode(
            x=alt.X("ppa_pub_decade", title="PPA Publication decade").axis(format="r"),
            y=alt.Y("min_chars", title="Poem characters quoted (min/max length)"),
            y2="max_chars",
        )
    )
    return


@app.cell
def _(mo, plot_quartiles, ref_excerpts_stats_df):
    # what percent of poems are quoted over time?

    mo.ui.altair_chart(
        plot_quartiles(
            ref_excerpts_stats_df,
            "ppa_pub_decade",
            "PPA Publication decade",
            "percent",
            "Percent of poem",
        ).properties(title="Percent of poem quoted in a single work")
    )
    return


if __name__ == "__main__":
    app.run()
