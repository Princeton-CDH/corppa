import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Excerpt quality: text similarity measurements

    The way that passim can be greedy about matches can be useful - it makes it possible to find things we might not with other methods, e.g. where a line is omitted or there is intervening text; but it can also mean we end up with bad matches where only a few characters are consistent. We also know anecdotally that refmatcha results include some poor quality poem idenfications based on insufficient text data for a confident match.

    Can we use text similarity measurments between PPA and reference poem excerpt text as a way to filter out bad matches from our data?
    """)
    return


@app.cell
def _():
    import pathlib

    import marimo as mo
    import polars as pl
    from polars import col as c

    from corppa.config import get_config
    from corppa.poetry_detection.polars_utils import load_excerpts_df

    config_opts = get_config()
    data_dir = pathlib.Path(config_opts["compiled_dataset"]["data_dir"])

    # Create a dictionary of data files for lookup based on file base name without any extension
    # so that excerpts data can be .csv or compressed .csv.gz
    data_paths = {
        data_file.stem.split(".", 1)[0]: data_file for data_file in data_dir.iterdir()
    }

    # load excerpts with poem metadata
    excerpts_df = load_excerpts_df(
        data_paths["excerpts"], ref_poems_meta=data_paths["poem_meta"]
    )

    excerpts_df
    return c, excerpts_df, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `polars-ds` (Polars for Data Science) package is a polars extension which includes a number of utilities for text, including several similarity and edit distance measurements. Refer to [pds documentation on string measurement and manipulation](https://polars-ds-extension.readthedocs.io/en/latest/string.html).

    This means we can efficiently calculate these metrics for our excerpts, and look to see which of these might be helpful for us.
    """)
    return


@app.cell
def _(c, excerpts_df):
    # make a subset for testing edit distance scores
    import polars_ds as pds

    # pages from Laure's earlier manual evaluation
    lt_eval_pageids = [
        "CW0115230851.0306",
        "CW0116527364.0377",
        "CB0131056467.0049",
        "CW0111842062.0300",
        "CW0113212334.0009",
    ]

    test_excerpts_df = excerpts_df.filter(c.page_id.is_in(lt_eval_pageids))

    # use polars_ds to calculate edit distance between ppa text and reference text
    test_excerpts_df = test_excerpts_df.with_columns(
        dist_leven=pds.str_leven(
            c.ppa_span_text, c.ref_span_text
        ),  # levenshtein distance
        dist_d_leven=pds.str_d_leven(
            c.ppa_span_text, c.ref_span_text
        ),  # damereau-levenshtein distance
        indel_sim=pds.str_fuzz(
            c.ppa_span_text, c.ref_span_text
        ),  # rapidfuzz normalized Indel similarity
        jw_sim=pds.str_jw(c.ppa_span_text, c.ref_span_text),  # Jaro-Winkler similarity
        # what if we lowercase first?
        dist_leven_lc=pds.str_leven(
            c.ppa_span_text.str.to_lowercase(), c.ref_span_text.str.to_lowercase()
        ),
        dist_d_leven_lc=pds.str_d_leven(
            c.ppa_span_text.str.to_lowercase(), c.ref_span_text.str.to_lowercase()
        ),
        indel_sim_lc=pds.str_fuzz(
            c.ppa_span_text.str.to_lowercase(), c.ref_span_text.str.to_lowercase()
        ),
        jw_sim_lc=pds.str_jw(
            c.ppa_span_text.str.to_lowercase(), c.ref_span_text.str.to_lowercase()
        ),
    )

    test_excerpts_df.select(
        "ppa_span_text",
        "ref_span_text",
        "dist_leven",
        "dist_d_leven",
        "indel_sim",
        "jw_sim",
        "dist_leven_lc",
        "dist_d_leven_lc",
        "indel_sim_lc",
        "jw_sim_lc",
        "identification_methods",
        "poem_id",
        "poem_author",
        "poem_title",
    )
    return (pds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Some of the measurements are low due to case-sensitive changes, which we don't really care about for our use case.

    We can see that by calculating the same measurement on the lower-case versions of the two texts w're comparing.

    ---

    _note_: it took ~20 minutes to run all the similarity metrics for the complete current set of excerpts; better to do offline and best to choose which metrics are most efficient.
    """)
    return


@app.cell(disabled=True)
def _(c, excerpts_df, pds):
    # use polars_ds to calculate edit distance between ppa text and reference text
    excerpt_simstats_df = excerpts_df.with_columns(
        dist_leven=pds.str_leven(
            c.ppa_span_text, c.ref_span_text
        ),  # levenshtein distance
        dist_d_leven=pds.str_d_leven(
            c.ppa_span_text, c.ref_span_text
        ),  # damereau-levenshtein distance
        indel_sim=pds.str_fuzz(
            c.ppa_span_text, c.ref_span_text
        ),  # rapidfuzz normalized Indel similarity
        jw_sim=pds.str_jw(c.ppa_span_text, c.ref_span_text),  # Jaro-Winkler similarity
        # what if we lowercase first?
        dist_leven_lc=pds.str_leven(
            c.ppa_span_text.str.to_lowercase(), c.ref_span_text.str.to_lowercase()
        ),
        dist_d_leven_lc=pds.str_d_leven(
            c.ppa_span_text.str.to_lowercase(), c.ref_span_text.str.to_lowercase()
        ),
        indel_sim_lc=pds.str_fuzz(
            c.ppa_span_text.str.to_lowercase(), c.ref_span_text.str.to_lowercase()
        ),
        jw_sim_lc=pds.str_jw(
            c.ppa_span_text.str.to_lowercase(), c.ref_span_text.str.to_lowercase()
        ),
    )

    excerpt_simstats_df.select(
        "ppa_span_text",
        "ref_span_text",
        "dist_leven",
        "dist_d_leven",
        "indel_sim",
        "jw_sim",
        "dist_leven_lc",
        "dist_d_leven_lc",
        "indel_sim_lc",
        "jw_sim_lc",
        "identification_methods",
        "poem_id",
        "poem_author",
        "poem_title",
    )
    return (excerpt_simstats_df,)


@app.cell
def _(excerpt_simstats_df):
    excerpt_simstats_df.head()
    return


@app.cell
def _(c, excerpt_simstats_df):
    # it took a while to calculate that, so let's save the results for later

    excerpt_simstats_df.with_columns(
        identification_methods=c.identification_methods.list.join("; "),
        detection_methods=c.detection_methods.list.join("; "),
        alt_poem_ids=c.alt_poem_ids.list.join("; "),
    ).write_csv("excerpt_text_similarity_stats.csv")
    return


@app.cell
def _(c, pl):
    # the full set is too large to look at, so let's save a subset of the best and worst for review in a spreadsheet

    # load saved results from disk; exclude any without reference text (no measurements could be calculated)
    similarity_stats_df = pl.read_csv("excerpt_text_similarity_stats.csv").filter(
        c.ref_span_text.is_not_null()
    )

    # report how many have zero distance
    zerodist_total = similarity_stats_df.filter(c.dist_leven_lc.eq(0)).height

    similarity_stats_df.sort("indel_sim_lc", descending=True)
    return (similarity_stats_df,)


@app.cell
def _():
    import altair as alt
    # borrow from myself again

    # define a custom box plot method using layered plots,
    # so that we can quickly generate plots from statistics generated by polars
    # adapted from prior work https://princeton-cdh.github.io/simulating-risk/notebooks/hawkdovemulti-noadjust/
    # also used by notebooks/poem-excerpt-length.py in this repo

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
            x=alt.X(x_field, title=x_field_title).axis(format="r"),
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

    return (alt,)


@app.cell
def _(pl, similarity_stats_df):
    # aggregate by decade and calculate min/max/average for all poem length measurements

    binstats_df = pl.DataFrame()

    for field in ["indel_sim", "indel_sim_lc", "jw_sim", "jw_sim_lc"]:
        binstats_df = pl.concat(
            [
                binstats_df,
                similarity_stats_df[field]
                .hist(bin_count=10)
                .with_columns(
                    measure=pl.lit(field.rstrip("_lc")),
                    case=pl.lit("lower" if "_lc" in field else "mixed"),
                ),
            ],
            how="diagonal",
        )

    binstats_df
    return (binstats_df,)


@app.cell
def _(alt, binstats_df, mo):
    mo.ui.altair_chart(
        alt.Chart(binstats_df)
        .mark_bar(width=10)
        .encode(x=alt.X("breakpoint").scale(domain=[0.0, 1.0]), y="count")
        .facet(column="case", row="measure")
        .properties(
            title="Distribution of similarity between PPA and reference text with lower and mixed case"
        )
    )
    return


if __name__ == "__main__":
    app.run()
