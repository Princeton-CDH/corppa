import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


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
    return c, excerpts_df


@app.cell
def _(c, excerpts_df):
    # make a subset for testing edit distance scores
    import polars_ds as pds

    # pages from Laure's evaluation
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


@app.cell
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


if __name__ == "__main__":
    app.run()
