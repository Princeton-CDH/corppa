import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from polars import col as c

    return c, mo, pl


@app.cell
def _():
    import pathlib

    from corppa.config import get_config
    from corppa.poetry_detection.polars_utils import load_excerpts_df

    config_opts = get_config()
    data_dir = pathlib.Path(config_opts["compiled_dataset"]["data_dir"])

    # Create a dictionary of data files for lookup based on file base name without any extension
    # so that excerpts data can be .csv or compressed .csv.gz
    data_paths = {
        data_file.stem.split(".", 1)[0]: data_file for data_file in data_dir.iterdir()
    }

    # load excerpts with poem metadata joined
    excerpts_df = load_excerpts_df(
        data_paths["excerpts"], ref_poems_meta=data_paths["poem_meta"]
    )
    excerpts_df
    return data_paths, excerpts_df


@app.cell
def _(c, excerpts_df, pl):
    # Do a self-join to identify nested excerpts.
    # How many excerpts are completely included within another excerpt?

    # This will result in pairs of excerpts where excerpt_id is completely contained within excerpt_id_right
    # (todo: better suffix?)

    # make a page-excerpt combined id to simplify filtering and checking for unique

    excerpts_id_df = excerpts_df.with_columns(
        uniq_excerpt_id=pl.concat_str(c.page_id, c.excerpt_id, separator="|")
    )

    nested_excerpts_df = excerpts_id_df.join_where(
        excerpts_id_df,
        # 1. Excerpts are on the same page
        pl.col("page_id") == pl.col("page_id_right"),
        # 2. Excerpt is nested:
        #    left span starts at or after right span starts
        pl.col("ppa_span_start") >= pl.col("ppa_span_start_right"),
        #  and left span ends at or before right span ends
        pl.col("ppa_span_end") <= pl.col("ppa_span_end_right"),
        # 3. Exclude self-matches
        pl.col("uniq_excerpt_id") != pl.col("uniq_excerpt_id_right"),
    )

    nested_excerpts_df
    return (nested_excerpts_df,)


@app.cell
def _(nested_excerpts_df, pl):
    nested_excerpts_df.filter(
        pl.col.uniq_excerpt_id.is_in(pl.col.uniq_excerpt_id_right.implode())
    ).select(
        "page_id",
        "ppa_span_text",
        "ppa_span_text_right",
        "poem_title",
        "poem_title_right",
        "poem_author",
        "poem_author_right",
        "ref_span_text",
        "ref_span_text_right",
    )
    return


@app.cell
def _(mo, nested_excerpts_df, pl):
    # get some summary numebers
    n_nested_pairs = nested_excerpts_df.height
    n_uniq_nested = nested_excerpts_df["uniq_excerpt_id"].n_unique()
    n_uniq_parent = nested_excerpts_df["uniq_excerpt_id_right"].n_unique()

    # how many parent/containing excerpts are also nested? (i.e., a is in b and b is in c)
    n_uniq_double_nested = nested_excerpts_df.filter(
        pl.col.uniq_excerpt_id.is_in(pl.col.uniq_excerpt_id_right.implode())
    )["uniq_excerpt_id"].n_unique()

    mo.md(
        f"""- {n_nested_pairs:,} pairs of nested excerpts
        - {n_uniq_nested:,} unique nested excerpt ids
        - {n_uniq_parent:,} unique excerpts with nested excerpts
        - {n_uniq_double_nested:,} unique excerpts that contain at least one excerpt and are contained within another
         """
    )
    return


@app.cell
def _(nested_excerpts_df):
    nested_excerpts_df
    return


@app.cell
def _(nested_excerpts_df):
    nested_excerpts_df.select(
        "ppa_span_text",
        "ref_span_text",
        "poem_author",
        "poem_title",
        "ppa_span_text_right",
        "ref_span_text_right",
        "poem_author_right",
        "poem_title_right",
    )
    return


@app.cell
def _(nested_excerpts_df):
    nested_excerpts_df.select("uniq_excerpt_id", "uniq_excerpt_id_right").rename(
        {"uniq_excerpt_id": "source", "uniq_excerpt_id_right": "target"}
    )
    return


@app.cell
def _(c, nested_excerpts_df):
    # find some examples of double-nested excerpts

    double_nested_df = nested_excerpts_df.filter(
        c.uniq_excerpt_id.is_in(c.uniq_excerpt_id_right.implode())
    ).unique("uniq_excerpt_id")
    double_nested_df
    return (double_nested_df,)


@app.cell
def _(double_nested_df, nested_excerpts_df):
    # join with nested excerpts where double-nested excerpt is the right-side excerpt (larger excerpt)
    # to get both sides of the double-nested passage

    double_nested_both_df = double_nested_df.join(
        nested_excerpts_df,
        left_on="uniq_excerpt_id",
        right_on="uniq_excerpt_id_right",
        suffix="_left",
    )
    double_nested_both_df
    return (double_nested_both_df,)


@app.cell
def _(double_nested_df):
    double_nested_df.select(
        "ppa_span_text_right",  # longest version
        "ref_span_text",
        "poem_author",
        "poem_title",
        "ref_span_text_right",
        "poem_author_right",
        "poem_title_right",
    )
    return


@app.cell
def _(double_nested_both_df):
    # select fields to display the double-nesting

    show_doublenested_df = double_nested_both_df.select(
        "ref_span_text_left",
        "poem_author_left",
        "poem_title_left",
        "ppa_span_text_right",  # longest version
        "ref_span_text",
        "poem_author",
        "poem_title",
        "ref_span_text_right",
        "poem_author_right",
        "poem_title_right",
    )

    # save to file for review
    show_doublenested_df.write_csv("double_nested_excerpts.csv")
    # full set is too many to review; save a subset
    show_doublenested_df.limit(n=1000).write_csv("double_nested_excerpts_subset.csv")

    show_doublenested_df
    return


@app.cell
def _(nested_excerpts_df):
    import networkx as nx

    # select id pairs and rename source -> target to construct a graph
    network_edges_df = nested_excerpts_df.select(
        "excerpt_id", "excerpt_id_right"
    ).rename({"excerpt_id": "source", "excerpt_id_right": "target"})

    # Create graph from edge pairs

    # save edge list
    # network_edges_df.write_csv("nested_excerpts.csv")

    G = nx.DiGraph()
    G.add_edges_from(network_edges_df.rows())
    print(f"{len(G.nodes):,} nodes, {len(G.edges):,} edges")

    # nt1 = Network("500px", "500px")
    # # populates the nodes and edges data structures
    # nt1.from_nx(G)
    # # nt.write_html("test.html", local=True)
    # # mo.Html(nt.html)
    # # print(nt.html)
    # mo.iframe(nt1.generate_html(), height=550)
    return (nx,)


@app.cell
def _(nested_excerpts_df, pl):
    # do we want a network of excerpts or a network of _poems_ ?
    # convert nested excerpt dataframe to a poem reference/reuse network; preserve the smaller ppa text that is reused
    nested_poem_edges_df = (
        nested_excerpts_df.filter(
            pl.col.poem_id.is_not_null(), pl.col.poem_id_right.is_not_null()
        )
        .select(
            "poem_id",
            "poem_id_right",
            "ppa_span_text",
            "poem_title",
            "poem_author",
            "poem_title_right",
            "poem_author_right",
        )
        .rename(
            {
                "poem_id": "source",
                "poem_id_right": "target",
                "poem_title": "source_poem",
                "poem_title_right": "target_poem",
                "poem_author": "source_author",
                "poem_author_right": "target_author",
            }
        )
        .group_by(pl.col.source, pl.col.target, pl.col.ppa_span_text)
        .agg(
            pl.first("source_poem"),
            pl.first("source_author"),
            pl.first("target_poem"),
            pl.first("target_author"),
            pl.len().alias("count"),
        )
        .sort(pl.col.count, descending=True)
    )
    nested_poem_edges_df
    return (nested_poem_edges_df,)


@app.cell
def _(nested_poem_edges_df, pl):
    # Which poems are nested most frequently? Which have nested excerpts most frequently?

    nested_poem_edges_df.group_by("source").agg(
        pl.len().alias(
            "count"
        ),  # number of times this poem occurs as a source (i.e., nested excerpt)
        pl.first("source_poem"),
        pl.first("source_author"),
        pl.col("target").mode(),
        # use mode to get the most frequent target
        pl.col("target_poem").mode(),
        pl.col("target_author").mode(),
        pl.col("count")
        .sum()
        .alias(
            "total"
        ),  # total number of times this set of poem excerpts occurs across all ppa
    ).sort("count", descending=True)
    return


@app.cell
def _(data_paths, nested_excerpts_df, pl):
    # load poem metadata, limit to poems included in the poem network
    poem_meta_df = pl.read_csv(data_paths["poem_meta"]).filter(
        pl.col.poem_id.is_in(nested_excerpts_df["poem_id"].implode())
        | pl.col.poem_id.is_in(nested_excerpts_df["poem_id_right"].implode())
    )
    # TODO: dates would help here!
    poem_meta_df = poem_meta_df.with_columns(
        label=pl.concat_str([pl.col.author, pl.lit(": "), pl.col.title])
    )
    poem_meta_df
    return (poem_meta_df,)


@app.cell
def _(nested_poem_edges_df, nx, poem_meta_df):
    # create a directed graph to track nested poetry excerpts
    poem_graph = nx.DiGraph()

    # populate graph nodes with metadata
    poem_graph.add_nodes_from(
        (row["poem_id"], {k: v for k, v in row.items() if k != "poem_id"})
        # include poem id, author, title (add date when we have it); fill nulls with empty string
        for row in poem_meta_df.select("poem_id", "label", "author", "title")
        .fill_null("")
        .iter_rows(named=True)
    )

    # NOTE: edge attributes can be passed as a dict; include the nested text
    # TODO: count here should probably be used to weight the edges
    poem_graph.add_edges_from(
        (
            row["source"],
            row["target"],
            {k: v for k, v in row.items() if k not in ["source", "target"]},
        )
        for row in nested_poem_edges_df.select(
            "source", "target", "ppa_span_text", "count"
        )
        .fill_null("")
        .iter_rows(named=True)
    )
    print(f"{len(poem_graph.nodes):,} nodes; {len(poem_graph.edges):,} edges")
    return (poem_graph,)


@app.cell
def _(poem_graph):
    # subgraphs ?

    poem_graph_und = poem_graph.to_undirected()
    # nx.number_connected_components(poem_graph_und)
    return


@app.cell
def _():
    # networkx is slow; better to do this in gephi / gephi lite

    # pos = nx.spring_layout(poem_graph, method="energy", seed=0)
    # pos
    return


@app.cell
def _(nx, poem_graph):
    nx.write_gexf(poem_graph, "nested_poems.gexf")
    return


if __name__ == "__main__":
    app.run()
