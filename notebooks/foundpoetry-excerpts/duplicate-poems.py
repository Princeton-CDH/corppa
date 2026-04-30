import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Found poem excerpts: investigate duplication

    This notebook investigates duplicate poems based on excerpt data. It looks at poems that are connected based on alternate ids and exactly matching spans, and also looks at poems with a high-confidence overlapping spans.
    """)
    return


@app.cell
def _():
    import pathlib

    import marimo as mo
    import polars as pl
    from polars import col as c  # for short hand column reference

    from corppa.config import get_config
    from corppa.poetry_detection.polars_utils import load_excerpts_df

    config_opts = get_config()
    data_dir = pathlib.Path(config_opts["compiled_dataset"]["data_dir"])

    # Create a dictionary of data files for lookup based on file base name without any extension
    # so that excerpts data can be .csv or compressed .csv.gz
    data_paths = {
        data_file.stem.split(".", 1)[0]: data_file for data_file in data_dir.iterdir()
    }

    # load excerpts - with no other metadata loaded for now (add poems once we are looking at pairs of ids)
    excerpts_df = load_excerpts_df(data_paths["excerpts"])
    initial_excerpt_total = excerpts_df.height
    # preserve a copy of all excerpts
    all_excerpts_df = excerpts_df

    # this version of excerpt data has records merged by exactly matching span; when merged, they have alt_ids
    # filter to the subset of records with alt ids present
    excerpts_df = excerpts_df.filter(pl.col.alt_poem_ids.is_not_null())

    print(
        f"Loaded {initial_excerpt_total:,} total excerpts; filtering to {excerpts_df.height:,} excerpts with alt ids"
    )

    excerpts_df
    return all_excerpts_df, c, data_paths, excerpts_df, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exactly matching spans

    Use the `alt_poem_ids` field in the excerpt data to create a dataframe of poem pairs with metadata, to see which poems have exactly matching spans, and how often that occurs.
    """)
    return


@app.cell
def _(data_paths, excerpts_df, pl):
    # limit to unique set of matching ids; first look without counts, only check co-occurrence
    # (although rare co-occurrences should probably be dropped...)

    # explode list of alternate ids so we get rows of poem id pairs
    alt_poem_pairs_df = (
        excerpts_df.group_by("poem_id", "alt_poem_ids")
        .agg(pl.len().alias("n_exact_matches"))
        .explode(pl.col("alt_poem_ids"))
        .rename({"alt_poem_ids": "alt_poem_id"})
        .sort("n_exact_matches", descending=True)
    )

    # some pairs occur in either order; consolidate
    alt_poem_pairs_df = (
        alt_poem_pairs_df.with_columns(
            # make a sorted list field of the two ids
            poem_id_list=pl.concat_list(
                pl.col("poem_id"), pl.col("alt_poem_id")
            ).list.sort()
        )
        # group and aggregate
        .group_by("poem_id_list")
        .agg(
            pl.col("poem_id").first(),
            pl.col("alt_poem_id").first(),
            pl.col("n_exact_matches").sum(),
        )
        .drop("poem_id_list")
    )

    # join poem metadata so chadwyck-healey ids are intrepretable
    poem_meta_df = (
        pl.read_csv(data_paths["poem_meta"])
        .select("poem_id", "author", "title", "num_lines")
        .rename(
            {
                "author": "poem_author",
                "title": "poem_title",
                "num_lines": "poem_num_lines",
            }
        )
    )

    # join once for each poem id
    alt_poem_pairs_df = alt_poem_pairs_df.join(poem_meta_df, on="poem_id").join(
        poem_meta_df, left_on="alt_poem_id", right_on="poem_id", suffix="_alt"
    )

    alt_poem_pairs_df.sort("n_exact_matches", descending=True)
    return alt_poem_pairs_df, poem_meta_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For each unique poem id in the dataframe of poem pairs, how many poems does it match with?
    """)
    return


@app.cell
def _(alt_poem_pairs_df, c):
    alt_poem_pairs_df.group_by("poem_id").agg(
        c.alt_poem_id.len().alias("num_poems"),
        c.poem_author.first(),
        c.poem_title.first(),
        c.poem_num_lines.first(),
        c.alt_poem_id,
        c.poem_author_alt,
        c.poem_title_alt,
        c.poem_num_lines_alt,
    ).sort("num_poems", descending=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Same but for alternate poem ids: for each unique alternate poem id, how many poems does it match with?
    """)
    return


@app.cell
def _(alt_poem_pairs_df, c):
    alt_poem_pairs_df.group_by("alt_poem_id").agg(
        c.poem_id.len().alias("num_poems"),
        c.poem_id,
        c.poem_author.first(),
        c.poem_title.first(),
        c.poem_num_lines.first(),
        c.poem_author_alt,
        c.poem_title_alt,
        c.poem_num_lines_alt,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Case Study : Paradise Lost

    We know we have two complete copies of Milton's _Paradise Lost_ in the Internet Poems corpus and in the Chadwyck-Healey data; there are possibly multiple partials in CH.

    For the two full versions, how frequently are they recognized together? Do they match exactly, to they have a high degree of overlap?
    """)
    return


@app.cell
def _(all_excerpts_df, c, mo):
    # test with a couple of pairs

    poem_ids = ["John-Milton_Paradise-Lost", "Z200437755"]

    # find all excerpts with the first id
    pl_excerpts_df = all_excerpts_df.filter(c.poem_id.eq(poem_ids[0])).unique()
    # how many with the second id as alt id?
    num_with_alt = pl_excerpts_df.filter(
        c.alt_poem_ids.list.contains(poem_ids[1])
    ).height
    num_without_alt = pl_excerpts_df.filter(
        ~c.alt_poem_ids.list.contains(poem_ids[1]) | c.alt_poem_ids.is_null()
    ).height
    pct_with = (num_with_alt / pl_excerpts_df.height) * 100

    mo.md(
        f"{pl_excerpts_df.height:,} excerpts with `poem_id={poem_ids[0]}; {num_with_alt:,}` with `{poem_ids[1]}` as alt id, {num_without_alt:,} without ({pct_with:.1f}% with)"
    )
    return (poem_ids,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What are some of the examples where these ids do not occur together? (i.e.. not the exact same character span)
    """)
    return


@app.cell
def _(all_excerpts_df, c, poem_ids):
    # go back to full excerpt data (including unset alt ids), and filter
    # to either PL poem id as primary, and alternate not set as secondary (not in list or no list)
    pl_unpairedids_df = all_excerpts_df.filter(
        (c.poem_id.eq(poem_ids[0]) & (~c.alt_poem_ids.list.contains(poem_ids[1])))
        | (c.poem_id.eq(poem_ids[1]) & (~c.alt_poem_ids.list.contains(poem_ids[0])))
        | c.poem_id.is_in(poem_ids) & c.alt_poem_ids.is_null()
    )
    pl_unpairedids_df
    return (pl_unpairedids_df,)


@app.cell
def _(all_excerpts_df, c):
    # what are the excerpts on that first page ?
    # all are from PL; the first two are the same passage but don't align exactly
    pl_onepage_exc_df = all_excerpts_df.filter(c.page_id.eq("CW0106390819.0044"))
    pl_onepage_exc_df
    return (pl_onepage_exc_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use the method for identifying spans with a high degree of overlap to check these, first on the context of one page with known overlap, and then on all the Paradise Lost excerpts from either copy.
    """)
    return


@app.cell
def _(pl_onepage_exc_df):
    # calculate overlap factor for this set, to check
    from corppa.poetry_detection.merge_excerpts import (
        identify_overlapping_excerpts,
    )

    # default logic identifies the expected pair - overlap length of 181, factor of 0.98
    identify_overlapping_excerpts(pl_onepage_exc_df)
    return (identify_overlapping_excerpts,)


@app.cell
def _(identify_overlapping_excerpts, pl_unpairedids_df):
    # what is the overlap for the unpaired PL excerpts?
    # we have to decrease overlap factor significantly to find more than a few
    # FIXME: this is only one half of the excerpts, need to pull in the other id
    pl_unpaired_overlaps_df = identify_overlapping_excerpts(
        pl_unpairedids_df, min_overlap_factor=0.8
    )
    pl_unpaired_overlaps_df
    return (pl_unpaired_overlaps_df,)


@app.cell
def _(pl_unpaired_overlaps_df):
    pl_unpaired_overlaps_df.select(
        "page_id",
        "ppa_span_text",
        "ref_span_text",
        "ref_span_text_right",
        "overlap_len",
        "overlap_factor",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Is either one ever recognized separately? What do we lose if we simply drop one?
    """)
    return


@app.cell
def _(all_excerpts_df, c, pl, poem_ids):
    # filter all excerpts to those for either PL id; aggregate by page
    all_pl_excerpts_df = all_excerpts_df.filter(c.poem_id.is_in(poem_ids))
    pl_page_excerpts_df = (
        all_pl_excerpts_df.group_by("page_id")
        .agg(
            pl.len().alias("num_excerpts"),
            c.poem_id,
            c.alt_poem_ids.explode().unique(),
        )
        .with_columns(all_poem_ids=c.poem_id.list.set_union(c.alt_poem_ids))
        .with_columns(
            has_ch=c.all_poem_ids.list.contains(poem_ids[1]),
            has_ip=c.all_poem_ids.list.contains(poem_ids[0]),
        )
        .with_columns(has_both=c.has_ch & c.has_ip)
    )
    pl_page_excerpts_df
    return all_pl_excerpts_df, pl_page_excerpts_df


@app.cell
def _(c, mo, pl_page_excerpts_df):
    # summarize findings
    total_pl_pages = pl_page_excerpts_df.height
    total_pages_both = pl_page_excerpts_df.filter(c.has_both).height

    total_pages_only_one = pl_page_excerpts_df.filter(
        (c.has_ch & ~c.has_ip) | (c.has_ip & ~c.has_ch)
    ).height

    total_ch = pl_page_excerpts_df.filter(c.has_ch).height
    total_ip = pl_page_excerpts_df.filter(c.has_ip).height

    mo.md(f"""
    - {total_pl_pages:,} total pages with excerpts from _Paradise Lost_ (either of the main complete versions)
    - {total_ch:,} pages have excerpts from Chadwyck-Healey version ({(total_ch / total_pl_pages) * 100:.0f}%)
    - {total_ip:,} pages have excerpts from Internet Poems version ({(total_ip / total_pl_pages) * 100:.0f}%)
    - {total_pages_both:,} pages with both ids ({(total_pages_both / total_pl_pages) * 100:.0f}%)
    - {total_pages_only_one:,} pages with only one and not the other ({(total_pages_only_one / total_pl_pages) * 100:.0f}%)
    """)
    return


@app.cell
def _(all_pl_excerpts_df, c, pl_page_excerpts_df):
    # are the non-overlapping excerpts bad matches?
    only_one_pl_pages_df = pl_page_excerpts_df.filter(
        (c.has_ch & ~c.has_ip) | (c.has_ip & ~c.has_ch)
    )

    all_pl_excerpts_df.filter(
        c.page_id.is_in(only_one_pl_pages_df["page_id"].implode())
    ).select("page_id", "ppa_span_text", "ref_span_text", "poem_id")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Partial overlapping excerpts

    Now run the method to detect substantially overlapping excerpts on all the excerpt data.
    """)
    return


@app.cell
def _(all_excerpts_df, identify_overlapping_excerpts):
    # get list of high overlap excerpts from all excerpt data

    excerpt_overlaps_df = identify_overlapping_excerpts(
        all_excerpts_df, min_overlap_factor=0.9
    )
    excerpt_overlaps_df
    return (excerpt_overlaps_df,)


@app.cell
def _(all_excerpts_df, excerpt_overlaps_df, poem_meta_df):
    # use the list of high-overlap excerpts to get pairs of poems

    # join all excerpts with poem metadata; limit to fields needed for joining and poem metadata
    all_excerpts_poems_df = all_excerpts_df.join(poem_meta_df, on="poem_id").select(
        "page_id",
        "excerpt_id",
        "poem_id",
        "poem_title",
        "poem_author",
        "poem_num_lines",
    )

    # join twice, once for each excerpt id in the pair
    excerpt_overlap_poems_df = excerpt_overlaps_df.join(
        all_excerpts_poems_df, on=["page_id", "excerpt_id"]
    ).join(
        all_excerpts_poems_df,
        left_on=["page_id", "excerpt_id_right"],
        right_on=["page_id", "excerpt_id"],
        suffix="_right",
    )
    excerpt_overlap_poems_df
    return (excerpt_overlap_poems_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use the set of overlapping excerpts to get a list of poems with frequent high overlap.
    """)
    return


@app.cell
def _(c, excerpt_overlap_poems_df, pl):
    # some pairs occur in either order; consolidate
    overlap_poem_pairs_df = (
        excerpt_overlap_poems_df.sort("poem_id")
        .with_columns(
            # make a sorted list field of the two ids
            poem_id_list=pl.concat_list(c.poem_id, c.poem_id_right).list.sort()
        )
        # group and aggregate
        .group_by("poem_id_list")
        .agg(
            # rename to match the earlier poem pairs structure, so they can be combined
            c.poem_id.first(),
            c.poem_id_right.first().alias("alt_poem_id"),
            c.poem_title.first(),
            c.poem_author.first(),
            c.poem_title_right.first().alias("alt_poem_title"),
            c.poem_author_right.first().alias("alt_poem_author"),
            pl.len().alias("n_90pct_overlap"),
        )
        .drop("poem_id_list")
    )

    overlap_poem_pairs_df.sort("n_90pct_overlap", descending=True)
    return (overlap_poem_pairs_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Combine this information on high-overlap poem ids with the previous set of poem pairs with exactly matching spans, and save to CSV for researcher review.
    """)
    return


@app.cell
def _(alt_poem_pairs_df, overlap_poem_pairs_df):
    poem_pairs_exact_partial_df = alt_poem_pairs_df.join(
        overlap_poem_pairs_df, on=["poem_id", "alt_poem_id"]
    ).sort("n_exact_matches", "n_90pct_overlap", descending=True)

    # re-order so counts are together early in the columns
    first_cols = ["poem_id", "alt_poem_id", "n_exact_matches", "n_90pct_overlap"]
    ordered_cols = first_cols + [
        col for col in poem_pairs_exact_partial_df.columns if col not in first_cols
    ]

    poem_pairs_exact_partial_df = poem_pairs_exact_partial_df.select(ordered_cols)
    poem_pairs_exact_partial_df
    return (poem_pairs_exact_partial_df,)


@app.cell
def _(poem_pairs_exact_partial_df):
    # save the list of alternate poem ids to a file

    poem_pairs_exact_partial_df.sort(
        "n_exact_matches", "n_90pct_overlap", descending=True
    ).write_csv("alt_poem_ids.csv")
    return


@app.cell
def _(c, pl, poem_pairs_exact_partial_df):
    # make a version for the starting point of poem cluster ids
    # get poet last name and simplified title, construct slugs for pairs,
    # and then make a list that can be split and rejoined with poem metadata

    poem_cluster_id_df = (
        poem_pairs_exact_partial_df.filter(c.n_exact_matches.gt(100))
        .fill_null("")
        .with_columns(
            poem_id=pl.concat_list([c.poem_id, c.alt_poem_id]),
            author_lname=c.poem_author.str.split(" ").list.last(),
            alt_author_lname=c.alt_poem_author.str.split(" ").list.last(),
            title_slug=c.poem_title.str.to_lowercase()
            .str.replace(r"\b(the|and|or|a|of|on)\b", "")
            .str.replace("'", ""),
        )
        .filter(
            c.author_lname.eq(c.alt_author_lname)
            | c.author_lname.eq("")
            | c.alt_author_lname.eq("")
        )
        .with_columns(
            cluster_id=pl.concat_str(
                [c.author_lname.str.to_lowercase(), c.title_slug], separator=" "
            )
            .str.replace_all(r"\s+", "-")
            .str.strip_chars("-.:")
        )
    )
    poem_cluster_id_df
    return (poem_cluster_id_df,)


@app.cell
def _(c, poem_cluster_id_df, poem_meta_df):
    # explode pair of poem ids into rows, poem id + cluster id, combine with poem metadata & overlap info
    poem_cluster_ids = (
        (
            poem_cluster_id_df.select("poem_id", "cluster_id")
            .explode("poem_id")
            .join(poem_meta_df, on="poem_id")
            .join_where(
                poem_cluster_id_df.select(
                    "poem_id", "n_exact_matches", "n_90pct_overlap"
                ),
                c.poem_id.is_in(c.poem_id_right),
            )
            .with_columns(id_pair=c.poem_id_right.list.join("; "))
            .drop("poem_id_right")
        )
        .unique(["poem_id", "cluster_id"])
        .sort("n_exact_matches", descending=True)
    )
    poem_cluster_ids
    return (poem_cluster_ids,)


@app.cell
def _(poem_cluster_ids):
    # save to CSV file for use in alt poem id spreadsheet
    poem_cluster_ids.write_csv("poem_cluster_ids.csv")
    return


if __name__ == "__main__":
    app.run()
