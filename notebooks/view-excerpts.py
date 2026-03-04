# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "corppa",
#     "marimo>=0.20.2",
#     "polars==1.38.1",
# ]
#
# [tool.uv.sources]
# corppa = { git = "https://github.com/Princeton-CDH/corppa.git", rev = "develop" }
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _(mo):
    mo.md(r"""
    # PPA found poems: manual annotation & passim

    [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_m9upHymqmyGsJXuLm59gFw)

    This notebook provides a page-level view of pages with manual annotation, showing adjudicated spans and passim-detected excerpts together in the context of the page of text in a PPA work.

    Pages are ordered by number of annotation, most annotated first.  Data can either be loaded from a precompiled parquet file of selected excerpts and PPA page content (used on molab) or from excerpt dataset and a PPA page subset.

    Use the slider to move through pages. A table of the excerpts is displayed after the page text.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
        <h2>Legend</h2>
        <div class="compare">
          <div>Content could have <mark class="adjudication">manual annotation</mark>, passim annotation, or some <mark class="adjudication">overlap of both</mark>.</div>
          <div>Content could have manual annotation, <mark class="passim">passim annotation</mark>, or <mark class="passim">some overlap of </mark>both.</div>
          <div class="spacer">Content could have manual annotation, passim annotation, or some overlap of both.</div>
        </div>""")
    return


@app.cell(hide_code=True)
def _(annotated_pages_df, mo):
    page_slider = mo.ui.slider(
        start=0,
        stop=annotated_pages_df.height - 1,
        step=1,
        label="annotated page",
        include_input=True,
    )

    page_slider
    return (page_slider,)


@app.cell(hide_code=True)
def _(annotated_pages_df, excerpts_annopages_df, mo, page_slider, pl):
    # page through annotated pages in order of # annotations

    manually_annotated_pages_df = excerpts_annopages_df.group_by("page_id").agg(
        pl.col("ppa_work_id").first().alias("work_id"),
        pl.col("page_num").first(),
        pl.len().alias("count"),
    )

    page = manually_annotated_pages_df.sort(pl.col("count"), descending=True).row(
        page_slider.value, named=True
    )

    selected_page = annotated_pages_df.filter(pl.col("id").eq(page["page_id"])).row(
        0, named=True
    )

    # we want ALL annotations for this page, not just manual
    page_annotations = (
        excerpts_annopages_df.filter(pl.col("page_id").eq(selected_page["id"]))
        .with_columns(system=pl.col("detection_methods").list.join(","))
        .sort("ppa_span_start", "ppa_span_end")
    )

    # get a list of tuples for start and end of each span to highlight
    adj_spans = (
        page_annotations.filter(pl.col("system").eq("adjudication"))
        .select("ppa_span_start", "ppa_span_end", "system")
        .rows()
    )

    passim_spans = (
        page_annotations.filter(pl.col("system").eq("passim"))
        .select("ppa_span_start", "ppa_span_end", "system")
        .rows()
    )

    mo.vstack(
        [
            # preserve whitespace
            mo.Html(
                f"""<section class="page">
                <header>{selected_page["work_id"]} page {selected_page["label"]} ({len(adj_spans)} adjudication spans, {len(passim_spans)} passim spans)</header>
                <div class="compare">
                  <div>{highlight_spans(selected_page["text"], adj_spans)}</div>
                  <div>{highlight_spans(selected_page["text"], passim_spans)}</div>
                  <div class="spacer">{selected_page["text"]}</div>
                </div>
                </section>"""
            ),
            mo.ui.table(page_annotations),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    import pathlib

    import marimo as mo
    import polars as pl

    return mo, pathlib, pl


@app.cell(hide_code=True)
def _(mo):
    # apply stylesheet to customize highlighting
    # NOTE: we apply styles here because molab doesn't support setting app-level custom css
    css_content = (mo.notebook_dir() / "highlight.css").read_text()

    mo.Html(f"""
    <style>
    {css_content}
    </style>
    """)
    return


@app.function(hide_code=True)
def highlight_spans(text: str, spans: list[tuple[int]]) -> str:
    # method to add <mark> highlighting for one or more spans within a text string
    # takes text and string with one or more spans in a format that can be parsed by intspan
    # returns the text with <mark> tags around the highlighted regions
    previous_end = 0
    text_parts = []
    for i, (start, end, label) in enumerate(spans):
        if previous_end >= end:
            continue
        # text up to the next the mark (if not overlapping / not already started)
        if previous_end < start:
            text_parts.append(text[previous_end:start])
        else:
            # if previous end is greater than current span start,
            # pick up where we left off

            start = previous_end
        # text to be highlighted
        # if this span ends before the next (no overlap), just output span
        if i < len(spans) - 1:
            next_span_start, next_span_end, next_label = spans[i + 1]
            if end < next_span_start:
                # no overlap - highlight span
                text_parts.append(f"<mark class='{label}'>{text[start:end]}</mark>")
            else:
                # spans overlap
                # highlight segment of current span before the next starts
                text_parts.append(
                    f"<mark class='{label}'>{text[start:next_span_start]}</mark>"
                )
                # does this one end before the next?
                # highlight segment of current span overlapping with next span
                if end < next_span_end:
                    text_parts.append(
                        f"<mark class='{label} {next_label}'>{text[next_span_start:end]}</mark>"
                    )
                    # set previous end to end of this span
                    previous_end = end
                else:
                    # next span is entirely contained within this one
                    # output entirety of next span with both labels
                    text_parts.append(
                        f"<mark class='{label} {next_label}'>{text[next_span_start:next_span_end]}</mark>"
                    )
                    # output the rest of this span after the contained span
                    text_parts.append(
                        f"<mark class='{label}'>{text[next_span_end:end]}</mark>"
                    )
                    previous_end = end
        else:
            # last segment
            text_parts.append(f"<mark class='{label}'>{text[start:end]}</mark>")
        # set previuos end to the portion after this span
        previous_end = end
    # append any text after the last highlighted portion
    text_parts.append(text[previous_end:])
    return "".join(text_parts)


@app.cell
def _(pathlib, pl):
    from corppa.config import get_config
    from corppa.poetry_detection.polars_utils import load_excerpts_df

    excerpt_parquet_file = pathlib.Path("ppa_excerpts_annotatedpages.parquet")
    annotated_pages_file = pathlib.Path("manually_annotated_pages.jsonl")

    def get_excerpt_data():
        config_opts = get_config()
        data_dir = pathlib.Path(config_opts["compiled_dataset"]["data_dir"])
        if not data_dir.exists() or not data_dir.is_dir():
            raise ValueError(
                f"Data directory {data_dir} not found. "
                + "\nCheck your configuration file, and remember to use an absolute path for the poem dataset data directory."
            )
        else:
            print(f"Loading excerpt data from {data_dir}")

        # Create a dictionary of data files for lookup based on file base name without any extension
        # so that excerpts data can be .csv or compressed .csv.gz
        data_paths = {
            data_file.stem.split(".", 1)[0]: data_file
            for data_file in data_dir.iterdir()
        }

        # load the excerpts into a polars dataframe
        # we need ppa work id to generate file for creating filtered page subset
        return load_excerpts_df(
            data_paths["excerpts"], ppa_works_meta=data_paths["ppa_work_metadata"]
        )

    def get_excerpts_annotatedpages():
        # get data for excerpts on annotated pages

        # load precompiled data if present
        if excerpt_parquet_file.exists():
            print(f"Loading precompiled data from {excerpt_parquet_file}")
            return pl.read_parquet(excerpt_parquet_file)
        else:
            print(
                f"Precompiled data file {excerpt_parquet_file} not found; loading and filtering excerpts."
            )
            # load the excerpts into a polars dataframe, joining ppa and poetry data
            excerpts_df = get_excerpt_data().filter(
                # limit to adjudication excerpts
                pl.col("detection_methods").list.contains("adjudication")
            )
            # save for future runs
            excerpts_df.write_parquet(excerpt_parquet_file)
            return excerpts_df

    def get_annotatedpages(excerpts_annopages_df):
        # load annotated page data
        if annotated_pages_file.exists():
            print(f"Loading annotated page subset from {annotated_pages_file}")
            return pl.read_ndjson(annotated_pages_file)
        else:
            # if file is not found, document how to create it
            print(
                f"\nPage data for manually annotated pages not found: {annotated_pages_file}"
            )
            # generate a list of work & page ids to generate a page data subset
            # load the excerpts into a polars dataframe, joining ppa
            page_csvfilename = "manual_annotation_pages.csv"
            excerpts_annopages_df.group_by("page_id").agg(
                pl.col("ppa_work_id").first().alias("work_id"),
                pl.col("page_num").first(),
                # output in the format supported by corpppa-filter --pgfile
            ).write_csv(page_csvfilename)

            print(f"""Use corppa-filter to generate the page subset file: 
        
    corppa-filter --pgfile {page_csvfilename} path/to/ppa_corpus_2026-XX-XX/ppa_pages.jsonl.gz {annotated_pages_file}
    """)
            return None

    # load excerpts for annotated pages (previously or dynamically filtered)
    excerpts_annopages_df = get_excerpts_annotatedpages()

    # load annotated pages
    annotated_pages_df = get_annotatedpages(excerpts_annopages_df)
    return annotated_pages_df, excerpts_annopages_df


if __name__ == "__main__":
    app.run()
