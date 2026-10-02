# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "polars",
#     "pillow",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Review image + text alignment

    Review the image/text alignment produced by
    `corppa.utils.dataset_prep`. Point this notebook at the output
    directory (containing `ppa_pages.jsonl` and `ppa_images.tar`),
    optionally filter by work id, then sample random pages and view
    the image + text pairs side by side.

    Use the **New sample** button to draw another random set of pages.
    """)
    return


@app.cell
def _():
    import io
    import random
    import tarfile
    from pathlib import Path

    import polars as pl
    from PIL import Image

    try:
        import orjson

        _loads = orjson.loads
    except ImportError:
        import json

        _loads = json.loads

    def load_pages(pages_path):
        """Read the pages JSONL, keeping only the scalar fields the notebook
        needs (work_id, id, text, image_path).

        We parse the JSON line-by-line and pull out just these fields rather
        than using polars' JSONL reader. polars infers one schema across the
        whole file, which fails on ragged nested fields -- e.g. a field that
        is a struct in some records and a list[struct] in others
        (``failed to determine supertype of list[struct[..]] and struct[..]``).
        Ignoring those nested fields entirely avoids the conflict."""
        wanted = ["work_id", "id", "text", "image_path"]
        records = []
        with open(pages_path, "rb") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = _loads(line)
                records.append({k: obj.get(k) for k in wanted})
        # force all columns to string so empty/missing values stay consistent
        return pl.DataFrame(records, schema={k: pl.String for k in wanted})

    return Image, Path, io, load_pages, pl, random, tarfile


@app.cell
def _(mo):
    # inputs: output directory (or explicit jsonl + tar), sample size
    data_dir_ui = mo.ui.text(
        value="",
        label="Output directory (contains ppa_pages.jsonl + ppa_images.tar)",
        full_width=True,
    )
    sample_size_ui = mo.ui.slider(
        start=1, stop=20, value=5, step=1, label="Sample size", include_input=True
    )
    mo.vstack([data_dir_ui, sample_size_ui])
    return data_dir_ui, sample_size_ui


@app.cell
def _(Path, data_dir_ui, load_pages, mo, pl):
    # load pages jsonl and resolve tar path from the output directory
    mo.stop(
        not data_dir_ui.value.strip(),
        mo.md("_Enter an output directory above to begin._"),
    )

    _data_dir = Path(data_dir_ui.value.strip()).expanduser()
    pages_path = _data_dir / "ppa_pages.jsonl"
    tar_path = _data_dir / "ppa_images.tar"

    mo.stop(
        not pages_path.exists(),
        mo.md(f"**Pages file not found:** `{pages_path}`"),
    )
    mo.stop(
        not tar_path.exists(),
        mo.md(f"**Image archive not found:** `{tar_path}`"),
    )

    # only pages that actually got an aligned image are useful for review
    pages_df = load_pages(pages_path)
    has_image = "image_path" in pages_df.columns
    if has_image:
        pages_df = pages_df.filter(pl.col("image_path").is_not_null())

    mo.stop(
        not has_image or pages_df.is_empty(),
        mo.md("**No pages with `image_path` found in the pages file.**"),
    )
    return pages_df, tar_path


@app.cell
def _(mo, pages_df):
    # optional work id filter; helps target known realigned texts
    work_ids = sorted(pages_df.get_column("work_id").unique().to_list())
    work_filter_ui = mo.ui.multiselect(
        options=work_ids,
        label=f"Filter by work id (optional; {len(work_ids):,} works)",
        full_width=True,
    )
    work_filter_ui
    return (work_filter_ui,)


@app.cell
def _(pages_df, pl, work_filter_ui):
    # apply the work-id filter (if any selected)
    if work_filter_ui.value:
        filtered_df = pages_df.filter(pl.col("work_id").is_in(work_filter_ui.value))
    else:
        filtered_df = pages_df
    return (filtered_df,)


@app.cell
def _(mo):
    resample_button = mo.ui.run_button(label="New sample", kind="success")
    resample_button
    return (resample_button,)


@app.cell
def _(filtered_df, mo, random, resample_button, sample_size_ui):
    # depend on the button and controls so a click / change draws a new sample
    resample_button

    mo.stop(
        filtered_df.is_empty(),
        mo.md("_No pages match the current filter._"),
    )

    n = min(sample_size_ui.value, filtered_df.height)
    # fresh random seed each run so the button produces a different sample
    sample_df = filtered_df.sample(n=n, seed=random.randrange(1_000_000))
    mo.md(f"Showing **{n}** of **{filtered_df.height:,}** matching pages.")
    return (sample_df,)


@app.cell
def _(Image, io, mo, sample_df, tar_path, tarfile):
    # read the sampled images out of the tar and build image+text pairs
    def _load_image(tar, image_path, max_size=(400, 550)):
        try:
            member = tar.getmember(image_path)
        except KeyError:
            return mo.md(f"_image `{image_path}` not found in tar_")
        with tar.extractfile(member) as fh:
            img = Image.open(io.BytesIO(fh.read()))
            img.load()
        img.thumbnail(max_size)
        # browsers can't render TIFF/JP2 (common in HathiTrust zips), and
        # passing a PIL image straight to mo.image can fail to display; always
        # transcode to PNG bytes so any input format renders reliably.
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return mo.image(buf.getvalue(), width=max_size[0])

    def _text_view(text):
        return mo.plain_text(text or "").style(
            {
                "white-space": "pre-wrap",
                "font-size": "0.75rem",
                "max-height": "550px",
                "overflow": "auto",
            }
        )

    pairs = []
    with tarfile.open(tar_path) as tar:
        for row in sample_df.iter_rows(named=True):
            header = mo.md(f"**{row['id']}** — `{row['image_path']}`")
            pair = mo.hstack(
                [
                    _load_image(tar, row["image_path"]),
                    _text_view(row.get("text", "")),
                ],
                widths=[1, 1],
                gap=1,
                align="start",
            )
            pairs.append(mo.vstack([header, pair]))

    mo.vstack(pairs, gap=2)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
