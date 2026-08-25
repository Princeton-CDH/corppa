# prep ppa text+image dataset for publication
import argparse
from pathlib import Path
from typing import Iterator
from zipfile import ZipFile

import polars as pl
import polars_ds as pds

from corppa.utils.path_utils import encode_htid, get_volume_id


def get_zip_textfiles(zipfile_path: Path) -> Iterator[tuple[str, str]]:
    with ZipFile(zipfile_path) as ht_zip:
        txtfile_list = [fn for fn in ht_zip.namelist() if fn.endswith(".txt")]
        for filename in txtfile_list:
            with ht_zip.open(filename) as txtfile:
                file_id = Path(filename).stem
                content = txtfile.read().decode("utf-8")
                yield (file_id, content)


# determine alignment between pages in different versions of hathitrust
def align_pages(pages_df: pl.DataFrame, zipfile_path: Path):  #  -> dict:
    expected_page_count = pages_df.height
    # load text files from zipfile into a polars dataframe
    zip_pages_df = pl.DataFrame(
        data=get_zip_textfiles(zipfile_path),
        schema=["page_filename", "text"],
    ).with_columns(
        # extract the numeric page id for joining with page data
        # some works have filenames like OSU_32435051461309_00000602 ; others are simply numeric
        page_id=pl.col.page_filename.str.extract(r"_?([0-9]+$)", 1)
    )
    # NOTE: for excerpt, page count is not expected to match but should be >= total
    if zip_pages_df.height < expected_page_count:
        print(
            f"Warning: insufficient pages found in zipfiles ({zip_pages_df.height}; expected at least {expected_page_count})"
        )
    # extract bare page id from work_id.page_id globally unique page identifier
    pages_join_df = (
        pages_df.with_columns(page_id=pl.col.id.str.extract(r"[._]([0-9]+$)", 1))
        # page_id=pl.col.id.str.split(".").list.last())
        .join(zip_pages_df, on="page_id")
        .with_columns(text_match=pds.str_fuzz("text", "text_right", parallel=True))
    )
    if expected_page_count != pages_join_df.height:
        print(
            f"Warning: joined pages ({pages_join_df.height}) does not match expected page count ({expected_page_count})"
        )
        print(zip_pages_df.head())
        print(pages_df.head())
        return

    # maybe filter out pages with no text when checking score? (probably omits nulls anyway...)

    # for now, just report the average score
    avg = pages_join_df["text_match"].mean()
    print(avg)
    if avg is not None and avg > 0.9:
        return {
            row["page_id"]: row["page_filename"]
            for row in pages_join_df.select(["page_id", "page_filename"]).iter_rows(
                named=True
            )
        }

    # if avg is None:
    # print(pages_join_df.head())
    print(
        pages_join_df.filter(pl.col.text.is_not_null())
        .select(["id", "text", "text_right", "text_match", "page_filename"])
        .sort("text_match", descending=False)
        .head(10)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare PPA full-text dataset for publication by aligning pages and organizing images",
    )
    parser.add_argument(
        "input",
        help="PPA full-text corpus; must be a JSONL file (compressed or not)",
        type=Path,
    )
    parser.add_argument(
        "image_dir",
        help="Base directory for images",
        type=Path,
    )
    parser.add_argument(
        "output",
        help="Filename where the updated corpus should be saved",
        type=Path,
    )

    args = parser.parse_args()

    # for now just read into a dataframe (will probably need orjsonl or lazy for full dataset)
    pages_df = pl.read_ndjson(args.input)
    work_ids = pages_df["work_id"].unique().to_list()
    for work_id in work_ids:
        # get pages for just this work
        work_pages_df = pages_df.filter(pl.col("work_id") == work_id)
        htid = get_volume_id(work_id)
        htid_suffix = htid.split(".")[-1]
        # construct path to the zip file that includes the text and images
        zipfile_path = args.image_dir / encode_htid(htid) / f"{htid_suffix}.zip"
        if not zipfile_path.exists():
            print(f"Warning: zipfile {zipfile_path} does not exist, skipping")
            continue
        print(f"{work_id} - {work_pages_df.height:,} pages")
        _page_mapping = align_pages(work_pages_df, zipfile_path)


if __name__ == "__main__":
    main()
