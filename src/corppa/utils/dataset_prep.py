# prep ppa text+image dataset for publication
import argparse
import tarfile
from pathlib import Path
from typing import Iterator, Optional
from zipfile import ZipFile

import orjsonl
import polars as pl
import polars_ds as pds
from tqdm import tqdm

from corppa.utils.path_utils import encode_htid, get_ppa_source, get_volume_id


def get_zip_textfiles(zipfile_path: Path) -> Iterator[tuple[str, str]]:
    """Return a generator of text files from a zip archive. Returns tuples of (file_id, content)
    where file_id is the stem of the filename."""
    with ZipFile(zipfile_path) as ht_zip:
        txtfile_list = [fn for fn in ht_zip.namelist() if fn.endswith(".txt")]
        for filename in txtfile_list:
            with ht_zip.open(filename) as txtfile:
                file_id = Path(filename).stem
                content = txtfile.read().decode("utf-8")
                yield (file_id, content)


def add_zip_file_to_tar(
    zf: ZipFile, zip_filename: str, tar: tarfile.TarFile, arcname: str
) -> None:
    """Add a single file from an open ZipFile to an open TarFile without
    extracting to disk. arcname sets the path inside the tar archive."""
    info = zf.getinfo(zip_filename)
    tarinfo = tarfile.TarInfo(name=arcname)
    tarinfo.size = info.file_size
    with zf.open(info) as f:
        tar.addfile(tarinfo, fileobj=f)


# determine alignment between pages in different versions of hathitrust
def align_pages(work_id: str, pages_df: pl.DataFrame, zipfile_path: Path):  #  -> dict:
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
    tqdm.write(
        f"{work_id} - {pages_df.height:,} pages; average text match score: {avg:.3f}"
    )
    # might be lower than this; at least one 0.87 is probably correct alignment
    if avg is not None and avg > 0.9:
        return {
            row["page_id"]: row["page_filename"]
            for row in pages_join_df.select(["page_id", "page_filename"]).iter_rows(
                named=True
            )
        }
    # TODO: handle case where we need to determine shifted alignment

    print(
        pages_join_df.filter(pl.col.text.is_not_null())
        .select(["id", "text", "text_right", "text_match", "page_filename"])
        .sort("text_match", descending=False)
        .head(10)
    )


def process_work(work_id: str, pages: list[dict], image_dir: Path) -> None:
    # generic process work method, which calls appropriate source-specific method
    match get_ppa_source(work_id):
        case "Gale":
            pass  #
            # process_gale_work(work_id, pages, image_dir)
        case "HathiTrust":
            process_ht_work(work_id, pages, image_dir)
        case "ECCO":
            pass  # no images


def process_ht_work(work_id: str, pages: list[dict], image_dir: Path) -> None:
    htid = get_volume_id(work_id)
    htid_suffix = htid.split(".")[-1]
    zipfile_path = image_dir / encode_htid(htid) / f"{htid_suffix}.zip"
    if not zipfile_path.exists():
        print(f"Warning: zipfile {zipfile_path} does not exist, skipping")
        return
    _page_mapping = align_pages(work_id, pl.DataFrame(pages), zipfile_path)


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
        "output_dir",
        help="Directory where the updated page corpus and image archive file should be saved",
        type=Path,
    )

    args = parser.parse_args()

    if not args.output_dir.is_dir():
        args.output_dir.mkdir(parents=True, exist_ok=True)
    output_pages_path = args.output_dir / "ppa_pages.jsonl.gz"
    output_archive_path = args.output_dir / "ppa_images.tar.gz"
    if output_pages_path.exists():
        print(f"Warning: output file {output_pages_path} already exists, overwriting")
    if output_archive_path.exists():
        print(f"Warning: output file {output_archive_path} already exists, overwriting")

    # Stream pages one at a time; corpus is sorted by work+page so we can
    # process pages by work as the work_id changes.
    with tarfile.open(output_archive_path, "w:gz") as _tar_filehandle:
        prev_work_id: Optional[str] = None
        pages: list[dict] = []
        for page in tqdm(orjsonl.stream(args.input), desc="Reading pages"):
            work_id = page["work_id"]
            # when work id changes, process the previous work pages and reset for the next
            if work_id != prev_work_id:
                if prev_work_id is not None:
                    process_work(prev_work_id, pages, args.image_dir)
                prev_work_id = work_id
                pages = []
            pages.append(page)

    # handle the pages for the last work at end of loop
    if prev_work_id is not None:
        process_work(prev_work_id, pages, args.image_dir)


if __name__ == "__main__":
    main()
