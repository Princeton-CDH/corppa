# prep ppa text+image dataset for publication
import argparse
import tarfile
from pathlib import Path
from time import mktime
from typing import Iterator, Optional
from zipfile import ZipFile

import orjsonl
import polars as pl
import polars_ds as pds
from tqdm import tqdm

from corppa.utils.path_utils import (
    encode_htid,
    get_ppa_source,
    get_vol_dir,
    get_volume_id,
)


def get_zip_textfiles(zipfile: ZipFile) -> Iterator[tuple[str, str]]:
    """Return a generator of text files from an open zip archive. Returns tuples of (file_id, content)
    where file_id is the stem of the filename."""
    txtfile_list = [fn for fn in zipfile.namelist() if fn.endswith(".txt")]
    for filename in txtfile_list:
        with zipfile.open(filename) as txtfile:
            # path stem is the filename without the extension; return file id + contents
            yield (Path(filename).stem, txtfile.read().decode("utf-8"))


def add_zip_file_to_tar(
    zipfile: ZipFile,
    zip_filename: str,
    tar: tarfile.TarFile,
    tar_file_path: str,
) -> None:
    """Add a single file from an open ZipFile to an open TarFile without
    extracting to disk. tar_file_path sets the path within the tar archive."""
    zipinfo = zipfile.getinfo(zip_filename)
    # create tar info object for destination path with size and modification time
    tarinfo = tarfile.TarInfo(name=tar_file_path)
    tarinfo.size = zipinfo.file_size
    # convert zip info modification time to tar info mtime
    tarinfo.mtime = mktime(zipinfo.date_time + (0, 0, -1))  # convert to timestamp
    with zipfile.open(zipinfo) as f:
        tar.addfile(tarinfo, fileobj=f)


def get_zip_imgexts(zipfile: ZipFile) -> list[str]:
    """HathiTrust zip files include in images in multiple formats; returns
    a list of all unique image extensions found in the zip file."""
    exts = set()
    for filename in zipfile.namelist():
        file_path = Path(filename)
        # compare case-insensitive but return actual case
        if file_path.suffix.lower() in [".tif", ".jpg", ".jpeg", ".jp2"]:
            exts.add(file_path.suffix)

    return list(exts)


# determine alignment between pages in different versions of hathitrust
def align_pages(work_id: str, pages_df: pl.DataFrame, zipfile: ZipFile):  #  -> dict:
    expected_page_count = pages_df.height
    # load text files from zipfile into a polars dataframe
    zip_pages_df = pl.DataFrame(
        data=get_zip_textfiles(zipfile),
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


def process_work(
    work_id: str, pages: list[dict], image_dir: Path, tar: tarfile.TarFile
) -> Iterator[dict]:
    # generic process work method, which calls appropriate source-specific method
    match get_ppa_source(work_id):
        case "Gale":
            yield from process_gale_work(work_id, pages, image_dir, tar)
        case "HathiTrust":
            yield from process_ht_work(work_id, pages, image_dir, tar)
        case "ECCO":
            yield from pages  # no images


def process_gale_work(
    work_id: str, pages: list[dict], image_dir: Path, tar: tarfile.TarFile
) -> Iterator[dict]:
    vol_img_dir = image_dir / get_vol_dir(get_volume_id(work_id))
    if vol_img_dir.is_dir():
        print(f"{work_id} : {vol_img_dir} : {len(pages)} pages")
        for page in pages:
            # page id is work id + sequence, e.g. CB0127060085.0005
            # image filename can be constructed directly from page id
            image_path = vol_img_dir / f"{page['id'].replace('.', '_')}0.TIF"
            if image_path.is_file():
                tar_image_path = f"{work_id}/{image_path.name}"
                tar.add(image_path, arcname=tar_image_path)
                # add the image path in the tar file to the page data
                page["image_path"] = tar_image_path
            # yield page data either way (with or without image path)
            yield page
    else:
        yield pages


def process_ht_work(
    work_id: str, pages: list[dict], image_dir: Path, tar: tarfile.TarFile
) -> Iterator[dict]:
    htid = get_volume_id(work_id)
    htid_suffix = htid.split(".")[-1]
    zipfile_path = image_dir / encode_htid(htid) / f"{htid_suffix}.zip"
    if not zipfile_path.exists():
        print(f"Warning: zipfile {zipfile_path} does not exist, omitting images")
        # yield pages without image paths
        yield from pages
    else:
        with ZipFile(zipfile_path) as ht_zip:
            page_mapping = align_pages(work_id, pl.DataFrame(pages), ht_zip)
            if not page_mapping:
                print(
                    f"Warning: no page mapping found for work {work_id}, omitting images"
                )
                # yield pages without image paths
                yield from pages
            else:
                # when image mapping was returned, add images to tar file and image paths to page data
                img_exts = get_zip_imgexts(ht_zip)
                # can we guarantee page mapping matches pages order?
                for page in pages:
                    page_id = page["id"].split(".")[-1]
                    # get the corresponding image from the zip, add to the tar file with appropriate name,
                    # and add the image path to the page record for output
                    page_basename = page_mapping.get(page_id)

                    # add the image from the corresponding path in the zipfile to the
                    # appropriate path for this page in the tarfile
                    file_namelist = ht_zip.namelist()
                    if page_basename is not None:
                        zip_image_basepath = f"{htid_suffix}/{page_basename}"
                        for img_ext in img_exts:
                            zip_image_path = f"{zip_image_basepath}{img_ext}"
                            if zip_image_path in file_namelist:
                                break
                        tar_image_path = f"{encode_htid(htid)}/{page_id}{img_ext}"
                        try:
                            add_zip_file_to_tar(
                                ht_zip, zip_image_path, tar, tar_image_path
                            )
                            # if adding succeeded, add the image path in the page record for output
                            page["image_path"] = tar_image_path
                        except KeyError:
                            has_text = page["text"].strip() != ""
                            if has_text:
                                print(
                                    f"Warning: image {zip_image_path} not found in zipfile but page has text; skipping"
                                )
                            print([f for f in file_namelist if page_basename in f])

                        yield page


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
    output_pages_path = (
        args.output_dir / "ppa_pages.jsonl"
    )  # .gz # disable compression for now, for testing
    output_archive_path = args.output_dir / "ppa_images.tar.gz"
    if output_pages_path.exists():
        # because we extend, we need to rename any existing outpt file
        old_output_pages = output_pages_path.with_suffix(".jsonl.bak")
        output_archive_path.rename(old_output_pages)
        print(
            f"Warning: output file {output_pages_path} exists; renamed to {old_output_pages}"
        )
    if output_archive_path.exists():
        print(f"Warning: output file {output_archive_path} already exists, overwriting")

    # Stream pages one at a time; corpus is sorted by work+page so we can
    # process pages by work as the work_id changes.
    with tarfile.open(output_archive_path, "w:gz") as tar:
        prev_work_id: Optional[str] = None
        pages: list[dict] = []
        for page in tqdm(orjsonl.stream(args.input), desc="Reading pages"):
            work_id = page["work_id"]
            # when work id changes, process the previous work pages and reset for the next
            if work_id != prev_work_id:
                if prev_work_id is not None:
                    pages = process_work(prev_work_id, pages, args.image_dir, tar)
                    orjsonl.extend(output_pages_path, list(pages))
                prev_work_id = work_id
                pages = []
            pages.append(page)

        # handle the pages for the last work at end of loop
        if prev_work_id is not None:
            pages = process_work(prev_work_id, pages, args.image_dir, tar)
            orjsonl.extend(output_pages_path, list(pages))


if __name__ == "__main__":
    main()
