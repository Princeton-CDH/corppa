# prep ppa text+image dataset for publication
import argparse
import tarfile
from pathlib import Path
from time import mktime, perf_counter
from typing import Iterator, Optional
from zipfile import ZipFile

import orjsonl
import polars as pl
import polars_ds as pds
import rapidfuzz
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


def align_shifted_pages(pages_df: pl.DataFrame, zip_pages_df: pl.DataFrame):
    # Implementation for finding alignment between pages
    # calculate text length - may need to filter out short pages to avoid mismatches
    # ensure pages are sorted by order so we can compare first, middle, and last pages for alignment
    pages_df = pages_df.sort("order", descending=False).with_row_index()
    orig_pages_df = pages_df  # save a copy of the unfiltered set
    # filter out short pages to avoid unreliable matches
    pages_df = pages_df.with_columns(text_len=pl.col.text.str.len_chars()).filter(
        pl.col.text_len.gt(600)
    )
    # empty mapping frame; also used as an early-return value for chunks we
    # can't align (see short-chunk guard below)
    empty_mapping_df = pl.DataFrame(
        schema={"id": pl.String, "page_filename": pl.String}
    )
    # No long-enough pages at all - can't determine a shift, so give up.
    # This is also the recursion base case for chunks where every page is short.
    total_pages = pages_df.height
    if total_pages == 0:
        return empty_mapping_df
    # positions (in the filtered pages df) of the anchor pages we'll compare
    # against the zip pages. Normally we sample first/middle/last long pages;
    # when the filtered set is small (< 3), sampling would produce duplicate
    # anchors, so use every long page as an anchor instead.
    if total_pages < 3:
        anchor_positions = list(range(total_pages))
    else:
        anchor_positions = [0, total_pages // 2, total_pages - 1]
    anchor_texts = pages_df["text"].gather(anchor_positions).to_list()

    scores = rapidfuzz.process.cdist(
        anchor_texts,
        zip_pages_df["text"].to_list(),
        scorer=rapidfuzz.fuzz.ratio,
        workers=-1,
        score_cutoff=85,
    )

    prev_index = None
    prev_shift = None
    # iterate over the resulting scores for each anchor; if an alignment is
    # consistent between two pairs, generate the page id to filename mapping
    # for that chunk.

    # accumulator: page id -> filename mappings across all aligned chunks
    page_mapping_df = empty_mapping_df

    # search_i indexes into anchor_positions and the rows of scores;
    # anchor_pos is the row index in the filtered pages df, which lets us
    # get back to the corresponding original page data via `page["index"]`
    last_search_i = len(anchor_positions) - 1
    for search_i, anchor_pos in enumerate(anchor_positions):
        page = pages_df.row(anchor_pos, named=True)
        # get the index of the highest scoreh in the zip pages for this search text
        zip_page_i = scores[search_i].argmax()
        # get the value for the best score
        match_score = scores[search_i][zip_page_i]
        if match_score:
            # get the data for the matched page
            zip_page = zip_pages_df.row(zip_page_i, named=True)
            # how much did pages shift?
            shift = page["order"] - zip_page["order"]
        else:
            # if match score is zero, it fell below our threshold - no good match was found
            shift = None

        # head chunk: on the first iteration, map any pages before the first
        # anchor (typically short pages that were filtered out) using this
        # anchor's shift so the mapping covers the start of the work. When
        # this is also the only anchor (single-page filtered set), extend
        # to the end of pages since the between-anchors branch will never fire.
        if search_i == 0 and shift is not None:
            head_end = (
                orig_pages_df.height if search_i == last_search_i else page["index"]
            )
            if head_end > 0:
                head_chunk_df = orig_pages_df.slice(0, head_end)
                zip_shift_df = zip_pages_df.with_columns(
                    aligned_order=pl.col("order") + shift
                )
                head_mapping_df = head_chunk_df.join(
                    zip_shift_df,
                    left_on="order",
                    right_on="aligned_order",
                    how="left",
                ).select(["id", "page_filename"])
                page_mapping_df = page_mapping_df.vstack(head_mapping_df)

        # generate mapping for chunk between this one and the previous
        if prev_shift is not None and prev_index is not None:
            # TODO: include short pages before/after first and last search pages when generating alignment
            # (assume matches alignment of pages we are able to match)
            # on the last iteration, extend the chunk to the end of pages so the
            # final anchor and any pages after it are included in the mapping
            if search_i == last_search_i:
                page_chunk_df = orig_pages_df.slice(prev_index)
            else:
                page_chunk_df = orig_pages_df.slice(
                    prev_index, page["index"] - prev_index
                )
            chunk_p1 = page_chunk_df.row(0, named=True)
            chunk_p2 = page_chunk_df.row(page_chunk_df.height - 1, named=True)
            # if shift amount matches, we have an alignment;
            # generate page mappings for the chunk between this search text and the previous
            if shift == prev_shift:
                print(
                    f"found alignment shift={shift} for pages {chunk_p1['order']} (i{chunk_p1['index']}) to {chunk_p2['order']} (i{chunk_p2['index']})"
                )
                # create an order field adjusted by the required shift, so we can join
                zip_shift_df = zip_pages_df.with_columns(
                    aligned_order=pl.col("order") + prev_shift
                )
                # join the chunk of pages with the zip pages based on the aligned order
                chunk_mapping_df = page_chunk_df.join(
                    zip_shift_df,
                    left_on="order",
                    right_on="aligned_order",
                    how="left",
                ).select(["id", "page_filename"])
            else:
                # if shifts don't match, recurse on this chunk of pages
                print(
                    f"### recursing for pages {chunk_p1['order']} (i{chunk_p1['index']}) to {chunk_p2['order']} (i{chunk_p2['index']})"
                )
                # drop row index for current loop so it can be added for the smaller chunk
                # TODO: limit zip_pages to pages *after* any previously found alignments
                chunk_mapping_df = align_shifted_pages(
                    page_chunk_df.drop("index"), zip_pages_df
                )

            # add mapping for each set of pages to the aggregate mapping df
            if chunk_mapping_df is not None and not chunk_mapping_df.is_empty():
                page_mapping_df = page_mapping_df.vstack(chunk_mapping_df)

        # update previous values for next loop
        prev_shift = shift
        prev_index = page["index"]

    return page_mapping_df


# determine alignment between pages in different versions of hathitrust
def align_pages(work_id: str, pages_df: pl.DataFrame, zipfile: ZipFile):  #  -> dict:
    expected_page_count = pages_df.height
    # load text files from zipfile into a polars dataframe
    zip_pages_df = (
        pl.DataFrame(
            data=get_zip_textfiles(zipfile),
            schema=["page_filename", "text"],
        )
        .with_columns(
            # extract the numeric page id for joining with page data
            # some works have filenames like OSU_32435051461309_00000602 ; others are simply numeric
            page_id=pl.col.page_filename.str.extract(r"_?([0-9]+$)", 1)
        )
        .with_columns(
            # make an order field to match page id so we can calculate size of shift
            order=pl.col.page_id.cast(pl.Int64),
            text_len=pl.col.text.str.len_chars(),
        )
    )

    # NOTE: for excerpt, page count is not expected to match but should be >= total
    if zip_pages_df.height < expected_page_count:
        print(
            f"Warning: page count mismatcH; pages in zipfiles ({zip_pages_df.height}, expected at least {expected_page_count})"
        )
    # extract bare page id from work_id.page_id globally unique page identifier
    pages_join_df = (
        pages_df.with_columns(page_id=pl.col.id.str.extract(r"[._]([0-9]+$)", 1))
        .join(zip_pages_df, on="page_id")
        .with_columns(text_match=pds.str_fuzz("text", "text_right", parallel=True))
    )
    if expected_page_count != pages_join_df.height:
        # TODO: don't repeat if we already warned about zip total page count
        print(
            f"Warning: joined pages ({pages_join_df.height}) does not match expected page count ({expected_page_count})"
        )

    # maybe filter out pages with no text when checking score? (probably omits nulls anyway...)

    # for now, just report the average score
    avg = pages_join_df["text_match"].mean()
    tqdm.write(
        f"{work_id: <30} {pages_df.height:> 4,} pages; average indel similarity score: {avg:.3f}"
    )
    # might be lower than this; at least one 0.87 is visibly correct alignment
    if avg is not None and avg > 0.87:
        page_mapping_df = pages_join_df
    else:
        page_mapping_df = align_shifted_pages(pages_df, zip_pages_df)
        if page_mapping_df is None or page_mapping_df.is_empty():
            return

    # construct and return a dictionary mapping original page id to corresponding filename in the zipfile
    return {
        r["id"]: r["page_filename"]
        for r in page_mapping_df.select(["id", "page_filename"]).iter_rows(named=True)
    }


def process_work(
    work_id: str, pages: list[dict], image_dir: Path, tar: tarfile.TarFile
) -> Iterator[dict]:
    # generic process work method, which calls appropriate source-specific method
    match get_ppa_source(work_id):
        case "Gale":
            pass  # skip for debug/test
            # yield from process_gale_work(work_id, pages, image_dir, tar)
        case "HathiTrust":
            yield from process_ht_work(work_id, pages, image_dir, tar)
        case "ECCO":
            yield from pages  # no images


def process_gale_work(
    work_id: str, pages: list[dict], image_dir: Path, tar: tarfile.TarFile
) -> Iterator[dict]:
    vol_img_dir = image_dir / get_vol_dir(get_volume_id(work_id))
    if vol_img_dir.is_dir():
        # print(f"{work_id} : {vol_img_dir} : {len(pages)} pages")
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
    # zip file is named based on id without institution prefix
    # must be encoded to convert ark style ids to file safe format
    htid_suffix = encode_htid(htid).split(".")[-1]
    zipfile_path = image_dir / encode_htid(htid) / f"{htid_suffix}.zip"
    if not zipfile_path.exists():
        # TODO: add a quiet mode or switch to logging to simplify running without all data present
        # print(
        # f"Warning: zipfile {zipfile_path} does not exist, omitting images"
        # )
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
                for page in pages:
                    page_id = page["id"]  # .split(".")[-1]
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
        output_pages_path.rename(old_output_pages)
        print(
            f"Warning: output file {output_pages_path} exists; renamed to {old_output_pages}"
        )
    if output_archive_path.exists():
        print(f"Warning: output file {output_archive_path} already exists, overwriting")

    # use a polars lazy frame to calculate the total so tqdm can estimate completion
    start_time = perf_counter()
    total_pages = pl.scan_ndjson(args.input).select(pl.len()).collect().item()
    end_time = perf_counter()
    print(f"{total_pages:,} total pages (calculated in {end_time - start_time:0.2f}s)")
    # configure tqdm to format as comma delimited numbers - from https://stackoverflow.com/a/76964589
    tqdm.format_sizeof = lambda x, divisor=None: (f"{x:,}" if divisor else f"{x:5.2f}")
    # Stream pages one at a time; corpus is sorted by work+page so we can
    # process pages by work as the work_id changes.
    with tarfile.open(output_archive_path, "w:gz") as tar:
        prev_work_id: Optional[str] = None
        pages: list[dict] = []
        for page in tqdm(
            orjsonl.stream(args.input),
            desc="Reading pages",
            total=total_pages,
            unit_scale=True,
        ):
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
