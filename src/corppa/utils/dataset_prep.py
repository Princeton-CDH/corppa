# prep ppa text+image dataset for publication
import argparse
import logging
import tarfile
from pathlib import Path
from time import mktime, perf_counter
from typing import Iterator, Optional
from zipfile import ZipFile

import numpy as np
import orjsonl
import polars as pl
import polars_ds as pds
import rapidfuzz
from intspan import intspan
from tqdm import tqdm

from corppa.utils.path_utils import (
    encode_htid,
    get_ppa_source,
    get_vol_dir,
    get_volume_id,
)

logger = logging.getLogger(__name__)


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


# minimum text length (in characters) for a page to be trusted as match
# evidence when determining the page shift; shorter pages match unreliably
MIN_MATCH_TEXT_LEN = 600
# minimum fuzzy ratio (0-100) for a page match to be considered at all
MATCH_SCORE_CUTOFF = 85
# a page's best zip match must beat its runner-up by at least this many ratio
# points to be trusted; guards against near-ties from repeated boilerplate pages
MATCH_SCORE_MARGIN = 3
# a best match at or above this ratio is treated as an unambiguous match and
# trusted regardless of the runner-up margin (near-exact text match)
MATCH_SCORE_STRONG = 99


def align_shifted_pages(pages_df: pl.DataFrame, zip_pages_df: pl.DataFrame):
    """Align corpus pages to zip page filenames when page order has shifted
    between versions. Shifts are determined only from long (reliable) pages via
    a single vectorized fuzzy comparison; short pages inherit the shift of the
    surrounding long pages. Contiguous runs of equal shift are handled
    independently so works with mid-document insertions/removals still align.

    Returns a DataFrame with ``id`` and ``page_filename`` columns (possibly
    empty if no reliable shift can be determined)."""
    # empty mapping frame, used as the early-return value when no long pages
    # clear the filter or no reliable shift can be determined
    empty_mapping_df = pl.DataFrame(
        schema={"id": pl.String, "page_filename": pl.String}
    )

    # sort by order and keep the full (unfiltered) set; short pages still need
    # to be mapped even though they don't contribute match evidence
    orig_pages_df = pages_df.sort("order", descending=False)

    # long pages only: these are the pages we trust to determine the shift
    long_pages_df = orig_pages_df.with_columns(
        text_len=pl.col.text.str.len_chars()
    ).filter(pl.col.text_len.gt(MIN_MATCH_TEXT_LEN))
    if long_pages_df.height == 0:
        # no long-enough pages at all - can't determine a shift, so give up
        logger.warning(
            "No pages over %d characters; cannot determine page shift",
            MIN_MATCH_TEXT_LEN,
        )
        return empty_mapping_df

    # single vectorized fuzzy comparison of every long page against every zip
    # page; scores is an (n_long, n_zip) matrix, 0 where below the cutoff
    scores = rapidfuzz.process.cdist(
        long_pages_df["text"].to_list(),
        zip_pages_df["text"].to_list(),
        scorer=rapidfuzz.fuzz.ratio,
        workers=-1,
        score_cutoff=MATCH_SCORE_CUTOFF,
    )

    zip_orders = zip_pages_df["order"].to_numpy()
    long_orders = long_pages_df["order"].to_numpy()

    # best match per long page, plus the runner-up for the ambiguity guard
    # - get index of best match for each long page
    best_idx = scores.argmax(axis=1)
    row_idx = np.arange(scores.shape[0])
    # - get the best score for each long page
    best_score = scores[row_idx, best_idx]
    if scores.shape[1] >= 2:
        # second-highest score in each row (np.partition puts the 2nd-largest
        # at position -2); used to reject near-ties
        second_score = np.partition(scores, -2, axis=1)[:, -2]
    else:
        # only one zip page: no runner-up to compare against
        second_score = np.zeros_like(best_score)

    # a long page's shift is trusted when it has a real match (score > 0, i.e.
    # above the cutoff) that is either near-exact or clearly beats its runner-up
    confident = (best_score > 0) & (
        (best_score >= MATCH_SCORE_STRONG)
        | ((best_score - second_score) >= MATCH_SCORE_MARGIN)
    )
    shifts = long_orders - zip_orders[best_idx]
    trusted_shift = np.where(confident, shifts, np.nan)

    long_shift_df = long_pages_df.select("order").with_columns(
        # NaN marks unconfident pages; convert to null so fill logic treats
        # them as gaps to be filled from neighbors
        shift=pl.Series(trusted_shift).fill_nan(None)
    )
    if long_shift_df["shift"].drop_nulls().is_empty():
        # no long page produced a confident, unambiguous match
        return empty_mapping_df

    # attach the trusted long-page shifts back onto the full page set; short
    # pages (and unconfident long pages) start with a null shift
    pages_shift_df = orig_pages_df.join(long_shift_df, on="order", how="left")

    # fill short/unconfident pages from their neighbors: forward-fill assigns
    # each gap the preceding long page's shift; backward-fill covers any
    # leading pages before the first long page. At a run boundary a short page
    # inherits the preceding run's shift (forward_fill wins).
    pages_shift_df = pages_shift_df.with_columns(
        seg_shift=pl.col.shift.forward_fill().backward_fill()
    )

    # split into contiguous runs of equal shift; each run aligns independently
    # so a mid-document insertion/removal doesn't corrupt the rest of the work
    pages_shift_df = pages_shift_df.with_columns(
        seg_id=(pl.col.seg_shift != pl.col.seg_shift.shift(1)).cum_sum()
    )

    seg_summary_df = pages_shift_df.group_by("seg_id", maintain_order=True).agg(
        shift=pl.col.seg_shift.first(),
        first_order=pl.col.order.first(),
        last_order=pl.col.order.last(),
        n_pages=pl.len(),
    )

    # summarize the page shift(s) applied to align this work at info level.
    # intspan consolidates each shift's orders into compact, contiguous page
    # ranges (e.g. 1-510), so segments that split for other reasons merge.
    shift_orders = (
        pages_shift_df.group_by("seg_shift")
        .agg(orders=pl.col.order, n_pages=pl.len())
        .sort("n_pages", descending=True)
    )
    shift_summary = ", ".join(
        f"{int(row['seg_shift'])} ({row['n_pages']:,} pages, "
        f"pp. {intspan(row['orders'])})"
        for row in shift_orders.iter_rows(named=True)
    )
    logger.info("page shift: %s", shift_summary)

    if logger.isEnabledFor(logging.DEBUG):
        for seg in seg_summary_df.iter_rows(named=True):
            logger.debug(
                "alignment shift=%s for orders %s-%s (%s pages)",
                seg["shift"],
                seg["first_order"],
                seg["last_order"],
                seg["n_pages"],
            )

    # single join for the whole work: shift each page's order and look up the
    # zip page filename at the aligned order
    page_mapping_df = (
        pages_shift_df.with_columns(
            aligned_order=(pl.col.order - pl.col.seg_shift).cast(pl.Int64)
        )
        .join(
            zip_pages_df.select(["order", "page_filename"]),
            left_on="aligned_order",
            right_on="order",
            how="left",
        )
        .select(["id", "page_filename"])
    )
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
    zip_count_mismatch = zip_pages_df.height < expected_page_count
    if zip_count_mismatch:
        logger.warning(
            "%s page count mismatch; pages in zipfiles (%d, expected at least %d)",
            work_id,
            zip_pages_df.height,
            expected_page_count,
        )
    # extract bare page id from work_id.page_id globally unique page identifier
    pages_join_df = (
        pages_df.with_columns(page_id=pl.col.id.str.extract(r"[._]([0-9]+$)", 1))
        .join(zip_pages_df, on="page_id")
        .with_columns(text_match=pds.str_fuzz("text", "text_right", parallel=True))
    )
    # only warn about the joined page count if we didn't already warn about
    # the zip page count above, to avoid a redundant warning
    if not zip_count_mismatch and expected_page_count != pages_join_df.height:
        logger.warning(
            "%s joined pages (%d) does not match expected page count (%d)",
            work_id,
            pages_join_df.height,
            expected_page_count,
        )

    # maybe filter out pages with no text when checking score? (probably omits nulls anyway...)

    # for now, just report the average score
    avg = pages_join_df["text_match"].mean()
    logger.info(
        f"{work_id: <30} {pages_df.height:> 4,} pages; average indel similarity score: {avg:.3f}"
    )
    # might be lower than this; at least one 0.87 is visibly correct alignment
    if avg is not None and avg > 0.87:
        page_mapping_df = pages_join_df
    else:
        page_mapping_df = align_shifted_pages(pages_df, zip_pages_df)
        if page_mapping_df.is_empty():
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
    zipfile_path = image_dir / "HathiTrust" / encode_htid(htid) / f"{htid_suffix}.zip"
    if not zipfile_path.exists():
        logger.warning("zipfile %s does not exist, omitting images", zipfile_path)
        # yield pages without image paths
        yield from pages
    else:
        with ZipFile(zipfile_path) as ht_zip:
            page_mapping = align_pages(work_id, pl.DataFrame(pages), ht_zip)
            if not page_mapping:
                logger.warning(
                    "no page mapping found for work %s, omitting images",
                    work_id,
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
                                logger.warning(
                                    "image %s not found in zipfile but page has text; skipping",
                                    zip_image_path,
                                )
                            logger.debug(
                                "matching filenames: %s",
                                [f for f in file_namelist if page_basename in f],
                            )

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
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Continue a previous run: append to existing output files and "
        "skip works already present in the output JSONL",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        type=str.lower,
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging verbosity (default: info); case-insensitive",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Write log output to this file instead of stderr",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(levelname)s: %(message)s",
        filename=args.log_file,
    )

    if not args.output_dir.is_dir():
        args.output_dir.mkdir(parents=True, exist_ok=True)
    output_pages_path = (
        args.output_dir / "ppa_pages.jsonl"
    )  # .gz # disable compression for now, for testing
    # uncompressed tar so it can be reopened in append mode when continuing
    output_archive_path = args.output_dir / "ppa_images.tar"

    # set of work ids already present in the output; populated when continuing
    completed_work_ids: set[str] = set()

    if args.continue_run:
        # append to existing output; collect work ids already written so we can
        # skip them, rather than renaming/overwriting the existing files
        if output_pages_path.exists():
            completed_work_ids = {
                page["work_id"] for page in orjsonl.stream(output_pages_path)
            }
            logger.info(
                "continuing run: %s works already in %s",
                f"{len(completed_work_ids):,}",
                output_pages_path,
            )
        else:
            logger.warning(
                "--continue set but output file %s does not exist; starting fresh",
                output_pages_path,
            )
        # append to the tar if it exists, otherwise create it
        tar_mode = "a" if output_archive_path.exists() else "w"
    else:
        tar_mode = "w"
        if output_pages_path.exists():
            # because we extend, we need to rename any existing output file
            old_output_pages = output_pages_path.with_suffix(".jsonl.bak")
            output_pages_path.rename(old_output_pages)
            logger.warning(
                "output file %s exists; renamed to %s",
                output_pages_path,
                old_output_pages,
            )
        if output_archive_path.exists():
            logger.warning(
                "output file %s already exists, overwriting", output_archive_path
            )

    # use a polars lazy frame to calculate the total so tqdm can estimate completion
    start_time = perf_counter()
    total_pages = pl.scan_ndjson(args.input).select(pl.len()).collect().item()
    end_time = perf_counter()
    logger.info(
        "%s total pages (calculated in %0.2fs)",
        f"{total_pages:,}",
        end_time - start_time,
    )
    # configure tqdm to format as comma delimited numbers - from https://stackoverflow.com/a/76964589
    tqdm.format_sizeof = lambda x, divisor=None: f"{x:,}" if divisor else f"{x:5.2f}"
    # Stream pages one at a time; corpus is sorted by work+page so we can
    # process pages by work as the work_id changes.
    with tarfile.open(output_archive_path, tar_mode) as tar:
        prev_work_id: Optional[str] = None
        pages: list[dict] = []
        # whether the current work should be skipped (already in output)
        skip_work = False
        for page in tqdm(
            orjsonl.stream(args.input),
            desc="Reading pages",
            total=total_pages,
            unit_scale=True,
        ):
            work_id = page["work_id"]
            # when work id changes, process the previous work pages and reset for the next
            if work_id != prev_work_id:
                if prev_work_id is not None and not skip_work:
                    pages = process_work(prev_work_id, pages, args.image_dir, tar)
                    orjsonl.extend(output_pages_path, list(pages))
                prev_work_id = work_id
                pages = []
                # skip this work if it is already present in the output
                skip_work = work_id in completed_work_ids
            if not skip_work:
                pages.append(page)

        # handle the pages for the last work at end of loop
        if prev_work_id is not None and not skip_work:
            pages = process_work(prev_work_id, pages, args.image_dir, tar)
            orjsonl.extend(output_pages_path, list(pages))


if __name__ == "__main__":
    main()
