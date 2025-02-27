#!/usr/bin/env python
"""

🎶🍵 matcha matcha poem / This script is gon / na find your poems / matcha matcha poem 🎶🍵

refmatcha is a script to identify poem excerpts by matching against a local
collection of reference poems.  It takes in a CSV of excerpts
and outputs a CSV of labeled excerpts for those excerpts it is able to identify.

Setup:

Download and extract poetry-ref-data.tar.bz2 from /tigerdata/cdh/prosody/poetry-detection
You should extract it in the same directory where you plan to run this script.
The script will compile reference content into full-text and metadata parquet files;
to recompile, rename or remove the parquet files.

"""

import argparse
import codecs
import csv
import logging
import pathlib
from glob import iglob

try:
    from itertools import batched
except ImportError:
    from more_itertools import batched

from time import perf_counter

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import rapidfuzz
from unidecode import unidecode

from corppa.poetry_detection.core import Excerpt, LabeledExcerpt

logger = logging.getLogger(__name__)

# for convenience, assume the poetry reference data directory is
# available relative to wherever this script is called from
REF_DATA_DIR = pathlib.Path("poetry-reference-data")
TEXT_PARQUET_FILE = REF_DATA_DIR / "poems.parquet"
META_PARQUET_FILE = REF_DATA_DIR / "poem_metadata.parquet"
# csv files to supplement .txt files
POETRY_FOUNDATION_CSV = REF_DATA_DIR / "poetryfoundationdataset.csv"
CHADWYCK_HEALEY_CSV = REF_DATA_DIR / "chadwyck_healey_metadata.csv"
# define source ids to ensure we are consistent
SOURCE_ID = {
    "Poetry Foundation": "poetry-foundation",
    "Chadwyck-Healey": "chadwyck-healey",
}


def compile_text(data_dir, output_file):
    """Compile reference poems into a parquet file for quick identification
    of poetry excerpts based on matching text. Looks for text files in
    directories under `data_dir`; uses the filename stem as poem identifier
    and the containing directory name as the id for the source reference corpus.
    Also looks for and includes content from `poetryfoundationdataset.csv`
    contained in the data directory.
    """

    # parquet file schema:
    # - poem id
    # - text of the poem
    # - source (identifier for the reference corpus)
    schema = pa.schema(
        [("id", pa.string()), ("text", pa.string()), ("source", pa.string())]
    )
    # open a parquet writer so we can add records in chunks
    pqwriter = pq.ParquetWriter(output_file, schema)

    # handle files in batches
    # look for .txt files in nested directories; use parent directory name as
    # the reference corpus source name/id
    for chunk in batched(iglob(f"{data_dir}/**/*.txt"), 1000):
        chunk_files = [pathlib.Path(f) for f in chunk]
        ids = [f.stem for f in chunk_files]
        sources = [f.parent.name for f in chunk_files]
        texts = [f.open().read() for f in chunk_files]
        # create and write a record batch
        record_batch = pa.RecordBatch.from_arrays(
            [ids, texts, sources], names=["id", "text", "source"]
        )
        pqwriter.write_batch(record_batch)

    # poetry foundation text content is included in the csv file
    if POETRY_FOUNDATION_CSV.exists():
        # load poetry foundation csv into a polars dataframe
        # - rename columns for our use
        # - add source column
        # - select only the columns we want to include
        pf_df = (
            pl.read_csv(POETRY_FOUNDATION_CSV)
            .rename({"Poetry Foundation ID": "id", "Content": "text"})
            .with_columns(source=pl.lit(SOURCE_ID["Poetry Foundation"]))
            .select(["id", "text", "source"])
        )
        # convert polars dataframe to arrow table, cast to our schema to
        # align types (large string vs string), then write out in batches
        for batch in pf_df.to_arrow().cast(target_schema=schema).to_batches():
            pqwriter.write_batch(batch)
    else:
        print(
            f"Poetry Foundation csv file not found for text compilation (expected at {POETRY_FOUNDATION_CSV})"
        )

    # close the parquet file
    pqwriter.close()


def compile_metadata(data_dir, output_file):
    # for poem dataset output, we need poem id, author, and title
    # to match text results, we need poem id and source id

    schema = pa.schema(
        [
            ("id", pa.string()),
            ("source", pa.string()),
            ("author", pa.string()),
            ("title", pa.string()),
        ]
    )
    # open a parquet writer for outputting content in batches
    pqwriter = pq.ParquetWriter(output_file, schema)

    # load chadwyck healey metadata
    if CHADWYCK_HEALEY_CSV.exists():
        # use polars to read in the csv and convert to the format we want
        # - rename main title to title
        # - add source id for all rows
        # - combine author first and last name
        # - reorder and limit columns to match parquet schema
        df = (
            # ignore parse errors in fields we don't care about (author_dob)
            pl.read_csv(CHADWYCK_HEALEY_CSV, ignore_errors=True)
            .rename({"title_main": "title"})
            .with_columns(source=pl.lit(SOURCE_ID["Chadwyck-Healey"]))
            .with_columns(
                pl.concat_str(
                    [pl.col("author_fname"), pl.col("author_lname")], separator=" "
                ).alias("author")
            )
            .select(["id", "source", "author", "title"])
        )
        # convert polars dataframe to arrow table, cast to our schema to
        # align types (large string vs string), then write out in batches
        for batch in df.to_arrow().cast(target_schema=schema).to_batches():
            pqwriter.write_batch(batch)
    else:
        print(
            f"Chadwyck-Healey csv file not found for metadata compilation (expected at {CHADWYCK_HEALEY_CSV})"
        )

    # for the directory of internet poems, metadata is embedded in file name
    internet_poems_dir = data_dir / "internet-poems"
    # this directory is a set of manually curated texts;
    # currently only 112 files, so don't worry about chunking until needed
    poem_files = list(internet_poems_dir.glob("*.txt"))
    # use filename without .txt as poem identifier
    ids = [p.stem for p in poem_files]
    # filename is : Firstname-Lastname_Poem-Title.txt
    # author name: filename before the _ with dashes replaced with spaces
    authors = [p.stem.split("_", 1)[0].replace("-", " ") for p in poem_files]
    # title: same as author for the text after the _
    titles = [p.stem.split("_", 1)[1].replace("-", " ") for p in poem_files]
    source = ["internet-poems"] * len(ids)

    # create a record batch to write out
    record_batch = pa.RecordBatch.from_arrays(
        [ids, source, authors, titles], names=["id", "source", "author", "title"]
    )
    pqwriter.write_batch(record_batch)

    # load poetry foundation data from csv file
    # do this one last since it is least preferred of our sources
    if POETRY_FOUNDATION_CSV.exists():
        # use polars to read in the csv and convert to the format we want
        # - rename columns to match desired output
        # - add source id
        # - reorder and limit columns to match parquet schema
        df = (
            pl.read_csv(POETRY_FOUNDATION_CSV)
            # .drop("Content", "")
            .rename(
                {"Author": "author", "Title": "title", "Poetry Foundation ID": "id"}
            )
            .with_columns(source=pl.lit(SOURCE_ID["Poetry Foundation"]))
            .select(["id", "source", "author", "title"])
        )
        # convert polars dataframe to arrow table, cast to our schema to
        # align types (large string vs string), then write out in batches
        for batch in df.to_arrow().cast(target_schema=schema).to_batches():
            pqwriter.write_batch(batch)
    else:
        print(
            f"Poetry Foundation csv file not found for metadata compilation (expected at {POETRY_FOUNDATION_CSV})"
        )

    # close the parquet file
    pqwriter.close()


def _text_for_search(expr):
    """Takes a polars expression (e.g. column or literal value) and applies
    text pattern replacements to clean up to make it easier to find matches."""
    return (
        # remove specific punctuation marks in the middle of words
        expr.str.replace_all(r"(\w)[-'](\w)", "$1$2")
        # metrical notation that splits words (e.g. sudden | -ly or visit | -or)
        .str.replace_all(r"(\w) \| -(\w)", "$1$2")
        # replace other puncutation with spaces
        .str.replace_all("[[:punct:]]", " ")
        # remove indent entity in CH (probably more of these...)
        .str.replace_all("&indent;", " ")
        # normalize whitespace
        # NOTE: if we can do this on the query term instead,
        # matching reference text in the output will be more readable
        .str.replace_all("[\t\n\v\f\r ]+", " ")  # could also use [[:space:]]
        # replace curly quotes with straight (both single and double)
        .str.replace_all("[”“]", '"')
        .str.replace_all("[‘’]", "'")
        .str.replace_all("ſ", "s")  # handle long s (also handled by unidecode)
        .str.strip_chars()
    )


def searchable_text(text):
    """Convert a text string into a searchable string, using the same rules
    applied to search text in the reference dataframe, with additional unicode decoding.
    """
    return unidecode(pl.select(_text_for_search(pl.lit(text))).item())


def generate_search_text(df, field="text", output_field=None):
    """Takes a Polars dataframe and generates a searchable text field
    based on the input column (by default, input column is "text" and
    output is "search_text").  Removes punctuation, normalizes whitespace,
    and strips trailing and leading whitespace, etc."""
    if output_field is None:
        output_field = f"search_{field}"
    return df.with_columns(**{output_field: _text_for_search(pl.col(field))})


def multiple_matches(df, search_field):
    # when a result has multiple matches, see if we can determine if
    # it is the same poem in different sources
    match_count = int(df.height)

    #  check if both author and title match (ignoring punctuation and case)
    # TODO: could use rapidfuzz here to check author & title are sufficiently similar
    # e.g. these should be treated as matches but are not currently:
    #    Walter Scott      ┆ Coronach
    #    Walter, Sir Scott ┆ CCLXXVIII CORONACH
    df = df.with_columns(
        _author=pl.col("author").str.replace_all("[[:punct:]]", "").str.to_lowercase(),
        _title=pl.col("title").str.replace_all("[[:punct:]]", "").str.to_lowercase(),
    )

    dupe_df = df.filter(df.select(["_author", "_title"]).is_duplicated())

    match_poem = None
    if not dupe_df.is_empty():
        # if all rows match, return the first one
        if int(dupe_df.height) == match_count:
            match_poem = dupe_df.to_dicts()[0]
            match_poem["notes"] = (
                f"multiple matches on {search_field}, all rows match author + title"
            )
        # if duplicate rows are a majority, return the first one
        elif dupe_df.height >= match_count / 2:
            # TODO: include alternates in notes?
            # these majority matches may be less confident
            match_poem = dupe_df.to_dicts()[0]
            match_poem["notes"] = (
                f"multiple matches on {search_field}, majority match author + title ({dupe_df.height} out of {match_count})"
            )

    if match_poem is not None:
        return match_poem

    # if author/title duplication check failed, check for author matches
    # poetry foundation includes shakespeare drama excerpts with alternate names
    authordupe_df = df.filter(df.select(["_author"]).is_duplicated())
    if not authordupe_df.is_empty():
        # shakespeare shows up oddly in poetry foundation;
        # if author matchnes assume the other source has the correct title
        non_poetryfoundtn = authordupe_df.filter(
            pl.col("source") != SOURCE_ID["Poetry Foundation"]
        )
        if non_poetryfoundtn.height == 1:
            match_poem = non_poetryfoundtn.to_dicts()[0]
            match_poem["notes"] = (
                "multiple matches, duplicate author but not title; excluding Poetry Foundation"
            )
            return match_poem


def fuzzy_partial_ratio(series, search_text):
    """Calculate rapidfuzz partial_ratio score for a single input
    search text across a whole series of potentially matching texts.
    Returns a list of scores."""
    scores = rapidfuzz.process.cdist(
        [search_text],
        series,
        scorer=rapidfuzz.fuzz.partial_ratio,
        score_cutoff=90,
        workers=-1,
    )
    # generates a list of scores for each input search text, but we only have
    # one input string, so return the first list of scores
    return scores[0]


def find_reference_poem(input_row, ref_df, meta_df):
    result = {"poem_id"}
    previously_searched = []
    for search_field in ["search_text", "search_first_line", "search_last_line"]:
        search_field_label = search_field.replace("search_", "").replace("_", " ")
        # use unidecode to drop accents (often used to indicate meter)
        search_text = unidecode(input_row[search_field])
        # skip if the exact same content has already been searched(e.g. for single-line excerpts)
        # and don't search on empty strings
        if search_text in previously_searched or search_text.strip() == "":
            continue
        previously_searched.append(search_text)
        logger.info(f"Searching on {search_field_label}: {search_text}")

        try:
            # do a case-insensitive search
            re_search = f"(?i){search_text}"
            re_extract = f"(?i)^(?<preceding_text>.*)(?<ref_span_text>{search_text})"
            result = (
                # filter to rows that match the regex search
                ref_df.filter(pl.col("search_text").str.contains(re_search))
                # for those that match, extract the search text AND
                # all of the text that comes before it
                .with_columns(
                    captures=pl.col("search_text").str.extract_groups(re_extract)
                )
                .unnest("captures")
                # calculate start based on character length of text before the match
                # NOTE: can't use str.find because it returns byte offset instead of char offset
                .with_columns(ref_span_start=pl.col("preceding_text").str.len_chars())
                # calculate span end based on span start and length of matching text
                .with_columns(
                    ref_span_end=pl.col("ref_span_start").add(
                        pl.col("ref_span_text").str.len_chars()
                    )
                )
                # drop preceding text, since it may be quite large
                .drop("preceding_text")
            )

        except pl.exceptions.ComputeError as err:
            print(f"Error searching: {err}")
            continue
        # if no matches, try the next search field
        if result.is_empty():
            continue

        # if the match was NOT found based on the full text,
        # adjust reference start/end/text for the full span
        if search_field != "search_text":
            # use search text length as basis for reference span length
            search_text_length = len(input_row["search_text"])

            if search_field == "search_first_line":
                # if matched on first line, start is correct;
                # recalculate end based on the length of the input search text
                result = result.with_columns(
                    ref_span_end=pl.col("ref_span_start").add(search_text_length)
                )
            # and use slice to get the substring for that content
            elif search_field == "search_last_line":
                # if matched on last line, end is correct; adjust start
                result = result.with_columns(
                    ref_span_start=pl.col("ref_span_end").add(-search_text_length)
                ).with_columns(
                    # dont' allow span start to be smaller than zero
                    ref_span_start=pl.when(pl.col("ref_span_start") < 0)
                    .then(0)
                    .otherwise(pl.col("ref_span_start"))
                )

            # then extract full reference text for adjusted indices
            result = result.with_columns(
                ref_span_text=pl.col("search_text").str.slice(
                    pl.col("ref_span_start"), search_text_length
                )
            )

        # otherwise, see if the results are useful
        num_matches = result.height  # height = number of rows
        # join matching poem with metadata dataframe
        # so we can include poem title and author in the output
        result = result.join(
            meta_df,
            # join on the combination of poem id and source id
            on=pl.concat_str([pl.col("id"), pl.col("source")], separator="|"),
            how="left",  # occasionally ids do not match,
            # e.g. Chadwyck Healey poem id we have text for but not in metadata
        )
        result = result.drop("text", "id_right", "source_right", "search_text")

        # if we get a single match, assume it is authoritative
        if num_matches == 1:
            # match poem includes id, author, title
            match_poem = result.to_dicts()[0]
            # add note about how the match was determined
            match_poem["notes"] = f"single match on {search_field_label}"
            # include number of matches found
            match_poem["match_count"] = num_matches

            return match_poem
        elif num_matches < 10:
            # if there's a small number of matches, check for duplicates
            match_poem = multiple_matches(result, search_field_label)
            # return match if a good enough result was found
            if match_poem:
                match_poem["match_count"] = num_matches
                return match_poem

    # skip fuzzy match for now, while revising output
    # TODO: add argparse flag to control whether we try fuzzy matching
    return

    # NOTE: could make configurable to skip fuzzy searches,
    # or maybe truncate large excepts before running fuzzy search

    # if no matches were found yet, try a fuzzy search on the full text
    search_text = unidecode(input_row["search_text"])
    # NOTE: might want some minimum length or uniqueness on the search text
    logger.info(f"Trying fuzzy match on: {search_text}")
    start_time = perf_counter()
    result = ref_df.with_columns(
        score=pl.col("search_text").map_batches(
            lambda x: fuzzy_partial_ratio(x, search_text)
        )
    ).filter(pl.col("score").ne(0))
    end_time = perf_counter()
    logger.debug(f"Calculated rapidfuzz partial_ratio in {end_time - start_time:0.2f}s")

    result = result.join(
        meta_df,
        # join on the combination of poem id and source id
        on=pl.concat_str([pl.col("id"), pl.col("source")], separator="|"),
        how="left",
    )
    result = result.drop("text", "id_right", "source_right", "search_text")
    num_matches = result.height

    result = result.sort(by="score", descending=True)
    if not result.is_empty():
        # when we only get a single match, results look pretty good
        if num_matches == 1:
            # match poem includes id, author, title
            match_poem = result.to_dicts()[0]
            # add note about how the match was determined
            match_poem["notes"] = f"fuzzy match; score: {result['score'].item():.1f}"
            # include number of matches found
            match_poem["num_matches"] = num_matches
            return match_poem
        elif num_matches <= 3:
            # if there's a small number of matches, check for duplicates
            match_poem = multiple_matches(result, "full text (fuzzy)")
            # return match if a good enough result was found
            if match_poem:
                match_poem["match_count"] = num_matches
                match_poem["notes"] += (
                    f"\nfuzzy match; score: {match_poem['score']:.1f}"
                )
                return match_poem
        else:
            # sometimes we get many results with 100 scores;
            # likely an indication that the search text is short and too common
            # filter to all results with the max score and check for a majority
            top_matches = result.filter(pl.col("score").eq(pl.col("score").max()))
            match_poem = multiple_matches(top_matches, "full text (fuzzy)")
            # return match if a good enough result was found
            if match_poem:
                match_poem["match_count"] = num_matches
                match_poem["notes"] += f"\nfuzzy match, score {match_poem['score']}"
                return match_poem

    # no good match found
    return None


def process(input_file):
    logging.basicConfig(encoding="utf-8", level=logging.DEBUG)
    # if the parquet files aren't present, generate them
    # (could add an option to recompile in future)
    if not TEXT_PARQUET_FILE.exists():
        print(f"Compiling reference poem text to {TEXT_PARQUET_FILE}")
        compile_text(REF_DATA_DIR, TEXT_PARQUET_FILE)
    if not META_PARQUET_FILE.exists():
        print(f"Compiling reference poem metadata to {META_PARQUET_FILE}")
        compile_metadata(REF_DATA_DIR, META_PARQUET_FILE)

    # load for searching
    df = pl.read_parquet(TEXT_PARQUET_FILE)
    meta_df = pl.read_parquet(META_PARQUET_FILE)
    print(f"Poetry reference text data: {df.height:,} entries")
    # some texts from poetry foundation and maybe Chadwyck-Healey are truncated
    # discard them to avoid bad partial/fuzzy matches
    df = df.with_columns(text_length=pl.col("text").str.len_chars())
    min_length = 30
    short_texts = df.filter(pl.col("text_length").lt(min_length))
    df = df.filter(pl.col("text_length").ge(min_length))
    print(f"  Omitting {short_texts.height} poems with text length < {min_length}")

    print(f"Poetry reference metadata:  {meta_df.height:,} entries")

    # generate a simplified text field for searching
    df = generate_search_text(df)

    # load csv with excerpt fieldnames
    try:
        input_df = pl.read_csv(input_file, columns=Excerpt.fieldnames())
    except pl.exceptions.ColumnNotFoundError as err:
        # if any excerpt fields are missing, report and exit
        print(f"Input file does not have expected excerpt fields: {err}")
        raise SystemExit(1)

    print(f"Input file has {input_df.height:,} excerpts")
    input_columns = input_df.columns  # store original columns for output

    # convert input text to search text using the same rules applied to reference df
    input_df = generate_search_text(
        input_df, field="ppa_span_text", output_field="search_text"
    )
    # split out text to isolate first and last linesfind based on
    input_df = input_df.with_columns(
        first_line=pl.col("ppa_span_text")
        .str.strip_chars()
        .str.split("\n")
        .list.first(),
        last_line=pl.col("ppa_span_text").str.strip_chars().str.split("\n").list.last(),
    )
    # generate searchable versions of first and last lines
    input_df = generate_search_text(input_df, "first_line")
    input_df = generate_search_text(input_df, "last_line")

    # number of matches found
    match_found = 0

    # NOTE: this is slow... there is almost certainly a more polars-ish way to
    # approach it
    # probably best option is to use map_batches
    # needs to return poem id, corpus id, span start/end, text
    # - can we do in waves, simplest first?
    #       figure out: full text match; how to get span in original from search text
    #    - if a match is found, it's removed from the next batch
    # ... how to get start/end/text?

    # notes could include rapidfuzz partial_ratio_alignment score
    # once we have the two sets

    output_file = input_file.with_name(f"{input_file.stem}_matched.csv")

    with output_file.open("w", encoding="utf-8") as outfile:
        # add byte-order-mark to indicate unicode
        outfile.write(codecs.BOM_UTF8.decode())
        # output matched excerpts as labeled excerpts
        csvwriter = csv.DictWriter(
            outfile,
            fieldnames=LabeledExcerpt.fieldnames(),
        )
        csvwriter.writeheader()

        for row in input_df.iter_rows(named=True):
            match_poem = find_reference_poem(row, df, meta_df)

            # if a match was found, generate and output
            # a LabeledExcerpt based on the original Excerpt
            if match_poem:
                # the dataframe row now includes search text fields;
                # filter to just excerpt fields
                excerpt_fields = {
                    k: v for k, v in row.items() if k in Excerpt.fieldnames()
                }
                notes = excerpt_fields.pop("notes") or ""
                notes += f"refmatcha: {match_poem['notes']}"

                # NOTE: if identification logic results in a bad span, this will
                # raise a value error
                excerpt = LabeledExcerpt(
                    # labeled excerpt fields - reference poem information
                    poem_id=match_poem["id"],
                    ref_corpus=match_poem["source"],
                    ref_span_start=match_poem["ref_span_start"],
                    ref_span_end=match_poem["ref_span_end"],
                    ref_span_text=match_poem["ref_span_text"],
                    identification_methods={"refmatcha"},
                    notes=notes,
                    # TODO: use new from_dict method ?
                    **excerpt_fields,
                )
                match_found += 1
                # TODO: use new to_csv method
                csvwriter.writerow(excerpt.to_dict())

    print(f"Poems with match information saved to {output_file}")
    print(
        f"{match_found} excerpts with matches ({match_found / input_df.height * 100:.2f}% of {input_df.height} rows processed)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Attempt to identify poem excerpts by matching against reference set"
    )
    parser.add_argument(
        "input",
        help="csv or tsv file with poem excerpts",
        type=pathlib.Path,
    )
    # TODO: add arg for output file
    args = parser.parse_args()

    process(args.input)


if __name__ == "__main__":
    main()
