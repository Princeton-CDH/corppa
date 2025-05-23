# Copyright (c) 2024-2025, Center for Digital Humanities, Princeton University
# SPDX-License-Identifier: Apache-2.0

import csv
import os
from unittest.mock import patch

import polars as pl
import pytest

from corppa.poetry_detection.core import Excerpt, LabeledExcerpt
from corppa.poetry_detection.refmatcha import (
    CHADWYCK_HEALEY_CSV,
    META_PARQUET_FILE,
    POETRY_FOUNDATION_CSV,
    SCRIPT_ID,
    TEXT_PARQUET_FILE,
    compile_metadata,
    compile_text,
    generate_search_text,
    identify_excerpt,
    main,
    multiple_matches,
    process,
    searchable_text,
)

search_texts = [
    # strip punctuation and remove accents
    ("So many ǎs lōve mě, ănd ūse mě ǎright", "So many as love me and use me aright"),
    # remove newlines, punctuation, omit single - or ' in middle of word
    (
        """the hour when | rites un | -holy
ed each | Paynim | voice to | pray'r""",
        """the hour when rites unholy
ed each Paynim voice to prayr""",
    ),
    # strip outer whitespace; handle long s
    (
        """  Thoſe darts whoſe points make gods adore
His might, and deprecate his power.   """,
        """Those darts whose points make gods adore
His might and deprecate his power""",
    ),
    # metric notation that splits words
    (
        "nearly napping, | sudden | -ly there came a | tapping ... some visit | -or",
        "nearly napping suddenly there came a tapping some visitor",
    ),
    # remove regex characters
    ("* The earth is the Lord's", "The earth is the Lords"),
]


@pytest.mark.parametrize("input,expected", search_texts)
def test_searchable_text(input, expected):
    assert searchable_text(input) == expected


excerpt_earth = Excerpt(
    page_id="nyp.33433069256851.00000599",
    ppa_span_start=491,
    ppa_span_end=531,
    ppa_span_text="The earth is the Lord's, and the fullness thereof. ",
    detection_methods={"adjudication"},
)

ref_poetry_data = [
    {
        "id": "Z200653845",
        "text": """By his wonderful work's we see plainly enough
That the earth is the Lord's and the fullness thereof;
When hungry and thirsty we're ready to faint,
He seeth our need and prevents our complaint;""",
        "source": "chadwyck-healey",
    },
    {
        "id": "King-James-Bible_Psalms",
        "text": "He hath made his wonderful works to be remembered",
        "source": "internet-poems",
    },
]


@pytest.fixture
def reference_df():
    # create a reference dataframe from reference poetry data
    # and generate search text
    return generate_search_text(pl.from_dicts(ref_poetry_data))


def test_generate_search_text():
    df = pl.from_dicts(ref_poetry_data)
    # default text and search field
    searchable_df = generate_search_text(df)
    assert "search_text" in searchable_df.columns

    # specify name for the output field
    named_output_field = "search_me"
    searchable_df = generate_search_text(df, output_field=named_output_field)
    assert named_output_field in searchable_df.columns

    # specify name for the input field
    searchable_df = generate_search_text(
        df.with_columns(pl.col("text").alias("first_line")), "first_line"
    )
    assert "search_first_line" in searchable_df.columns


reference_fields = [
    "poem_id",
    "ref_corpus",
    "ref_span_text",
    "ref_span_start",
    "ref_span_end",
    "identification_methods",
]


def test_identify_excerpt(reference_df):
    # test identify_excerpt method directly
    excerpt_row = excerpt_earth.to_dict()
    # manually set search text for now
    excerpt_row["search_text"] = "The earth is the Lords and the fullness thereof"

    # single match
    id_result = identify_excerpt(excerpt_row, reference_df)
    assert isinstance(id_result, LabeledExcerpt)
    assert id_result.poem_id == ref_poetry_data[0]["id"]
    assert id_result.ref_corpus == ref_poetry_data[0]["source"]
    # these are from the searchable version of the input text
    assert id_result.ref_span_start == 50
    assert id_result.ref_span_end == 97
    assert id_result.ref_span_text == "the earth is the Lords and the fullness thereof"
    assert id_result.notes == "refmatcha: single match on text"
    assert id_result.identification_methods == {SCRIPT_ID}

    # single match, whitespace agnostic
    excerpt_row["search_text"] = (
        "The earth    is the\nLords\t\tand the fullness thereof"
    )
    id_result = identify_excerpt(excerpt_row, reference_df)
    assert id_result.poem_id == ref_poetry_data[0]["id"]

    # no match
    excerpt_row["search_text"] = "Disdain forbids me and my dread of shame"
    assert identify_excerpt(excerpt_row, reference_df) is None


def test_identify_excerpt_special_characters(reference_df, capsys):
    # test identify_excerpt method directly
    excerpt_row = excerpt_earth.to_dict()
    # characters that look like malformed regex cause problems
    excerpt_row["search_text"] = "* The earth is the Lords"
    assert identify_excerpt(excerpt_row, reference_df) is None
    captured = capsys.readouterr()
    assert (
        f"Error searching on {excerpt_earth.page_id} {excerpt_earth.excerpt_id}: regex error"
        in captured.err
    )
    assert "error: repetition operator missing expression" in captured.err


def test_identify_excerpt_first_line(reference_df):
    excerpt_row = excerpt_earth.to_dict()
    # manually set search text for now with multiline content
    excerpt_row["search_text"] = (
        "The earth is the Lords and the fullness thereof\nWhen hungry and thirsty we're ready to faint"
    )
    excerpt_row["search_first_line"] = "The earth is the Lords and the fullness thereof"

    # single match
    id_result = identify_excerpt(excerpt_row, reference_df, "first_line")
    assert isinstance(id_result, LabeledExcerpt)
    assert id_result.poem_id == ref_poetry_data[0]["id"]
    assert id_result.ref_span_start == 50
    # end and text adjusted based on length of search text
    assert id_result.ref_span_end == 142
    assert (
        id_result.ref_span_text
        == "the earth is the Lords and the fullness thereof \nWhen hungry and thirsty were ready to faint"
    )


def test_identify_excerpt_last_line(reference_df):
    excerpt_row = excerpt_earth.to_dict()
    # manually set search text for now with multiline content
    excerpt_row["search_text"] = (
        "The earth is the Lords and the fullness thereof\nWhen hungry and thirsty we're ready to faint"
    )
    excerpt_row["search_last_line"] = "When hungry and thirsty were ready to faint"

    # single match
    id_result = identify_excerpt(excerpt_row, reference_df, "last_line")
    assert id_result.poem_id == ref_poetry_data[0]["id"]
    # start and text adjusted based on length of search text
    assert id_result.ref_span_start == 50
    assert id_result.ref_span_end == 142
    assert (
        id_result.ref_span_text
        == "the earth is the Lords and the fullness thereof \nWhen hungry and thirsty were ready to faint"
    )


@patch("corppa.poetry_detection.refmatcha.multiple_matches")
def test_identify_excerpt_multiple(mock_multimatch, reference_df):
    # test identify_excerpt method when multiple matches are found
    excerpt_row = excerpt_earth.to_dict()
    # manually set search text for now
    # too many matches, can't consolidate
    excerpt_row["search_text"] = "wonderful works"

    # duplicate results don't match
    mock_multimatch.return_value = (None, None)
    assert identify_excerpt(excerpt_row, reference_df) is None

    reason = "all rows match author + title"
    # fill in values for reference span
    ref_match_df = reference_df.with_columns(
        ref_span_text=pl.lit("matched text"),
        ref_span_start=pl.lit(10),
        ref_span_end=pl.lit(20),
    )
    mock_multimatch.return_value = (ref_match_df.limit(1), reason)
    id_result = identify_excerpt(excerpt_row, reference_df)
    assert id_result.poem_id == ref_poetry_data[0]["id"]
    assert id_result.ref_corpus == ref_poetry_data[0]["source"]
    assert id_result.ref_span_text == "matched text"
    assert id_result.ref_span_start == 10
    assert id_result.ref_span_end == 20
    assert (
        id_result.notes == "refmatcha: 2 matches on text: all rows match author + title"
    )


def test_multiple_matches():
    # title + author match (ignore case, punctuation)
    reference_data = [
        {"author": "James Thomson", "title": "Winter", "source": "internet-poems"},
        {"author": "James Thomson", "title": "WINTER.", "source": "chadwyck-healey"},
    ]
    match, reason = multiple_matches(pl.from_dicts(reference_data))
    assert reason == "all rows match author + title"
    # first match is returned
    assert match["source"][0] == "internet-poems"

    # first source is prioritized when everything matches
    reference_data.reverse()
    match, reason = multiple_matches(pl.from_dicts(reference_data))
    # first match is returned
    assert match["source"][0] == "chadwyck-healey"

    # similar title; current logic can't match this
    reference_data = [
        {
            "author": "Robert Burns",
            "title": "Stay My Charmer Can You Leave",
            "source": "internet-poems",
        },
        {
            "author": "Robert Burns",
            "title": "STAY, MY CHARMER",
            "source": "chadwyck-healey",
        },
    ]
    match, reason = multiple_matches(pl.from_dicts(reference_data))
    assert match is None
    assert reason is None

    # majority match
    reference_data = [
        {
            "author": "James Hogg",
            "title": "Mador of the Moor",
            "source": "internet-poems",
        },
        {
            "author": "James Hogg",
            "title": "The Palmers Morning Hymn",
            "source": "internet-poems",
        },
        {
            "author": "James Hogg",
            "title": "MADOR OF THE MOOR. ",
            "source": "chadwyck-healey",
        },
    ]
    match, reason = multiple_matches(pl.from_dicts(reference_data))
    assert reason == "majority match author + title (2 out of 3)"
    assert match["source"][0] == "internet-poems"

    # poetry foundation title mismatch
    # majority match
    reference_data = [
        {
            "author": "William Shakespeare",
            "title": "As You Like It",
            "source": "internet-poems",
        },
        {
            "author": "William Shakespeare",
            "title": "Song: Blow, blow thou...",
            "source": "poetry-foundation",
        },
    ]
    match, reason = multiple_matches(pl.from_dicts(reference_data))
    assert reason == "duplicate author but not title; excluding Poetry Foundation"
    assert match["source"][0] == "internet-poems"


@pytest.fixture
def internet_poem(tmp_path):
    "test fixture to create an internet poem text file"
    poem = tmp_path / "internet-poems" / "Virgil_Aeneid.txt"
    poem.parent.mkdir()
    poem.write_text(
        "ARMA virumque cano, Troiae qui primus ab oris\nItaliam, fato profugus, Laviniaque venit"
    )
    return poem


@pytest.fixture
def chadwyck_healey_poem(tmp_path):
    "test fixture to create a chadwyck-healey poem text file"
    poem = tmp_path / "chadwyck-healey" / "Z200437771.txt"
    poem.parent.mkdir()
    poem.write_text(
        "Thou Spirit who ledst this glorious Eremite\nInto the Desert, his Victorious Field"
    )
    return poem


@pytest.fixture
def chadwyck_healey_csv(tmp_path):
    "fixture to create a test version of the chadwyck-healey metadata csv file"
    ch_meta_csv = tmp_path / CHADWYCK_HEALEY_CSV
    ch_meta_csv.parent.mkdir(exist_ok=True)
    # old version for now
    ch_meta_csv.write_text("""_id,author_dob,author_dob_valid,author_dod,author_dod_valid,author_firstname,author_lastname,author_role,id,id_link,meta_genre,meta_period,meta_rhymes,num_lines,title_edition,title_figure,title_hi,title_main,title_sub
5163205e1177fb6b57000000,1928,TRUE,,FALSE,Donald,Hall,orig.,Z200132299,http://lilaserver.stanford.edu:8080/poetry/Z200132299,,Twentieth-Century 1900-1999,,271,Sappho's Gymnasium (1994),,,The Night of the Day,""")
    return ch_meta_csv


@pytest.fixture
def poetry_foundation_csv(tmp_path):
    "fixture to create a test version of the poetry foundation csv file"
    pfound_csv = tmp_path / POETRY_FOUNDATION_CSV
    pfound_csv.parent.mkdir(exist_ok=True)
    pfound_csv.write_text(""",Author,Title,Poetry Foundation ID,Content
0,Wendy Videlock,!,55489,"Dear Writers, I’m compiling the first in what I hope ...\"""")
    return pfound_csv


def test_compile_text(
    tmp_path, capsys, internet_poem, chadwyck_healey_poem, poetry_foundation_csv
):
    os.chdir(tmp_path)
    # output file to be created
    text_file = tmp_path / "text.parquet"
    # using fixtures to simulate reference data to be compiled
    compile_text(tmp_path, text_file)
    assert text_file.exists()
    text_df = pl.read_parquet(text_file)
    # we expect three rows based on fixture data
    assert text_df.height == 3
    # sort so test order is reliable
    text_df = text_df.sort(pl.col("source"))
    text_row = text_df.row(0, named=True)
    assert text_row["id"] == chadwyck_healey_poem.stem
    assert text_row["text"].startswith("Thou Spirit who ledst")
    assert text_row["source"] == "chadwyck-healey"
    text_row = text_df.row(1, named=True)
    assert text_row["id"] == internet_poem.stem
    assert text_row["text"].startswith("ARMA virumque cano")
    assert text_row["source"] == "internet_poems"
    text_row = text_df.row(2, named=True)
    assert text_row["id"] == "55489"
    assert text_row["text"].startswith("Dear Writers, I’m compiling")
    assert text_row["source"] == "poetry-foundation"

    # should get a warning if poetry foundation csv is missing
    poetry_foundation_csv.unlink()
    compile_text(tmp_path, text_file)
    captured = capsys.readouterr()
    assert "Poetry Foundation csv file not found for text compilation" in captured.err


def test_compile_metadata(
    tmp_path, capsys, internet_poem, poetry_foundation_csv, chadwyck_healey_csv
):
    os.chdir(tmp_path)
    # output file to be created
    metadata_file = tmp_path / "metadata.parquet"
    # using fixtures to simulate reference data to be compiled
    compile_metadata(tmp_path, metadata_file)
    assert metadata_file.exists()
    meta_df = pl.read_parquet(metadata_file)
    assert meta_df.height == 3
    # sort so test order is reliable
    meta_df = meta_df.sort(pl.col("source"))
    poem_meta = meta_df.row(0, named=True)
    assert poem_meta["id"] == "Z200132299"
    assert poem_meta["source"] == "chadwyck-healey"
    assert poem_meta["author"] == "Donald Hall"
    assert poem_meta["title"] == "The Night of the Day"
    poem_meta = meta_df.row(1, named=True)
    assert poem_meta["id"] == internet_poem.stem
    assert poem_meta["author"] == "Virgil"
    assert poem_meta["title"] == "Aeneid"
    assert poem_meta["source"] == "internet_poems"
    poem_meta = meta_df.row(2, named=True)
    assert poem_meta["id"] == "55489"
    assert poem_meta["source"] == "poetry-foundation"
    assert poem_meta["author"] == "Wendy Videlock"
    assert poem_meta["title"] == "!"

    # when CSVs are not found, should see error messages
    poetry_foundation_csv.unlink()
    chadwyck_healey_csv.unlink()
    compile_metadata(tmp_path, metadata_file)
    captured = capsys.readouterr()
    assert "Chadwyck-Healey csv file not found for metadata compilation" in captured.err
    assert (
        "Poetry Foundation csv file not found for metadata compilation" in captured.err
    )


def test_process(
    tmp_path,
    capsys,
    internet_poem,
    chadwyck_healey_poem,
    poetry_foundation_csv,
    chadwyck_healey_csv,
):
    # minimal test to run process code
    # use fixture data and allow compile text/metadata methods to run
    os.chdir(tmp_path)
    input_file = tmp_path / "excerpts.csv"
    # create minimal excerpt input file from fixture object
    with input_file.open("w") as excerpts_csv:
        csvwriter = csv.DictWriter(excerpts_csv, fieldnames=Excerpt.fieldnames())
        csvwriter.writeheader()
        csvwriter.writerow(excerpt_earth.to_csv())

    output_file = tmp_path / "matches.csv"
    with patch("corppa.poetry_detection.refmatcha.REF_DATA_DIR", new=tmp_path):
        process(input_file, output_file)
        # expect to create the parquet files
        assert (tmp_path / TEXT_PARQUET_FILE).exists()
        assert (tmp_path / META_PARQUET_FILE).exists()

    captured = capsys.readouterr()
    # inspect summary output
    assert "Poetry reference text data: 3 entries" in captured.out
    assert "Omitting 0 poems with text length < 15" in captured.out
    assert "Poetry reference metadata: 3 entries" in captured.out
    assert "Input file has 1 excerpt" in captured.out
    assert "0 excerpts with matches" in captured.out

    # should error on invalid input file
    with input_file.open("w") as excerpts_csv:
        csvwriter = csv.writer(excerpts_csv)
        csvwriter.writerows([["foo", "bar", "baz"], ["a", "b", "c"]])

    with patch("corppa.poetry_detection.refmatcha.REF_DATA_DIR", new=tmp_path):
        with pytest.raises(SystemExit):
            process(input_file, output_file)

        captured = capsys.readouterr()
        assert "Input file does not have expected excerpt fields" in captured.err
        assert 'unable to find column "page_id"' in captured.err

    # should error on empty input file
    input_file.unlink()
    input_file.touch()
    with patch("corppa.poetry_detection.refmatcha.REF_DATA_DIR", new=tmp_path):
        with pytest.raises(SystemExit):
            process(input_file, output_file)

        captured = capsys.readouterr()
        assert "Input file does not have any data" in captured.err
        assert "empty CSV" in captured.err


@patch("corppa.poetry_detection.refmatcha.process")
def test_main(mock_process, capsys, tmp_path):
    input_file = tmp_path / "excerpts.csv"
    # call with non-existent input file
    with patch("sys.argv", ["refmatcha", str(input_file)]):
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert f"Error: input file {str(input_file)} does not exist" in captured.err

        # input file exists now; default output file
        input_file.touch()
        expected_output = input_file.with_name(f"{input_file.stem}_matched.csv")
        main()
        mock_process.assert_called_with(input_file, expected_output, recompile=False)

        # if output file exists, complain
        expected_output.touch()
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert (
            f"Error: output file {str(expected_output)} already exists, not overwriting"
            in captured.err
        )

    # specify output file
    output_file = tmp_path / "matches.csv"
    with patch(
        "sys.argv", ["refmatcha", str(input_file), "--output", str(output_file)]
    ):
        main()
        mock_process.assert_called_with(input_file, output_file, recompile=False)

        # still complains if output file already exists
        output_file.touch()
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert (
            f"Error: output file {str(output_file)} already exists, not overwriting"
            in captured.err
        )
