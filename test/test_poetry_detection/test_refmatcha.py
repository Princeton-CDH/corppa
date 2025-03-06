import polars as pl
import pytest

from corppa.poetry_detection.core import Excerpt
from corppa.poetry_detection.refmatcha import (
    SCRIPT_ID,
    generate_search_text,
    identify_excerpt,
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


def test_identify_excerpt(reference_df):
    # test identify_excerpt method directly
    excerpt_row = excerpt_earth.to_dict()
    # manually set search text for now
    excerpt_row["search_text"] = "The earth is the Lords and the fullness thereof"

    # single match
    id_result = identify_excerpt(excerpt_row, reference_df)
    assert id_result["poem_id"] == ref_poetry_data[0]["id"]
    assert id_result["ref_corpus"] == ref_poetry_data[0]["source"]
    # these are from the searchable version of the input text
    assert id_result["ref_span_start"] == 50
    assert id_result["ref_span_end"] == 97
    assert (
        id_result["ref_span_text"] == "the earth is the Lords and the fullness thereof"
    )
    assert id_result["notes"] == "refmatcha: single match on text"
    assert id_result["identification_methods"] == [SCRIPT_ID]

    # single match, whitespace agnostic
    excerpt_row["search_text"] = (
        "The earth    is the\nLords\t\tand the fullness thereof"
    )
    id_result = identify_excerpt(excerpt_row, reference_df)
    assert id_result["poem_id"] == ref_poetry_data[0]["id"]

    reference_fields = [
        "poem_id",
        "ref_corpus",
        "ref_span_text",
        "ref_span_start",
        "ref_span_end",
        "identification_methods",
    ]

    # no match
    excerpt_row["search_text"] = "Disdain forbids me and my dread of shame"
    id_result = identify_excerpt(excerpt_row, reference_df)
    for ref_field in reference_fields:
        assert id_result[ref_field] is None

    # too many matches, can't consolidate
    excerpt_row["search_text"] = "wonderful works"
    id_result = identify_excerpt(excerpt_row, reference_df)
    for ref_field in reference_fields:
        assert id_result[ref_field] is None


def test_identify_excerpt_map_elements(reference_df):
    # test identify_excerpt method as it will be called
    # on a dataframe using map_elements

    # load test excerpt as dataframe
    input_df = pl.from_dicts([excerpt_earth.to_dict()])
    # generate search text field from excerpt text
    input_df = generate_search_text(
        input_df, field="ppa_span_text", output_field="search_text"
    )

    result = input_df.with_columns(
        pl.struct(pl.all())
        .map_elements(
            lambda row: identify_excerpt(row, reference_df), return_dtype=pl.Struct
        )
        .alias("t_struct")
    ).unnest("t_struct")

    row_dict = result.row(0, named=True)
    # expect poem id to be set from reference poem
    assert row_dict["poem_id"] == ref_poetry_data[0]["id"]
    assert row_dict["ref_corpus"] == ref_poetry_data[0]["source"]
    assert row_dict["notes"] == "refmatcha: single match on text"
