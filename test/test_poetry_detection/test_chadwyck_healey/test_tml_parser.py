import pytest

from corppa.poetry_detection.chadwyck_healey.tml_parser import (
    determine_encoding,
    filter_post_1928,
    replace_entities,
)


def test_determine_encoding(tmp_path):
    # Case: ASCII
    test_string = "PHOEBUS'S AEGIS\n"
    file_ascii = tmp_path / "ascii.txt"
    file_ascii.write_text("Phoebus's luck\n")
    assert determine_encoding(file_ascii) == "utf-8"

    # Case: UTF-8
    test_string = "PHŒBUS'S ÆGIS\n"
    file_utf8 = tmp_path / "uft8.txt"
    file_utf8.write_text(test_string, encoding="utf-8")
    assert determine_encoding(file_utf8) == "utf-8"

    # Case: Windows-1252
    test_string = "PHŒBUS'S ÆGIS\n"
    file_cp1252 = tmp_path / "cp1252.txt"
    file_cp1252.write_text(test_string, encoding="cp1252")
    assert determine_encoding(file_cp1252) == "cp1252"

    # Latin-1
    test_string = "PHOEBUS'S ÆGIS\n"
    file_latin1 = tmp_path / "latin1.txt"
    file_latin1.write_text(test_string, encoding="latin1")
    assert determine_encoding(file_latin1) == "latin1"

    # An oversight of this heuristic
    test_string = "PHŒBUS'S ÆGIS\n"
    file_macroman = tmp_path / "macroman.txt"
    file_macroman.write_text(test_string, encoding="macroman")
    assert determine_encoding(file_macroman) == "latin1"


def test_replace_entities():
    # Nothing to replace
    assert replace_entities("") == ""
    assert replace_entities("some text &c; ") == "some text &c; "

    # Replace indent
    text = "&indent;Who walks the wilderness."
    expected_result = "\tWho walks the wilderness."
    assert replace_entities(text) == expected_result

    # Replace some greek
    text = "&GREs;&grP;&grW;&GRST;"
    expected_result = "ἘΠΩς"
    assert replace_entities(text) == expected_result


def test_filter_post_1928():
    # Basic required poem metadata form
    basic_meta = {
        k: "" for k in ["period", "author_birth", "author_death", "edition_text"]
    }

    # 1. Poems with non-20th century period tags pass
    for period in {
        "Fifteenth-Century Poetry",
        "Middle English Poetry 1100-1400",
        "foo",
    }:
        poem_meta = basic_meta | {"period": period}
        # Basic case
        assert filter_post_1928(poem_meta)
        # Passes regardless of date(s) in author_birth and edition_text fields
        assert filter_post_1928(poem_meta | {"author_birth": "2000"})
        assert filter_post_1928(poem_meta | {"edition_text": "2000"})

    # 2. Author death year
    for period in ["", "Twentieth-Century 1900-1999"]:
        poem_meta = basic_meta | {"period": period}
        # Passes if author_death is before 1929
        for year in ["1901", "1928", "BC", "BC50"]:
            dod_meta = poem_meta | {"author_death": year}
            assert filter_post_1928(dod_meta)
            # Passes regardless of date(s) in author_birth and edition_text fields
            assert filter_post_1928(dod_meta | {"author_birth": "2000"})
            assert filter_post_1928(dod_meta | {"edition_text": "2000"})

    # 3. Check author birth year
    for period in ["", "Twentieth-Century 1900-1999"]:
        poem_meta = basic_meta | {"period": period}
        # Passes if author_birth is before 1915
        for year in {"1901", "B.C.40", "BC120", "cent.15th"}:
            assert filter_post_1928(poem_meta | {"author_birth": year})
            ## Passes regardless of date(s) in edition_fields
            assert filter_post_1928(
                poem_meta | {"author_birth": year, "edition_text": "2000"}
            )

        # Fails if author_birth is 1915 or later
        for year in {"1915", "1916", "2000"}:
            assert not filter_post_1928(poem_meta | {"author_birth": year})
            ## Fails regardless of date(s) in edition_fields
            assert not filter_post_1928(
                poem_meta | {"author_birth": year, "edition_text": "1900"}
            )

    # 3. Check edition title
    pass_titles = ["1915", "title [1928]", "poems 1901-1938 [1999]"]
    fail_titles = ["2010", "title (1929)", "poems 1940-60 (1980)"]
    for period in ["", "Twentieth-Century 1900-1999"]:
        poem_meta = basic_meta | {"period": period}
        # Passes if edition contains a date before 1929
        for title in pass_titles:
            assert filter_post_1928(poem_meta | {"edition_text": title})
        # Fails if editions contains date that are >= 1929
        for title in fail_titles:
            assert not filter_post_1928(poem_meta | {"edition_text": title})

    # 4. Catch-all
    ## 20th century poems without additional metadata fail
    assert not filter_post_1928(basic_meta | {"period": "Twentieth-Century 1900-1999"})
    ## poems without tags or any other additional metadata pass
    assert filter_post_1928(basic_meta)
