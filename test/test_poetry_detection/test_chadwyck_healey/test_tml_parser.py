import pytest

from corppa.poetry_detection.chadwyck_healey.tml_parser import (
    determine_encoding,
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
