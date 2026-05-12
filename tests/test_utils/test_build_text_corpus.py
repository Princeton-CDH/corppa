# Copyright (c) 2024-2025, Center for Digital Humanities, Princeton University
# SPDX-License-Identifier: Apache-2.0

import tarfile
from inspect import isgenerator
from unittest.mock import call, patch

from corppa.utils.build_text_corpus import (
    build_text_corpus,
    get_text_record,
    save_text_corpus,
    text_corpus_from_tarfile,
)


def test_get_text_record(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("Some\n text.", encoding="utf-8")

    result = get_text_record(test_file)
    expected_result = {"id": "test", "text": "Some\n text."}
    assert result == expected_result


@patch("corppa.utils.build_text_corpus.get_text_record")
def test_build_text_corpus(mock_get_text_record, tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    txt_a = tmp_path / "a.txt"
    txt_a.touch()
    txt_b = corpus_dir / "b.txt"
    txt_b.touch()
    sub_dir = corpus_dir / "c"
    sub_dir.mkdir()
    txt_c = sub_dir / "c.txt"
    txt_c.write_text("test", encoding="utf-8")
    other_file = sub_dir / "other.xml"
    other_file.touch()

    # Simple case
    mock_get_text_record.return_value = "some record"
    results = build_text_corpus(sub_dir)
    assert isgenerator(results)
    assert list(results) == ["some record"]
    mock_get_text_record.assert_called_once_with(txt_c)

    # Nested directories & ignored files
    mock_get_text_record.reset_mock()
    mock_get_text_record.side_effect = ["b", "c"]
    results = build_text_corpus(corpus_dir)
    assert list(results) == ["b", "c"]
    assert mock_get_text_record.call_count == 2
    mock_get_text_record.assert_has_calls([call(txt_b), call(txt_c)])


def test_build_text_corpus_from_tarfile(tmp_path):
    # create tar.gzip of text files to test
    tarfile_path = tmp_path / "texts.tar.gz"
    textfile = tmp_path / "foo.txt"
    textfile.write_text("some texty text")
    osx_meta_file = tmp_path / "._meta"
    osx_meta_file.touch()

    with tarfile.open(tarfile_path, "w:gz") as tar:
        tar.add(textfile)
        tar.add(osx_meta_file)

    # should ignore the meta file and result in a corpus with one entry
    corpus = list(text_corpus_from_tarfile(tarfile_path, disable_progress=True))
    assert len(corpus) == 1
    assert corpus[0]["id"] == "foo"
    assert corpus[0]["text"] == "some texty text"


@patch("corppa.utils.build_text_corpus.build_text_corpus")
@patch("corppa.utils.build_text_corpus.orjsonl")
def test_save_text_corpus(mock_orjsonl, mock_build_text_corpus):
    mock_build_text_corpus.return_value = "text corpus"
    save_text_corpus("input dir", "output file")
    mock_build_text_corpus.assert_called_once_with("input dir")
    mock_orjsonl.save.assert_called_once_with("output file", "text corpus")
