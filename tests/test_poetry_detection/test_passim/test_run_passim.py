# Copyright (c) 2024-2025, Center for Digital Humanities, Princeton University
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path
from unittest.mock import NonCallableMock, patch

import pytest

from corppa.poetry_detection.passim.run_passim import (
    PASSIM_DEFAULTS,
    build_input_string,
    get_java_version,
    main,
    run_passim,
    set_spark_env_vars,
)


@patch.dict("os.environ")
def test_set_spark_env_vars():
    set_spark_env_vars(local_ip="ip", submit_args="args")
    assert os.environ["SPARK_LOCAL_IP"] == "ip"
    assert os.environ["SPARK_SUBMIT_ARGS"] == "args"


@patch("corppa.poetry_detection.passim.run_passim.run")
def test_get_java_version(mock_run):
    # Java 17
    java_version = "17.X.X"
    mock_run.return_value = NonCallableMock(stderr=f'"{java_version}"')
    assert get_java_version() == java_version
    mock_run.assert_called_once_with(
        "java -version", shell=True, capture_output=True, text=True
    )

    # Java 11
    java_version = "11.X.X"
    mock_run.reset_mock()
    mock_run.return_value = NonCallableMock(stderr=f'"{java_version}"')
    assert get_java_version() == java_version
    mock_run.assert_called_once_with(
        "java -version", shell=True, capture_output=True, text=True
    )

    # Java 8u371
    java_version = "1.8.0_371"
    mock_run.reset_mock()
    mock_run.return_value = NonCallableMock(stderr=f'"{java_version}"')
    assert get_java_version() == java_version
    mock_run.assert_called_once_with(
        "java -version", shell=True, capture_output=True, text=True
    )

    # Java 8u381
    java_version = "1.8.0_381"
    mock_run.reset_mock()
    mock_run.return_value = NonCallableMock(stderr=f'"{java_version}"')
    assert get_java_version() == java_version
    mock_run.assert_called_once_with(
        "java -version", shell=True, capture_output=True, text=True
    )

    # Error: Java 8u361
    java_version = "1.8.0_361"
    mock_run.reset_mock()
    mock_run.return_value = NonCallableMock(stderr=f'"{java_version}"')
    err_msg = f"Java {java_version} is unsupported. Only versions 8u371 and higher are supported."
    with pytest.raises(RuntimeError, match=err_msg):
        get_java_version()
    mock_run.assert_called_once_with(
        "java -version", shell=True, capture_output=True, text=True
    )

    # Error: Java 23
    java_version = "23.X.X"
    mock_run.reset_mock()
    mock_run.return_value = NonCallableMock(stderr=f'"{java_version}"')
    err_msg = r"Java 23.X.X is unsupported. Spark requires Java 8\*/11/17."
    with pytest.raises(RuntimeError, match=err_msg):
        get_java_version()
    mock_run.assert_called_once_with(
        "java -version", shell=True, capture_output=True, text=True
    )


def test_build_input_string():
    # single reference corpus
    ppa_corpus = Path("ppa.jsonl")
    ref_corpus = [Path("ref.jsonl")]
    expected_result = "{ppa.jsonl,ref.jsonl}"
    assert build_input_string(ppa_corpus, ref_corpus) == expected_result

    # multiple reference corpus
    ref_corpora = [Path("refa.jsonl"), Path("refb.jsonl")]
    expected_result = "{ppa.jsonl,refa.jsonl,refb.jsonl}"
    assert build_input_string(ppa_corpus, ref_corpora) == expected_result


@patch("corppa.poetry_detection.passim.run_passim.run")
@patch("corppa.poetry_detection.passim.run_passim.build_input_string")
@patch("corppa.poetry_detection.passim.run_passim.get_java_version")
def test_run_passim(mock_java, mock_build_input_str, mock_run, tmp_path, capsys):
    # File setup
    out_dir = tmp_path / "out"
    align_dir = out_dir / "align.json"
    align_dir.mkdir(parents=True)

    # Basic case, without floating ngrams
    mock_build_input_str.return_value = "input"
    success_file = align_dir / "_SUCCESS"
    success_file.touch()
    assert run_passim(
        "ppa",
        "ref",
        out_dir,
        max_df=0,
        min_match=1,
        ngram_size=2,
        gap=3,
        min_align=4,
        floating_ngrams=False,
        verbose=False,
    )
    mock_java.assert_called_once_with()
    mock_build_input_str.assert_called_once_with("ppa", "ref")
    passim_args = [
        "passim",
        "input",
        out_dir,
        "--fields",
        "corpus",
        "--filterpairs",
        "corpus <> 'ppa' AND corpus2 = 'ppa'",
        "--pairwise",
        "-u",
        "0",
        "-m",
        "1",
        "-n",
        "2",
        "-g",
        "3",
        "-a",
        "4",
    ]
    mock_run.assert_called_once_with(passim_args, check=True, capture_output=True)
    # Basic case with floating ngrams
    mock_java.reset_mock()
    mock_build_input_str.reset_mock()
    mock_run.reset_mock()
    assert run_passim(
        "ppa",
        "ref",
        out_dir,
        max_df=0,
        min_match=1,
        ngram_size=2,
        gap=3,
        min_align=4,
        floating_ngrams=True,
        verbose=False,
    )
    mock_java.assert_called_once_with()
    mock_build_input_str.assert_called_once_with("ppa", "ref")
    passim_args.append("--floating-ngrams")
    mock_run.assert_called_once_with(passim_args, check=True, capture_output=True)

    # Spark fails for some reason, but an error does not occur while running passim
    success_file.unlink()
    assert not run_passim(
        "ppa",
        "ref",
        out_dir,
        max_df=0,
        min_match=1,
        ngram_size=2,
        gap=3,
        min_align=4,
        floating_ngrams=True,
        verbose=False,
    )
    captured_stderr = capsys.readouterr().err
    assert captured_stderr == "ERROR: An error occurred while running passim\n"

    # Running passim throws an error
    mock_run.side_effect = subprocess.CalledProcessError(1, "passim")
    assert not run_passim(
        "ppa",
        "ref",
        out_dir,
        max_df=0,
        min_match=1,
        ngram_size=2,
        gap=3,
        min_align=4,
        floating_ngrams=True,
        verbose=False,
    )
    captured_stderr = capsys.readouterr().err
    assert captured_stderr == "ERROR: An error occurred while running passim\n"


@patch("corppa.poetry_detection.passim.run_passim.set_spark_env_vars")
@patch("corppa.poetry_detection.passim.run_passim.run_passim")
def test_main(mock_run_passim, mock_set_spark_env_vars, tmp_path):
    # required parameters
    ppa_corpus = tmp_path / "ppa_pages.jsonl"
    ppa_corpus.touch()
    ref_corpus = tmp_path / "ref_corpus.jsonl"
    ref_corpus.touch()
    output_dir = tmp_path / "passim-output"

    args = [
        "--ppa-corpus",
        str(ppa_corpus),
        "--ref-corpus",
        str(ref_corpus),
        "--output-dir",
        str(output_dir),
    ]
    main(args)
    mock_set_spark_env_vars.assert_called_once()
    mock_run_passim.assert_called_once_with(
        ppa_corpus,
        [ref_corpus],
        output_dir,
        # defaults from passim defaults object
        max_df=PASSIM_DEFAULTS.max_df,
        min_match=PASSIM_DEFAULTS.min_match,
        ngram_size=PASSIM_DEFAULTS.ngram_size,
        gap=PASSIM_DEFAULTS.gap,
        min_align=PASSIM_DEFAULTS.min_align,
        floating_ngrams=False,  # off by default
        verbose=False,  # off by default
    )

    # test with non-default parameters
    max_df = 50
    min_match = 7
    gap = 21
    min_align = 19
    ngram_size = 5
    args.extend(
        [
            "--max-df",
            str(max_df),
            "--min-match",
            str(min_match),
            "--ngram-size",
            str(ngram_size),
            "--floating-ngrams",
            "--gap",
            str(gap),
            "--min-align",
            str(min_align),
            "-v",
        ]
    )
    mock_run_passim.reset_mock()
    main(args)
    mock_run_passim.assert_called_once_with(
        ppa_corpus,
        [ref_corpus],
        output_dir,
        # uses values specified
        max_df=max_df,
        min_match=min_match,
        ngram_size=ngram_size,
        gap=gap,
        min_align=min_align,
        floating_ngrams=True,
        verbose=True,
    )
