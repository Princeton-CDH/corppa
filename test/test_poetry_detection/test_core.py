from dataclasses import replace
from typing import Optional
from unittest.mock import patch

import pytest

from corppa.poetry_detection.core import (
    MULTIVAL_DELIMITER,
    Excerpt,
    LabeledExcerpt,
    Span,
    field_real_type,
    input_to_set,
)


class TestSpan:
    def test_init(self):
        # Invalid range: end index < start index
        error_message = "Start index must be less than end index"
        with pytest.raises(ValueError, match=error_message):
            span = Span(9, 2, "label")
        # Invalid range: end index = < start index
        with pytest.raises(ValueError, match=error_message):
            span = Span(2, 2, "label")

    def test_len(self):
        assert len(Span(2, 5, "label")) == 3
        assert len(Span(0, 42, "label")) == 42

    def test_has_overlap(self):
        span_a = Span(3, 6, "same")
        # test once with a span with the same label and once with a different label
        for label in ["same", "different"]:
            same_label = label == "same"
            ## exact overlap
            span_b = Span(3, 6, label)
            assert span_a.has_overlap(span_b) == same_label
            assert span_a.has_overlap(span_b, ignore_label=True)
            ## partial overlap: subsets
            span_b = Span(4, 5, label)
            assert span_a.has_overlap(span_b) == same_label
            assert span_a.has_overlap(span_b, ignore_label=True)
            span_b = Span(3, 5, label)
            assert span_a.has_overlap(span_b) == same_label
            assert span_a.has_overlap(span_b, ignore_label=True)
            span_b = Span(4, 6, label)
            assert span_a.has_overlap(span_b) == same_label
            assert span_a.has_overlap(span_b, ignore_label=True)
            ## partial overlap: not subsets
            span_b = Span(1, 5, label)
            assert span_a.has_overlap(span_b) == same_label
            assert span_a.has_overlap(span_b, ignore_label=True)
            span_b = Span(4, 8, label)
            assert span_a.has_overlap(span_b) == same_label
            assert span_a.has_overlap(span_b, ignore_label=True)
            ## no overlap
            span_b = Span(0, 1, label)
            assert not span_a.has_overlap(span_b)
            assert not span_a.has_overlap(span_b, ignore_label=True)
            span_b = Span(0, 3, label)
            assert not span_a.has_overlap(span_b)
            assert not span_a.has_overlap(span_b, ignore_label=True)

    def test_is_exact_match(self):
        span_a = Span(3, 6, "label")

        for label in ["label", "different"]:
            # exact overlap
            span_b = Span(3, 6, label)
            assert span_a.is_exact_match(span_b) == (label == "label")
            assert span_a.is_exact_match(span_b, ignore_label=True)
            # partial overlap
            span_b = Span(3, 5, label)
            assert not span_a.is_exact_match(span_b)
            assert not span_a.is_exact_match(span_b, ignore_label=True)
            span_b = Span(2, 8, label)
            assert not span_a.is_exact_match(span_b)
            assert not span_a.is_exact_match(span_b, ignore_label=True)
            # no overlap
            span_b = Span(0, 1, label)
            assert not span_a.is_exact_match(span_b)
            assert not span_a.is_exact_match(span_b, ignore_label=True)
            span_b = Span(0, 3, label)
            assert not span_a.is_exact_match(span_b)
            assert not span_a.is_exact_match(span_b, ignore_label=True)

    @patch.object(Span, "has_overlap")
    def test_overlap_length(self, mock_has_overlap):
        span_a = Span(3, 6, "label")

        # no overlap
        mock_has_overlap.return_value = False
        assert span_a.overlap_length("other span", ignore_label="bool") == 0
        mock_has_overlap.assert_called_once_with("other span", ignore_label="bool")

        # has overlap
        mock_has_overlap.reset_mock()
        mock_has_overlap.return_value = True
        ## exact overlap
        span_b = Span(3, 6, "label")
        assert span_a.overlap_length(span_b) == 3
        mock_has_overlap.assert_called_once_with(span_b, ignore_label=False)
        ## partial overlap
        span_b = Span(3, 5, "label")
        assert span_a.overlap_length(span_b) == 2
        span_b = Span(2, 8, "label")
        assert span_a.overlap_length(span_b) == 3

    @patch.object(Span, "overlap_length")
    def test_overlap_factor(self, mock_overlap_length):
        span_a = Span(3, 6, "label")

        # no overlap
        mock_overlap_length.return_value = 0
        assert span_a.overlap_factor("other span", ignore_label="bool") == 0
        mock_overlap_length.assert_called_once_with("other span", ignore_label="bool")

        # has overlap
        mock_overlap_length.reset_mock()
        mock_overlap_length.return_value = 3
        ## exact overlap
        span_b = Span(3, 6, "label")
        assert span_a.overlap_factor(span_b, ignore_label="bool") == 1
        mock_overlap_length.assert_called_once_with(span_b, ignore_label="bool")
        ## partial overlap
        mock_overlap_length.reset_mock()
        mock_overlap_length.return_value = 2
        span_b = Span(3, 5, "label")
        assert span_a.overlap_factor(span_b) == 2 / 3
        mock_overlap_length.assert_called_once_with(span_b, ignore_label=False)
        span_b = Span(2, 8, "label")
        mock_overlap_length.return_value = 3
        assert span_a.overlap_factor(span_b) == 3 / 6


class TestExcerpt:
    def test_init(self):
        # Invalid PPA span indices
        error_message = "PPA span's start index 0 must be less than its end index 0"
        with pytest.raises(ValueError, match=error_message):
            excerpt = Excerpt(
                page_id="page_id",
                ppa_span_start=0,
                ppa_span_end=0,
                ppa_span_text="page_text",
                detection_methods={"detect"},
            )
        error_message = "PPA span's start index 1 must be less than its end index 0"
        with pytest.raises(ValueError, match=error_message):
            excerpt = Excerpt(
                page_id="page_id",
                ppa_span_start=1,
                ppa_span_end=0,
                ppa_span_text="page_text",
                detection_methods={"detect"},
            )
        # Empty detection method set
        error_message = "Must specify at least one detection method"
        with pytest.raises(ValueError, match=error_message):
            excerpt = Excerpt(
                page_id="page_id",
                ppa_span_start=0,
                ppa_span_end=1,
                ppa_span_text="page_text",
                detection_methods={},
            )

        # Single unsupported detection method
        error_message = "Unsupported detection method: unknown"
        with pytest.raises(ValueError, match=error_message):
            excerpt = Excerpt(
                page_id="page_id",
                ppa_span_start=0,
                ppa_span_end=1,
                ppa_span_text="page_text",
                detection_methods={"unknown"},
            )

        # Multiple unsupported detection methods
        error_message = r"Unsupported detection methods: (u1, u2)|(u2, u1)"
        with pytest.raises(ValueError, match=error_message):
            excerpt = Excerpt(
                page_id="page_id",
                ppa_span_start=0,
                ppa_span_end=1,
                ppa_span_text="page_text",
                detection_methods={"u1", "u2", "manual"},
            )

    def test_excerpt_id(self):
        # Single detection method
        for detection_method in ["adjudication", "manual", "passim", "xml"]:
            excerpt = Excerpt(
                page_id="page_id",
                ppa_span_start=0,
                ppa_span_end=1,
                ppa_span_text="page_text",
                detection_methods={detection_method},
            )
            expected_result = f"{detection_method[0]}@0:1"
            assert excerpt.excerpt_id == expected_result

        # Multiple detection methods
        excerpt = Excerpt(
            page_id="page_id",
            ppa_span_start=0,
            ppa_span_end=1,
            ppa_span_text="page_text",
            detection_methods={"adjudication", "passim"},
        )
        assert excerpt.excerpt_id == "c@0:1"

    def test_to_dict(self):
        # No optional fields
        excerpt = Excerpt(
            page_id="page_id",
            ppa_span_start=0,
            ppa_span_end=1,
            ppa_span_text="page_text",
            detection_methods={"manual"},
        )
        expected_result = {
            "page_id": "page_id",
            "ppa_span_start": 0,
            "ppa_span_end": 1,
            "ppa_span_text": "page_text",
            "detection_methods": ["manual"],
            "excerpt_id": "m@0:1",
        }

        result = excerpt.to_dict()
        assert result == expected_result

        # With optional fields
        excerpt = replace(excerpt, notes="notes")
        expected_result["notes"] = "notes"

        result = excerpt.to_dict()
        assert result == expected_result

    def test_to_csv(self):
        # No optional fields
        excerpt = Excerpt(
            page_id="page_id",
            ppa_span_start=0,
            ppa_span_end=1,
            ppa_span_text="page_text",
            detection_methods={"manual"},
        )
        expected_result = {
            "page_id": "page_id",
            "ppa_span_start": 0,
            "ppa_span_end": 1,
            "ppa_span_text": "page_text",
            "detection_methods": "manual",
            "excerpt_id": "m@0:1",
        }
        assert excerpt.to_csv() == expected_result

        # With optional fields
        excerpt = replace(excerpt, notes="notes")
        expected_result["notes"] = "notes"
        assert excerpt.to_csv() == expected_result

        # with multiple values for set field
        excerpt = replace(excerpt, detection_methods={"manual", "passim"})
        expected_result["excerpt_id"] = "c@0:1"
        expected_result["detection_methods"] = MULTIVAL_DELIMITER.join(
            ["manual", "passim"]
        )
        assert excerpt.to_csv() == expected_result

    def test_from_dict(self):
        # JSONL-friendly dict
        excerpt = Excerpt(
            page_id="page_id",
            ppa_span_start=0,
            ppa_span_end=1,
            ppa_span_text="page_text",
            detection_methods={"manual", "xml"},
        )
        jsonl_dict = excerpt.to_dict()
        assert Excerpt.from_dict(jsonl_dict) == excerpt

        # CSV-friendly dict
        ## Multiple detection methods
        csv_dict = excerpt.to_csv()
        assert Excerpt.from_dict(csv_dict) == excerpt
        ## Single detection methods
        excerpt = replace(excerpt, detection_methods={"adjudication"})
        csv_dict = excerpt.to_csv()
        assert Excerpt.from_dict(csv_dict) == excerpt

        # Error if detection_methods field has bad type
        bad_dict = csv_dict | {"detection_methods": 0}
        error_message = "Unexpected value type 'int' for detection_methods"
        with pytest.raises(ValueError, match=error_message):
            Excerpt.from_dict(bad_dict)

    def test_strip_whitespace(self):
        # No leading or trailing whitespace
        excerpt = Excerpt(
            page_id="page_id",
            ppa_span_start=1,
            ppa_span_end=12,
            ppa_span_text="page_text",
            detection_methods={"manual"},
        )
        expected_result = Excerpt(
            page_id="page_id",
            ppa_span_start=1,
            ppa_span_end=12,
            ppa_span_text="page_text",
            detection_methods={"manual"},
        )
        assert excerpt.strip_whitespace() == expected_result

        # Leading whitespace
        excerpt = Excerpt(
            page_id="page_id",
            ppa_span_start=0,
            ppa_span_end=13,
            ppa_span_text="\r\npage_text",
            detection_methods={"manual"},
        )
        expected_result = Excerpt(
            page_id="page_id",
            ppa_span_start=2,
            ppa_span_end=13,
            ppa_span_text="page_text",
            detection_methods={"manual"},
        )
        assert excerpt.strip_whitespace() == expected_result

        # Trailing whitespace
        excerpt = Excerpt(
            page_id="page_id",
            ppa_span_start=0,
            ppa_span_end=13,
            ppa_span_text="page_text\t ",
            detection_methods={"manual"},
        )
        expected_result = Excerpt(
            page_id="page_id",
            ppa_span_start=0,
            ppa_span_end=11,
            ppa_span_text="page_text",
            detection_methods={"manual"},
        )
        assert excerpt.strip_whitespace() == expected_result

        # Leading & trailing whitespace
        excerpt = Excerpt(
            page_id="page_id",
            ppa_span_start=0,
            ppa_span_end=13,
            ppa_span_text=" page_text\n",
            detection_methods={"manual"},
        )
        expected_result = Excerpt(
            page_id="page_id",
            ppa_span_start=1,
            ppa_span_end=12,
            ppa_span_text="page_text",
            detection_methods={"manual"},
        )
        assert excerpt.strip_whitespace() == expected_result

    def test_fieldnames(self):
        fieldnames = Excerpt.fieldnames()
        # should match the names of the fields as declared
        # and in the same order
        assert fieldnames == [
            "page_id",
            "ppa_span_start",
            "ppa_span_end",
            "ppa_span_text",
            "detection_methods",
            "notes",
            "excerpt_id",
        ]

    def test_field_types(self):
        field_types = Excerpt.field_types()
        assert field_types == {
            "page_id": str,
            "ppa_span_start": int,
            "ppa_span_end": int,
            "ppa_span_text": str,
            "detection_methods": set,
            "notes": str,
            "excerpt_id": str,
        }


class TestLabeledExcerpt:
    def test_init(self):
        # Partially unset reference span indices
        error_message = "Reference span's start and end index must both be set"
        with pytest.raises(ValueError, match=error_message):
            excerpt = LabeledExcerpt(
                page_id="page_id",
                ppa_span_start=0,
                ppa_span_end=1,
                ppa_span_text="page_text",
                poem_id="poem_id",
                ref_corpus="corpus_id",
                detection_methods={"manual"},
                identification_methods={"id"},
                ref_span_start=0,
            )
        with pytest.raises(ValueError, match=error_message):
            excerpt = LabeledExcerpt(
                page_id="page_id",
                ppa_span_start=0,
                ppa_span_end=1,
                ppa_span_text="page_text",
                poem_id="poem_id",
                ref_corpus="corpus_id",
                detection_methods={"manual"},
                identification_methods={"id"},
                ref_span_end=1,
            )
        # Invalid reference span indices
        error_message = (
            "Reference span's start index 0 must be less than its end index 0"
        )
        with pytest.raises(ValueError, match=error_message):
            excerpt = LabeledExcerpt(
                page_id="page_id",
                ppa_span_start=0,
                ppa_span_end=1,
                ppa_span_text="page_text",
                poem_id="poem_id",
                ref_corpus="corpus_id",
                detection_methods={"manual"},
                identification_methods={"id"},
                ref_span_start=0,
                ref_span_end=0,
            )
        error_message = (
            "Reference span's start index 1 must be less than its end index 0"
        )
        with pytest.raises(ValueError, match=error_message):
            excerpt = LabeledExcerpt(
                page_id="page_id",
                ppa_span_start=0,
                ppa_span_end=1,
                ppa_span_text="page_text",
                poem_id="poem_id",
                ref_corpus="corpus_id",
                detection_methods={"manual"},
                identification_methods={"id"},
                ref_span_start=1,
                ref_span_end=0,
            )

        # Empty identification method set
        error_message = "Must specify at least one identification method"
        with pytest.raises(ValueError, match=error_message):
            excerpt = LabeledExcerpt(
                page_id="page_id",
                ppa_span_start=0,
                ppa_span_end=1,
                ppa_span_text="page_text",
                poem_id="poem_id",
                ref_corpus="corpus_id",
                detection_methods={"manual"},
                identification_methods={},
            )

    def test_to_dict(self):
        # No optional fields
        excerpt = LabeledExcerpt(
            page_id="page_id",
            ppa_span_start=0,
            ppa_span_end=1,
            ppa_span_text="page_text",
            poem_id="poem_id",
            ref_corpus="corpus_id",
            detection_methods={"manual"},
            identification_methods={"id"},
        )
        expected_result = {
            "page_id": "page_id",
            "ppa_span_start": 0,
            "ppa_span_end": 1,
            "ppa_span_text": "page_text",
            "poem_id": "poem_id",
            "ref_corpus": "corpus_id",
            "detection_methods": ["manual"],
            "identification_methods": ["id"],
            "excerpt_id": "m@0:1",
        }

        result = excerpt.to_dict()
        assert result == expected_result

        # With optional fields
        excerpt = replace(
            excerpt,
            ref_span_start=2,
            ref_span_end=3,
            ref_span_text="ref_text",
            notes="notes",
        )

        expected_result |= {
            "ref_span_start": 2,
            "ref_span_end": 3,
            "ref_span_text": "ref_text",
            "notes": "notes",
        }

        result = excerpt.to_dict()
        assert result == expected_result

    def test_from_dict(self):
        # JSONL-friendly dict
        excerpt = LabeledExcerpt(
            page_id="page_id",
            ppa_span_start=0,
            ppa_span_end=1,
            ppa_span_text="page_text",
            poem_id="poem_id",
            ref_corpus="poems",
            detection_methods={"adjudication", "manual"},
            identification_methods={"manual", "matcha"},
        )
        jsonl_dict = excerpt.to_dict()
        assert LabeledExcerpt.from_dict(jsonl_dict) == excerpt

        # CSV-friendly dict
        csv_dict = excerpt.to_csv()
        assert LabeledExcerpt.from_dict(csv_dict) == excerpt

        # Error if detection or identification methods fields have bad type
        for field in ["detection_methods", "identification_methods"]:
            bad_dict = csv_dict | {field: 0}
            error_message = f"Unexpected value type 'int' for {field}"
            with pytest.raises(ValueError, match=error_message):
                LabeledExcerpt.from_dict(bad_dict)

    def test_fieldnames(self):
        fieldnames = LabeledExcerpt.fieldnames()
        # should inherit from Excerpt but include
        # additional fields
        assert fieldnames == Excerpt.fieldnames() + [
            "poem_id",
            "ref_corpus",
            "ref_span_start",
            "ref_span_end",
            "ref_span_text",
            "identification_methods",
        ]

    def test_field_types(self):
        field_types = LabeledExcerpt.field_types()
        expected_types = Excerpt.field_types()
        expected_types.update(
            {
                "poem_id": str,
                "ref_corpus": str,
                "ref_span_start": int,
                "ref_span_end": int,
                "ref_span_text": str,
                "identification_methods": set,
            }
        )

        assert field_types == expected_types

        # field types for subclass should not be the same
        # (caching the method breaks this)
        assert LabeledExcerpt.field_types() != Excerpt.field_types()

    def test_from_excerpt(self):
        excerpt = Excerpt(
            page_id="page_id",
            ppa_span_start=0,
            ppa_span_end=1,
            ppa_span_text="page_text",
            detection_methods={"manual"},
        )
        # initialize with all required fields
        labeled_ex = LabeledExcerpt.from_excerpt(
            excerpt,
            poem_id="Z1234",
            ref_corpus="test-corpus",
            identification_methods={"manual"},
        )
        # spot check resulting object
        assert labeled_ex.page_id == excerpt.page_id
        assert labeled_ex.excerpt_id == excerpt.excerpt_id
        assert labeled_ex.poem_id == "Z1234"
        assert labeled_ex.ref_corpus == "test-corpus"

        # should be able to override fields
        labeled_ex = LabeledExcerpt.from_excerpt(
            excerpt,
            ppa_span_end=10,
            poem_id="Z1234",
            ref_corpus="test-corpus",
            identification_methods={"manual"},
        )
        assert labeled_ex.ppa_span_end == 10

        # results in init error without required fields
        with pytest.raises(TypeError, match="missing 3 required"):
            LabeledExcerpt.from_excerpt(excerpt)


def test_field_real_type():
    # regular type
    assert field_real_type(str) == str
    # type annotation / alas
    assert field_real_type(set[str]) == set
    # optional (= union of type and NoneType)
    assert field_real_type(Optional[int]) == int

    with pytest.raises(TypeError):
        # method doesn't support anything that isn't a type or type alias
        field_real_type("text content")


def test_input_to_set():
    # string, single value
    assert input_to_set("a") == {"a"}
    # delimited string
    assert input_to_set(MULTIVAL_DELIMITER.join(["a", "b"])) == {"a", "b"}
    # list
    assert input_to_set(["a", "b", "c"]) == {"a", "b", "c"}
    # set
    assert input_to_set({"a", "b", "c"}) == {"a", "b", "c"}
    # unsupported input
    with pytest.raises(ValueError, match="Unexpected value type 'int'"):
        input_to_set(1)
