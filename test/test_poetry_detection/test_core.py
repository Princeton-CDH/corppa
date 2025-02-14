from unittest.mock import patch

import pytest

from corppa.poetry_detection.core import PoemExcerpt, Span


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

    @patch.object(Span, "has_overlap")
    @patch.object(Span, "overlap_length")
    def test_overlap_factor(self, mock_overlap_length, mock_has_overlap):
        span_a = Span(3, 6, "label")

        # no overlap
        mock_has_overlap.return_value = False
        assert span_a.overlap_factor("other span", ignore_label="bool") == 0
        mock_has_overlap.assert_called_once_with("other span", ignore_label="bool")
        mock_overlap_length.assert_not_called()

        # has overlap
        mock_has_overlap.reset_mock()
        mock_has_overlap.return_value = True
        mock_overlap_length.return_value = 3
        ## exact overlap
        span_b = Span(3, 6, "label")
        assert span_a.overlap_factor(span_b, ignore_label="bool") == 1
        mock_has_overlap.assert_called_once_with(span_b, ignore_label="bool")
        mock_overlap_length.assert_called_once_with(span_b, ignore_label="bool")
        ## partial overlap
        mock_has_overlap.reset_mock()
        mock_overlap_length.reset_mock()
        mock_overlap_length.return_value = 2
        span_b = Span(3, 5, "label")
        assert span_a.overlap_factor(span_b) == 2 / 3
        mock_has_overlap.assert_called_once_with(span_b, ignore_label=False)
        mock_overlap_length.assert_called_once_with(span_b, ignore_label=False)
        span_b = Span(2, 8, "label")
        mock_overlap_length.return_value = 3
        assert span_a.overlap_factor(span_b) == 3 / 6


class TestPoemExcerpt:
    def test_init(self):
        # Invalid PPA span indices
        error_message = "PPA span's start index 0 must be less than its end index 0"
        with pytest.raises(ValueError, match=error_message):
            poem_excerpt = PoemExcerpt(
                page_id="page_id",
                ppa_span_start=0,
                ppa_span_end=0,
                ppa_span_text="page_text",
                poem_id="poem_id",
                ref_corpus="corpus_id",
                detection_methods={"detect"},
                identification_methods={"id"},
            )
        error_message = "PPA span's start index 1 must be less than its end index 0"
        with pytest.raises(ValueError, match=error_message):
            poem_excerpt = PoemExcerpt(
                page_id="page_id",
                ppa_span_start=1,
                ppa_span_end=0,
                ppa_span_text="page_text",
                poem_id="poem_id",
                ref_corpus="corpus_id",
                detection_methods={"detect"},
                identification_methods={"id"},
            )
        # Partially unset reference span indices
        error_message = "Reference span's start and end index must both be set"
        with pytest.raises(ValueError, match=error_message):
            poem_excerpt = PoemExcerpt(
                page_id="page_id",
                ppa_span_start=0,
                ppa_span_end=1,
                ppa_span_text="page_text",
                poem_id="poem_id",
                ref_corpus="corpus_id",
                detection_methods={"detect"},
                identification_methods={"id"},
                ref_span_start=0,
            )
        with pytest.raises(ValueError, match=error_message):
            poem_excerpt = PoemExcerpt(
                page_id="page_id",
                ppa_span_start=0,
                ppa_span_end=1,
                ppa_span_text="page_text",
                poem_id="poem_id",
                ref_corpus="corpus_id",
                detection_methods={"detect"},
                identification_methods={"id"},
                ref_span_end=1,
            )
        # Invalid reference span indices
        error_message = (
            "Reference span's start index 0 must be less than its end index 0"
        )
        with pytest.raises(ValueError, match=error_message):
            poem_excerpt = PoemExcerpt(
                page_id="page_id",
                ppa_span_start=0,
                ppa_span_end=1,
                ppa_span_text="page_text",
                poem_id="poem_id",
                ref_corpus="corpus_id",
                detection_methods={"detect"},
                identification_methods={"id"},
                ref_span_start=0,
                ref_span_end=0,
            )
        error_message = (
            "Reference span's start index 1 must be less than its end index 0"
        )
        with pytest.raises(ValueError, match=error_message):
            poem_excerpt = PoemExcerpt(
                page_id="page_id",
                ppa_span_start=0,
                ppa_span_end=1,
                ppa_span_text="page_text",
                poem_id="poem_id",
                ref_corpus="corpus_id",
                detection_methods={"detect"},
                identification_methods={"id"},
                ref_span_start=1,
                ref_span_end=0,
            )

    def test_to_dict(self):
        # No optional fields
        poem_excerpt = PoemExcerpt(
            page_id="page_id",
            ppa_span_start=0,
            ppa_span_end=1,
            ppa_span_text="page_text",
            poem_id="poem_id",
            ref_corpus="corpus_id",
            detection_methods={"detect"},
            identification_methods={"id"},
        )
        expected_result = {
            "page_id": "page_id",
            "ppa_span_start": 0,
            "ppa_span_end": 1,
            "ppa_span_text": "page_text",
            "poem_id": "poem_id",
            "ref_corpus": "corpus_id",
            "detection_methods": ["detect"],
            "identification_methods": ["id"],
        }

        result = poem_excerpt.to_dict()
        assert result == expected_result

        # With optional fields
        poem_excerpt.ref_span_start = 2
        poem_excerpt.ref_span_end = 3
        poem_excerpt.ref_span_text = "ref_text"
        poem_excerpt.notes = "notes"

        expected_result |= {
            "ref_span_start": 2,
            "ref_span_end": 3,
            "ref_span_text": "ref_text",
            "notes": "notes",
        }

        result = poem_excerpt.to_dict()
        assert result == expected_result
