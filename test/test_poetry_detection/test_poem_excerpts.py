import pytest

from corppa.poetry_detection.poem_excerpts import PoemExcerpt


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
