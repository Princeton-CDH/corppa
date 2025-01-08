import sys
from unittest.mock import NonCallableMock, call, patch

import pytest

from corppa.poetry_detection.evaluation.evaluate_poetry_spans import (
    PageEvaluation,
    PageReferenceSpans,
    PageSystemSpans,
    Span,
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

        # Normal case
        span = Span(2, 5, "label")
        assert span.start == 2
        assert span.end == 5
        assert span.label == "label"

    def test_len(self):
        assert len(Span(2, 5, "label")) == 3
        assert len(Span(0, 42, "label")) == 42

    def test_eq(self):
        span_a = Span(3, 6, "label")
        assert span_a == Span(3, 6, "label")
        assert span_a != Span(4, 8, "label")
        assert span_a != Span(3, 6, "different")

    def test_repr(self):
        span_a = Span(3, 6, "label")
        assert repr(span_a) == "Span(3, 6, label)"

    def test_has_overlap(self):
        span_a = Span(3, 6, "label")

        for label in ["label", "different"]:
            ## exact overlap
            span_b = Span(3, 6, label)
            assert span_a.has_overlap(span_b) == (label == "label")
            assert span_a.has_overlap(span_b, ignore_label=True)
            ## partial overlap: subsets
            span_b = Span(4, 5, label)
            assert span_a.has_overlap(span_b) == (label == "label")
            assert span_a.has_overlap(span_b, ignore_label=True)
            span_b = Span(3, 5, label)
            assert span_a.has_overlap(span_b) == (label == "label")
            assert span_a.has_overlap(span_b, ignore_label=True)
            span_b = Span(4, 6, label)
            assert span_a.has_overlap(span_b) == (label == "label")
            assert span_a.has_overlap(span_b, ignore_label=True)
            ## partial overlap: not subsets
            span_b = Span(1, 5, label)
            assert span_a.has_overlap(span_b) == (label == "label")
            assert span_a.has_overlap(span_b, ignore_label=True)
            span_b = Span(4, 8, label)
            assert span_a.has_overlap(span_b) == (label == "label")
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


class TestPageReferenceSpans:
    @patch.object(PageReferenceSpans, "_get_spans")
    def test_init(self, mock_get_spans):
        mock_get_spans.return_value = "spans list"
        page_json = {"page_id": "12345"}
        result = PageReferenceSpans(page_json)
        assert result.page_id == "12345"
        assert result.spans == "spans list"
        mock_get_spans.assert_called_once_with(page_json)

    def test_get_spans(self):
        # No spans
        page_json = {"page_id": "id", "n_excerpts": 0}
        result = PageReferenceSpans._get_spans(page_json)
        assert result == []

        # With spans
        excerpts = [
            {"start": 0, "end": 5, "poem_id": "c"},
            {"start": 3, "end": 4, "poem_id": "a"},
            {"start": 1, "end": 8, "poem_id": "b"},
            {"start": 1, "end": 3, "poem_id": "c"},
        ]
        page_json = {"page_id": "id", "n_excerpts": 4, "excerpts": excerpts}
        result = PageReferenceSpans._get_spans(page_json)
        expected_spans = [
            Span(0, 5, "c"),
            Span(1, 3, "c"),
            Span(1, 8, "b"),
            Span(3, 4, "a"),
        ]
        assert len(result) == 4
        for i, span in enumerate(result):
            assert span == expected_spans[i]


class TestPageSystemSpans:
    @patch.object(PageSystemSpans, "_get_unlabeled_spans")
    @patch.object(PageSystemSpans, "_get_labeled_spans")
    def test_init(self, mock_get_labeled_spans, mock_get_unlabeled_spans):
        mock_get_labeled_spans.return_value = "labeled spans list"
        mock_get_unlabeled_spans.return_value = "unlabeled spans list"
        page_json = {"page_id": "12345"}
        result = PageSystemSpans(page_json)
        assert result.page_id == "12345"
        assert result.labeled_spans == "labeled spans list"
        assert result.unlabeled_spans == "unlabeled spans list"
        mock_get_labeled_spans.assert_called_once_with(page_json)
        mock_get_unlabeled_spans.assert_called_once()

    def test_get_labeled_spans(self):
        # No spans
        page_json = {"page_id": "id", "n_spans": 0}
        result = PageSystemSpans._get_labeled_spans(page_json)
        assert result == []

        # With spans
        poem_spans = [
            {"page_start": 0, "page_end": 5, "ref_id": "c"},
            {"page_start": 3, "page_end": 4, "ref_id": "a"},
            {"page_start": 1, "page_end": 8, "ref_id": "b"},
            {"page_start": 1, "page_end": 3, "ref_id": "c"},
        ]
        page_json = {"page_id": "id", "n_spans": 4, "poem_spans": poem_spans}
        result = PageSystemSpans._get_labeled_spans(page_json)
        expected_spans = [
            Span(0, 5, "c"),
            Span(1, 3, "c"),
            Span(1, 8, "b"),
            Span(3, 4, "a"),
        ]
        assert len(result) == 4
        for i, span in enumerate(result):
            assert span == expected_spans[i]

    def test_get_unlabeled_spans(self):
        # No spans
        result = PageSystemSpans._get_unlabeled_spans([])
        assert result == []

        # With spans
        labeled_spans = [
            Span(0, 1, "a"),
            Span(1, 4, "a"),
            Span(2, 3, "b"),
            Span(3, 5, "d"),
            Span(9, 10, "a"),
        ]
        result = PageSystemSpans._get_unlabeled_spans(labeled_spans)
        expected_spans = [Span(0, 1, ""), Span(1, 5, ""), Span(9, 10, "")]
        assert len(result) == 3
        for i, span in enumerate(result):
            assert span == expected_spans[i]


class TestPageEvaluation:
    @patch.object(PageEvaluation, "_get_span_pairs")
    @patch.object(PageEvaluation, "_get_span_mappings")
    def test_init(self, mock_get_span_mappings, mock_get_span_pairs):
        # Page id mismatch
        page_ref_spans = NonCallableMock(page_id="a")
        page_sys_spans = NonCallableMock(page_id="b")
        with pytest.raises(
            ValueError,
            match="Reference and system spans must correspond to the same page",
        ):
            PageEvaluation(page_ref_spans, page_sys_spans)
            mock_get_span_mappings.assert_not_called()
            mock_get_span_pairs.assert_not_called()

        # Setup for non-error cases
        page_ref_spans = NonCallableMock(page_id="id", spans="spans")
        page_sys_spans = NonCallableMock(
            page_id="id",
            labeled_spans="labeled spans",
            unlabeled_spans="unlabeled spans",
        )

        # Default case (ignore_label = False)
        mock_get_span_mappings.return_value = (
            "ref span --> sys span",
            "sys span --> ref spans",
        )
        mock_get_span_pairs.return_value = "span pairs"
        result = PageEvaluation(page_ref_spans, page_sys_spans)
        assert result.ignore_label == False
        assert result.ref_spans == "spans"
        assert result.sys_spans == "labeled spans"
        assert result.ref_to_sys == "ref span --> sys span"
        assert result.sys_to_refs == "sys span --> ref spans"
        assert result.span_pairs == "span pairs"
        mock_get_span_mappings.assert_called_once()
        mock_get_span_pairs.assert_called_once()

        # Ignore labels
        mock_get_span_mappings.reset_mock()
        mock_get_span_pairs.reset_mock()
        result = PageEvaluation(page_ref_spans, page_sys_spans, ignore_label=True)
        assert result.ignore_label == True
        assert result.ref_spans == "spans"
        assert result.sys_spans == "unlabeled spans"
        assert result.ref_to_sys == "ref span --> sys span"
        assert result.sys_to_refs == "sys span --> ref spans"
        assert result.span_pairs == "span pairs"
        mock_get_span_mappings.assert_called_once()
        mock_get_span_pairs.assert_called_once()

    def test_get_span_mappings(self):
        # Simple 1-1 cases
        ref_spans = [Span(2, 5, "a"), Span(10, 15, "b")]
        sys_spans = [Span(1, 6, "a"), Span(11, 13, "c")]
        ## Label sensitive
        results = PageEvaluation._get_span_mappings(ref_spans, sys_spans, False)
        assert results[0] == [0, None]
        assert results[1] == [[0], []]
        ## Label insensitive
        results = PageEvaluation._get_span_mappings(ref_spans, sys_spans, True)
        assert results[0] == [0, 1]
        assert results[1] == [[0], [1]]

        # Best overlap
        ref_spans = [Span(3, 8, "a")]
        sys_spans = [Span(1, 4, "a"), Span(4, 7, "a"), Span(7, 10, "a")]
        for ignore_label in [False, True]:
            results = PageEvaluation._get_span_mappings(
                ref_spans, sys_spans, ignore_label
            )
            assert results[0] == [1]
            assert results[1] == [[], [0], []]

        # System span mapping to multiple reference spans
        ref_spans = [Span(2, 5, "a"), Span(7, 11, "b"), Span(18, 20, "a")]
        sys_spans = [Span(0, 25, "a")]
        ## Label sensitive
        results = PageEvaluation._get_span_mappings(ref_spans, sys_spans, False)
        assert results[0] == [0, None, 0]
        assert results[1] == [[0, 2]]
        ## Label insensitive
        results = PageEvaluation._get_span_mappings(ref_spans, sys_spans, True)
        assert results[0] == [0, 0, 0]
        assert results[1] == [[0, 1, 2]]

    def test_get_span_pairs(self):
        # Simple 1-1 case
        ref_spans = ["a", "b", "c"]
        sys_spans = ["A", "_", "C", "_"]
        sys_to_refs = [[0], [], [2], []]
        result = PageEvaluation._get_span_pairs(ref_spans, sys_spans, sys_to_refs)
        assert result == [("a", "A"), ("c", "C")]

        # Requires system span splitting
        ## System span maps to two reference spans
        ref_spans = [Span(0, 3, "a"), Span(10, 12, "b"), Span(15, 17, "b")]
        sys_spans = [Span(0, 3, "A"), Span(8, 25, "B")]
        sys_to_refs = [[0], [1, 2]]
        expected_result = [
            (ref_spans[0], Span(0, 3, "A")),
            (ref_spans[1], Span(8, 15, "B")),
            (ref_spans[2], Span(15, 25, "B")),
        ]
        result = PageEvaluation._get_span_pairs(ref_spans, sys_spans, sys_to_refs)
        assert result == expected_result

    @patch.object(Span, "overlap_factor")
    @patch.object(Span, "is_exact_match")
    def test_relevance_score(self, mock_is_exact_match, mock_overlap_factor):
        # test pairs (note the actual values are not used directly in test)
        span_pairs = [
            (Span(0, 1, "a"), Span(0, 1, "A")),
            (Span(1, 2, "b"), Span(1, 2, "B")),
            (Span(2, 3, "c"), Span(2, 3, "C")),
        ]
        # Only exact matches
        mock_is_exact_match.return_value = True
        result = PageEvaluation._relevance_score(
            span_pairs, "ignore_label", "partial_weight"
        )
        assert result == 3
        assert mock_is_exact_match.call_count == 3
        mock_overlap_factor.assert_not_called()

        # With partial matches
        ## No downweighting
        mock_is_exact_match.reset_mock()
        mock_is_exact_match.side_effect = [True, False, False]
        mock_overlap_factor.reset_mock()
        mock_overlap_factor.side_effect = [0.2, 0.3]
        result = PageEvaluation._relevance_score(span_pairs, "ignore_label", 1)
        assert result == 1.5
        assert mock_is_exact_match.call_count == 3
        assert mock_overlap_factor.call_count == 2
        ## With downweighting
        mock_is_exact_match.reset_mock()
        mock_is_exact_match.side_effect = [True, False, False]
        mock_overlap_factor.reset_mock()
        mock_overlap_factor.side_effect = [0.2, 0.3]
        result = PageEvaluation._relevance_score(span_pairs, "ignore_label", 0.5)
        assert result == 1.25

    def test_retrieved_count(self):
        # Simple matches
        assert PageEvaluation._retrieved_count([[]]) == 1
        assert PageEvaluation._retrieved_count([[1]]) == 1
        assert PageEvaluation._retrieved_count([[3], []]) == 2
        assert PageEvaluation._retrieved_count([[], [1], [5]]) == 3

        # System span covers matches multiple reference spans
        assert PageEvaluation._retrieved_count([[3, 5]]) == 2
        assert PageEvaluation._retrieved_count([[], [1, 2, 3], [6]]) == 5

    @patch.object(PageEvaluation, "_relevance_score")
    @patch.object(PageEvaluation, "_retrieved_count")
    @patch.object(PageEvaluation, "_get_span_pairs")
    @patch.object(PageEvaluation, "_get_span_mappings")
    def test_precision(
        self, mock_span_mappings, mock_span_pairs, mock_retrieved_count, mock_relevance
    ):
        mock_span_mappings.return_value = "", "sys_to_refs"
        mock_span_pairs.return_value = "span_pairs"
        # Edge case: No system spans
        page_sys_spans = NonCallableMock(page_id="id", labeled_spans=[])
        ## With reference spans
        page_ref_spans = NonCallableMock(page_id="id", spans=["s"])
        page_eval = PageEvaluation(page_ref_spans, page_sys_spans)
        assert page_eval.precision() == 0
        mock_retrieved_count.assert_not_called()
        mock_relevance.assert_not_called()
        ## No reference spans
        page_ref_spans = NonCallableMock(page_id="id", spans=[])
        page_eval = PageEvaluation(page_ref_spans, page_sys_spans)
        assert page_eval.precision() == 1
        mock_retrieved_count.assert_not_called()
        mock_relevance.assert_not_called()

        # With system spans
        mock_retrieved_count.return_value = 2
        mock_relevance.return_value = 0.5
        page_sys_spans = NonCallableMock(page_id="id", labeled_spans=["s"] * 2)
        page_eval = PageEvaluation(
            page_ref_spans, page_sys_spans, ignore_label="ignore_label"
        )
        assert page_eval.precision(partial_match_weight="weight") == 0.5 / 2
        mock_retrieved_count.assert_called_once_with("sys_to_refs")
        mock_relevance.assert_called_once_with("span_pairs", "ignore_label", "weight")

        mock_retrieved_count.reset_mock()
        mock_retrieved_count.return_value = 7
        mock_relevance.reset_mock()
        mock_relevance.return_value = 4.5
        page_sys_spans = NonCallableMock(page_id="id", labeled_spans=["s"] * 4)
        page_eval = PageEvaluation(
            page_ref_spans, page_sys_spans, ignore_label="ignore_label"
        )
        assert page_eval.precision(partial_match_weight="weight") == 4.5 / 7
        mock_retrieved_count.assert_called_once_with("sys_to_refs")
        mock_relevance.assert_called_once_with("span_pairs", "ignore_label", "weight")

    @patch.object(PageEvaluation, "_relevance_score")
    @patch.object(PageEvaluation, "_get_span_pairs")
    @patch.object(PageEvaluation, "_get_span_mappings")
    def test_recall(self, mock_span_mappings, mock_span_pairs, mock_relevance):
        # Mocking needed for PageEvaluation initialization
        mock_span_mappings.return_value = "", ""
        mock_span_pairs.return_value = "span_pairs"

        # Edge case: No reference spans
        page_ref_spans = NonCallableMock(page_id="id", spans=[])
        ## With system spans
        page_sys_spans = NonCallableMock(page_id="id", labeled_spans=["s"])
        page_eval = PageEvaluation(page_ref_spans, page_sys_spans)
        assert page_eval.recall() == 0
        mock_relevance.assert_not_called()
        ## No system spans
        page_sys_spans = NonCallableMock(page_id="id", labeled_spans=[])
        page_eval = PageEvaluation(page_ref_spans, page_sys_spans)
        assert page_eval.recall() == 1
        mock_relevance.assert_not_called()

        # With reference spans
        mock_relevance.return_value = 0.5
        page_ref_spans = NonCallableMock(page_id="id", spans=["a"])
        page_eval = PageEvaluation(
            page_ref_spans, page_sys_spans, ignore_label="ignore_label"
        )
        assert page_eval.recall(partial_match_weight="weight") == 0.5 / 1
        mock_relevance.assert_called_once_with("span_pairs", "ignore_label", "weight")

        mock_relevance.reset_mock()
        mock_relevance.return_value = 2.1
        page_ref_spans = NonCallableMock(page_id="id", spans=["a"] * 3)
        page_eval = PageEvaluation(
            page_ref_spans, page_sys_spans, ignore_label="ignore_label"
        )
        assert page_eval.recall(partial_match_weight="weight") == 2.1 / 3
        mock_relevance.assert_called_once_with("span_pairs", "ignore_label", "weight")
