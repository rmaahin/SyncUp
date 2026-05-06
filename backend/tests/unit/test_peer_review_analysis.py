"""Unit tests for peer-review aggregation and bias detection."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agents.peer_review import DIMENSION_KEYS
from services.peer_review_analysis import aggregate_peer_reviews, detect_bias
from state.schema import PeerReview

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _full_ratings(value: int) -> dict[str, int]:
    return {k: value for k in DIMENSION_KEYS}


def _review(
    reviewer: str, reviewee: str, ratings: dict[str, int] | int
) -> PeerReview:
    if isinstance(ratings, int):
        ratings = _full_ratings(ratings)
    return PeerReview(
        reviewer_id=reviewer,
        reviewee_id=reviewee,
        ratings=ratings,
        comments={},
        submitted_at=NOW,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_uniform_ratings(self) -> None:
        reviews = [
            _review("r1", "x", 4),
            _review("r2", "x", 4),
            _review("r3", "x", 4),
        ]
        out = aggregate_peer_reviews(reviews, ["x"])
        s = out["x"]
        assert s.review_count == 3
        assert s.overall_avg == pytest.approx(4.0)
        for d in DIMENSION_KEYS:
            assert s.avg_per_dimension[d] == pytest.approx(4.0)
            assert s.std_dev_per_dimension[d] == pytest.approx(0.0)

    def test_uneven_review_counts(self) -> None:
        reviews = [
            _review("r1", "x", 5),
            _review("r2", "x", 3),
            _review("r3", "x", 5),
            _review("r1", "y", 4),
        ]
        out = aggregate_peer_reviews(reviews, ["x", "y"])
        assert out["x"].review_count == 3
        assert out["y"].review_count == 1
        assert out["x"].overall_avg == pytest.approx(13 / 3)
        assert out["y"].overall_avg == pytest.approx(4.0)

    def test_empty_reviews_returns_blank_summaries(self) -> None:
        out = aggregate_peer_reviews([], ["a", "b"])
        assert out["a"].review_count == 0
        assert out["a"].overall_avg == 0.0
        assert out["b"].avg_per_dimension == {}


# ---------------------------------------------------------------------------
# Bias detection
# ---------------------------------------------------------------------------


class TestBiasInflation:
    def test_all_fives_flagged(self) -> None:
        reviews = [
            _review("r1", "x", 5),
            _review("r1", "y", 5),
            _review("r1", "z", 5),
        ]
        flags = detect_bias(reviews, ["r1", "x", "y", "z"])
        assert any(f.flag_type == "inflation" and f.reviewer_id == "r1" for f in flags)

    def test_mixed_ratings_not_inflation(self) -> None:
        reviews = [
            _review("r1", "x", 5),
            _review("r1", "y", 4),
            _review("r1", "z", 5),
        ]
        flags = detect_bias(reviews, ["r1", "x", "y", "z"])
        assert not any(f.flag_type == "inflation" for f in flags)


class TestOutlierReviewer:
    def test_one_harsh_reviewer_flagged(self) -> None:
        # 6 reviewers — 1 harsh, 5 normal. With 4 reviewers the 2σ threshold
        # is too generous to flag a single outlier; 6 gives a clear signal.
        reviews = []
        for r in ("r1", "r2", "r3", "r4", "r5", "r6"):
            for v in ("a", "b", "c"):
                reviews.append(_review(r, v, 1 if r == "r1" else 4))
        flags = detect_bias(
            reviews, ["r1", "r2", "r3", "r4", "r5", "r6", "a", "b", "c"]
        )
        outliers = [f for f in flags if f.flag_type == "outlier_reviewer"]
        assert any(f.reviewer_id == "r1" for f in outliers)


class TestTargetedLow:
    def test_targeted_low_detected_and_no_outlier_double_flag(self) -> None:
        # All others rate Y in the 4-ish range (with mild variation so
        # sigma_v > 0); R1 gives Y a 1. R1's overall mean stays near peer mean.
        reviews = [
            _review("r1", "y", 1),
            _review("r1", "a", 4), _review("r1", "b", 4), _review("r1", "c", 4),
            # Other reviewers' ratings of Y vary slightly around 4
            _review("r2", "y", _full_ratings(4)),
            _review("r3", "y", _full_ratings(4)),
            _review("r4", "y", _full_ratings(4)),
            _review("r5", "y", _full_ratings(5)),
            _review("r6", "y", _full_ratings(3)),
            # And spread out other reviews so reviewer means are similar
            _review("r2", "a", 4), _review("r2", "b", 4), _review("r2", "c", 4),
            _review("r3", "a", 4), _review("r3", "b", 4), _review("r3", "c", 4),
            _review("r4", "a", 4), _review("r4", "b", 4), _review("r4", "c", 4),
            _review("r5", "a", 4), _review("r5", "b", 4), _review("r5", "c", 4),
            _review("r6", "a", 4), _review("r6", "b", 4), _review("r6", "c", 4),
        ]
        flags = detect_bias(
            reviews,
            ["r1", "r2", "r3", "r4", "r5", "r6", "y", "a", "b", "c"],
        )
        targeted = [f for f in flags if f.flag_type == "targeted_low"]
        assert any(
            f.reviewer_id == "r1" and f.reviewee_id == "y" for f in targeted
        )
        # R1 must not also be flagged as outlier_reviewer
        assert not any(
            f.flag_type == "outlier_reviewer" and f.reviewer_id == "r1"
            for f in flags
        )


class TestRetaliation:
    def test_mutual_low_emits_two_flags(self) -> None:
        reviews = [
            # A and B mutually rate low
            _review("a", "b", _full_ratings(2)),
            _review("b", "a", _full_ratings(2)),
            # Other normal reviews so retaliation is the only signal
            _review("a", "c", 4),
            _review("b", "c", 4),
            _review("c", "a", 4),
            _review("c", "b", 4),
        ]
        flags = detect_bias(reviews, ["a", "b", "c"])
        ret = [f for f in flags if f.flag_type == "retaliation"]
        # One per direction
        assert len(ret) == 2
        directions = {(f.reviewer_id, f.reviewee_id) for f in ret}
        assert directions == {("a", "b"), ("b", "a")}


class TestNoBias:
    def test_uniform_non_5_returns_no_flags(self) -> None:
        reviews = [
            _review("a", "b", 4), _review("a", "c", 4),
            _review("b", "a", 4), _review("b", "c", 4),
            _review("c", "a", 4), _review("c", "b", 4),
        ]
        flags = detect_bias(reviews, ["a", "b", "c"])
        assert flags == []

    def test_empty_reviews_returns_empty(self) -> None:
        assert detect_bias([], ["a", "b"]) == []
