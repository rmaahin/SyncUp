"""Peer-review aggregation and bias detection.

Pure-Python — no LLM. Operates on lists of ``PeerReview`` records and produces
``PeerReviewSummary`` objects (per-student averages and stdev) and a list of
``BiasFlag`` entries (inflation / outlier reviewer / targeted-low / retaliation).
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Iterable

from agents.peer_review import DIMENSION_KEYS
from state.schema import BiasFlag, PeerReview, PeerReviewSummary


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_peer_reviews(
    reviews: Iterable[PeerReview],
    student_ids: Iterable[str],
) -> dict[str, PeerReviewSummary]:
    """Compute per-student review summaries.

    Args:
        reviews: All submitted peer reviews.
        student_ids: All students in the project (so students with zero
            reviews still appear in the result).

    Returns:
        Mapping student_id → PeerReviewSummary.
    """
    reviews_list = list(reviews)
    by_reviewee: dict[str, list[PeerReview]] = defaultdict(list)
    for r in reviews_list:
        by_reviewee[r.reviewee_id].append(r)

    summaries: dict[str, PeerReviewSummary] = {}
    for sid in student_ids:
        rs = by_reviewee.get(sid, [])
        if not rs:
            summaries[sid] = PeerReviewSummary(student_id=sid)
            continue

        avg_per_dim: dict[str, float] = {}
        std_per_dim: dict[str, float] = {}
        for dim in DIMENSION_KEYS:
            values = [r.ratings[dim] for r in rs if dim in r.ratings]
            if not values:
                continue
            avg_per_dim[dim] = statistics.fmean(values)
            std_per_dim[dim] = statistics.pstdev(values) if len(values) >= 2 else 0.0

        overall = (
            statistics.fmean(avg_per_dim.values()) if avg_per_dim else 0.0
        )
        summaries[sid] = PeerReviewSummary(
            student_id=sid,
            avg_per_dimension=avg_per_dim,
            overall_avg=overall,
            std_dev_per_dimension=std_per_dim,
            review_count=len(rs),
        )
    return summaries


# ---------------------------------------------------------------------------
# Bias detection
# ---------------------------------------------------------------------------


_RETALIATION_THRESHOLD: float = 2.0  # both sides ≤ this average → retaliation


def _all_ratings(review: PeerReview) -> list[int]:
    return [v for k, v in review.ratings.items() if k in DIMENSION_KEYS]


def _reviewer_overall_mean(
    reviewer_id: str, by_reviewer: dict[str, list[PeerReview]]
) -> float:
    """Mean across all (reviewee × dimension) ratings issued by a reviewer."""
    flat: list[int] = []
    for r in by_reviewer.get(reviewer_id, []):
        flat.extend(_all_ratings(r))
    return statistics.fmean(flat) if flat else 0.0


def _pair_mean(
    reviewer_id: str,
    reviewee_id: str,
    by_reviewer: dict[str, list[PeerReview]],
) -> float | None:
    """Mean rating reviewer gave reviewee across dimensions; None if no review."""
    for r in by_reviewer.get(reviewer_id, []):
        if r.reviewee_id == reviewee_id:
            vals = _all_ratings(r)
            return statistics.fmean(vals) if vals else None
    return None


def detect_bias(
    reviews: Iterable[PeerReview],
    student_ids: Iterable[str],
) -> list[BiasFlag]:
    """Detect bias patterns in peer reviews.

    Detection order: inflation → outlier reviewer → targeted-low → retaliation.
    Returns a flat list of ``BiasFlag`` entries.
    """
    reviews_list = list(reviews)
    student_id_list = list(student_ids)
    if not reviews_list:
        return []

    by_reviewer: dict[str, list[PeerReview]] = defaultdict(list)
    for r in reviews_list:
        by_reviewer[r.reviewer_id].append(r)

    flags: list[BiasFlag] = []

    # --- (1) INFLATION: every rating issued is exactly 5 ---
    inflated_reviewers: set[str] = set()
    for rid, rs in by_reviewer.items():
        all_vals = [v for r in rs for v in _all_ratings(r)]
        if all_vals and all(v == 5 for v in all_vals):
            inflated_reviewers.add(rid)
            flags.append(BiasFlag(
                flag_type="inflation",
                reviewer_id=rid,
                description=f"{rid} gave every teammate a 5 on every dimension.",
                severity="medium",
            ))

    # --- (2) OUTLIER REVIEWER (z-score > 2) ---
    reviewer_means: dict[str, float] = {
        rid: _reviewer_overall_mean(rid, by_reviewer) for rid in by_reviewer
    }
    outlier_reviewers: set[str] = set()
    population_mu: float = 0.0
    population_sigma: float = 0.0
    if len(reviewer_means) >= 3:
        means_vals = list(reviewer_means.values())
        population_mu = statistics.fmean(means_vals)
        population_sigma = statistics.pstdev(means_vals)
        if population_sigma > 0:
            for rid, m in reviewer_means.items():
                if abs(m - population_mu) > 2 * population_sigma:
                    outlier_reviewers.add(rid)
                    direction = "harsher" if m < population_mu else "more lenient"
                    severity = "high" if rid in inflated_reviewers else "medium"
                    flags.append(BiasFlag(
                        flag_type="outlier_reviewer",
                        reviewer_id=rid,
                        description=(
                            f"{rid} rates teammates {direction} than peers "
                            f"(avg {m:.2f} vs population {population_mu:.2f})."
                        ),
                        severity=severity,
                    ))

    # --- (3) TARGETED LOW: pair singled-out, reviewer otherwise normal ---
    # Build reviewers-of-each-reviewee for fast lookup
    reviewers_of: dict[str, list[str]] = defaultdict(list)
    for rid, rs in by_reviewer.items():
        for r in rs:
            reviewers_of[r.reviewee_id].append(rid)

    for rvid in student_id_list:
        rids_of_v = reviewers_of.get(rvid, [])
        if len(rids_of_v) < 3:
            continue  # need ≥3 reviewers to detect a singled-out pair
        for rid in rids_of_v:
            pm = _pair_mean(rid, rvid, by_reviewer)
            if pm is None:
                continue
            others = [
                _pair_mean(o, rvid, by_reviewer)
                for o in rids_of_v
                if o != rid
            ]
            others_clean = [v for v in others if v is not None]
            if len(others_clean) < 2:
                continue
            mu_v = statistics.fmean(others_clean)
            sigma_v = statistics.pstdev(others_clean)
            if sigma_v == 0:
                continue
            if pm >= mu_v - 2 * sigma_v:
                continue  # not singled out
            # Skip if this reviewer is already caught by the outlier rule —
            # don't double-flag.
            if rid in outlier_reviewers:
                continue
            flags.append(BiasFlag(
                flag_type="targeted_low",
                reviewer_id=rid,
                reviewee_id=rvid,
                description=(
                    f"{rid} rated {rvid} {pm:.2f} avg while peers rated {rvid} "
                    f"{mu_v:.2f} avg."
                ),
                severity="high",
            ))

    # --- (4) RETALIATION: both A→B and B→A means ≤ threshold ---
    seen_pairs: set[tuple[str, str]] = set()
    reviewer_ids = list(by_reviewer.keys())
    for a in reviewer_ids:
        for b in reviewer_ids:
            if a == b:
                continue
            key = tuple(sorted((a, b)))
            if key in seen_pairs:
                continue
            ab = _pair_mean(a, b, by_reviewer)
            ba = _pair_mean(b, a, by_reviewer)
            if ab is None or ba is None:
                continue
            if ab <= _RETALIATION_THRESHOLD and ba <= _RETALIATION_THRESHOLD:
                seen_pairs.add(key)
                flags.append(BiasFlag(
                    flag_type="retaliation",
                    reviewer_id=a,
                    reviewee_id=b,
                    description=(
                        f"Mutual low ratings: {a}→{b} {ab:.2f}, {b}→{a} {ba:.2f}."
                    ),
                    severity="high",
                ))
                flags.append(BiasFlag(
                    flag_type="retaliation",
                    reviewer_id=b,
                    reviewee_id=a,
                    description=(
                        f"Mutual low ratings: {b}→{a} {ba:.2f}, {a}→{b} {ab:.2f}."
                    ),
                    severity="high",
                ))

    return flags
