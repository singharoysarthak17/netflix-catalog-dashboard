"""Evaluation math and the offline leave-one-out run."""

from __future__ import annotations

import pytest

from src.evaluate import EvalResult, summary_table


def test_lift_is_relative_to_the_baseline():
    strong = EvalResult(users=10, k=10, precision=0.04, recall=0.4, ndcg=0.3,
                        baseline_precision=0.02, baseline_ndcg=0.1,
                        coverage=0.5, novelty=10.0)
    assert strong.lift == pytest.approx(2.0)

    no_baseline = EvalResult(users=10, k=10, precision=0.0, recall=0.0, ndcg=0.0,
                             baseline_precision=0.0, baseline_ndcg=0.0,
                             coverage=0.0, novelty=0.0)
    assert no_baseline.lift == float("inf")


def test_summary_table_has_the_expected_rows():
    table = summary_table()
    assert {"Metric", "This model", "Popularity baseline", "Lift"} <= set(table.columns)
    assert any("Precision@10" in str(m) for m in table["Metric"])


@pytest.mark.slow
def test_item_item_cf_beats_a_popularity_baseline():
    """The headline claim: CF must actually outperform 'just recommend the
    most-rated films', otherwise the collaborative component is decoration."""
    from src.evaluate import get_evaluation

    result = get_evaluation()
    assert result.users > 500
    assert result.precision > result.baseline_precision * 1.5
    assert result.ndcg > result.baseline_ndcg
    assert 0.0 < result.coverage <= 1.0
