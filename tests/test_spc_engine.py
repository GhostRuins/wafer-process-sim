from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from spc.engine import analyze_series, compute_label_metrics


def _timestamps(n: int):
    t0 = datetime(2026, 1, 1)
    return [t0 + timedelta(minutes=i) for i in range(n)]


def test_rule_one_outlier_detection():
    vals = np.array([1.0] * 40 + [10.0], dtype=np.float64)
    points, violations = analyze_series(
        vals,
        _timestamps(len(vals)),
        baseline_n=30,
        ewma_lambda=0.2,
        ewma_L=3.0,
        sigma_floor=0.01,
    )
    assert len(points) == len(vals)
    assert any(v.rule_id == 1 for v in violations)


def test_label_metrics_shape():
    vals = np.array([1.0] * 30 + [1.2, 1.3, 1.4, 2.0], dtype=np.float64)
    points, _ = analyze_series(
        vals,
        _timestamps(len(vals)),
        baseline_n=20,
        ewma_lambda=0.2,
        ewma_L=3.0,
        sigma_floor=0.01,
    )
    labels = [False] * 32 + [True, True]
    out = compute_label_metrics(points, labels)
    assert set(out.keys()) == {"precision", "recall", "f1", "lead_time_runs"}
