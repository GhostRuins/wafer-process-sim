"""Monotonicity audit for ensemble predictions (physics QC)."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ml.feature_engineering import (
    ALL_FEATURE_NAMES,
    MONO_DECREASE,
    MONO_INCREASE,
)


class MonotonicityViolationWarning(UserWarning):
    """Stacking or GP may break local monotonicity despite XGB constraints."""


@dataclass
class MonotonicityAuditResult:
    violations_pct: float
    violated_features: list[str]
    details: dict[str, dict[str, float]]


def audit_monotonicity_arrays(
    predict_fn,
    X_train: np.ndarray,
    *,
    n_grid: int = 200,
    increase_names: list[str] | None = None,
    decrease_names: list[str] | None = None,
) -> MonotonicityAuditResult:
    """Sweep one column of the model matrix at a time (median baseline).

    predict_fn(X: np.ndarray shape (n, n_features)) -> np.ndarray (n,)
    """
    inc = increase_names or MONO_INCREASE
    dec = decrease_names or MONO_DECREASE
    med = np.median(X_train, axis=0)
    mins = np.min(X_train, axis=0)
    maxs = np.max(X_train, axis=0)

    violated: list[str] = []
    details: dict[str, dict[str, float]] = {}
    total = 0
    bad = 0

    for name in inc + dec:
        if name not in ALL_FEATURE_NAMES:
            continue
        j = ALL_FEATURE_NAMES.index(name)
        if maxs[j] <= mins[j] + 1e-12:
            continue
        grid = np.linspace(mins[j], maxs[j], n_grid)
        Xg = np.tile(med, (len(grid), 1))
        Xg[:, j] = grid
        preds = predict_fn(Xg)
        diffs = np.diff(preds)
        if name in inc:
            frac_bad = float(np.mean(diffs < -1e-4))
            ok = frac_bad < 0.05
        else:
            frac_bad = float(np.mean(diffs > 1e-4))
            ok = frac_bad < 0.05
        total += 1
        if not ok:
            bad += 1
            violated.append(name)
        details[name] = {
            "frac_bad_steps": frac_bad,
            "pred_min": float(np.min(preds)),
            "pred_max": float(np.max(preds)),
        }

    violations_pct = (bad / total * 100.0) if total else 0.0
    return MonotonicityAuditResult(
        violations_pct=violations_pct,
        violated_features=violated,
        details=details,
    )


def audit_xgboost_monotonicity(
    xgb_predict: Callable[[np.ndarray], np.ndarray],
    X_train: np.ndarray,
    *,
    n_grid: int = 120,
) -> MonotonicityAuditResult:
    """Physics QA on the constrained booster alone (should respect monotone_constraints)."""

    return audit_monotonicity_arrays(xgb_predict, X_train, n_grid=n_grid)


def warn_if_audit_failed(result: MonotonicityAuditResult) -> None:
    if result.violated_features:
        warnings.warn(
            f"Monotonicity audit: {result.violations_pct:.1f}% constrained directions "
            f"show violations on ensemble: {result.violated_features}",
            MonotonicityViolationWarning,
            stacklevel=2,
        )
