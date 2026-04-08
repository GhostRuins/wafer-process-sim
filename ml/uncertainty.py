"""Confidence interval aggregation and calibration diagnostics."""

from __future__ import annotations

import numpy as np
from scipy import stats


def combine_confidence_intervals(
    lgbm_lower: float,
    lgbm_upper: float,
    gp_mean: float,
    gp_std: float,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Blend LightGBM quantiles with Gaussian CI using nonnegative weights.

    weights order: [w_lgb_low, w_lgb_mid_ignored, w_lgb_high, w_xgb_ignored, w_gp_mean, w_gp_std]
    We use w_lgb for bounds and w_gp for normal approximation tails.
    """
    # Ridge coefficients may be negative; magnitude encodes reliance on that term.
    w = np.abs(weights.astype(np.float64))
    w = np.maximum(w, 0.0)
    if w.sum() < 1e-12:
        w = np.ones_like(w) / len(w)
    wl, _, wh, _, wgm, wgs = w[:6]
    gp_lo = gp_mean - 1.96 * max(gp_std, 1e-6)
    gp_hi = gp_mean + 1.96 * max(gp_std, 1e-6)
    wsum_lo = wl + wgm + wgs
    wsum_hi = wh + wgm + wgs
    ci_lo = (wl * lgbm_lower + (wgm + wgs) * gp_lo) / max(wsum_lo, 1e-9)
    ci_hi = (wh * lgbm_upper + (wgm + wgs) * gp_hi) / max(wsum_hi, 1e-9)
    return float(ci_lo), float(ci_hi)


def expected_calibration_error(
    y_true: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Mean squared gap between empirical coverage and Gaussian nominal coverage per bin."""
    if len(y_true) < n_bins * 5:
        return 0.0
    order = np.argsort(pred_std)
    y_t = y_true[order]
    m = pred_mean[order]
    s = np.maximum(pred_std[order], 1e-9)
    z = (y_t - m) / s
    bins = np.array_split(np.arange(len(y_t)), n_bins)
    ece = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        prop_in_95 = np.mean(np.abs(z[b]) <= 1.96)
        ece += len(b) * abs(prop_in_95 - 0.95)
    return float(ece / len(y_true))


def reliability_diagram_data(
    y_true: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    *,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Bucketed data for reliability / coverage plot."""
    if len(y_true) < 2:
        return []
    order = np.argsort(pred_std)
    y_t = y_true[order]
    m = pred_mean[order]
    s = np.maximum(pred_std[order], 1e-9)
    z = (y_t - m) / s
    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    rows: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (s >= lo) & (s <= hi)
        else:
            mask = (s >= lo) & (s < hi)
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "mean_sigma": float(np.mean(s[mask])),
                "empirical_coverage_95": float(np.mean(np.abs(z[mask]) <= 1.96)),
                "count": int(np.sum(mask)),
            }
        )
    return rows


def interval_coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of y inside [lo, hi]."""
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def pass_probability_gaussian(mean: float, std: float, threshold: float = 0.85) -> float:
    """P(Y > threshold) for Y ~ N(mean, std^2)."""
    std = max(float(std), 1e-6)
    return float(1.0 - stats.norm.cdf((threshold - mean) / std))
