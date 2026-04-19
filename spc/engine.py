from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np

SPC_METRIC_COLUMN_MAP: dict[str, str] = {
    "wafer_yield": "wafer_yield",
    "mean_film_thickness_nm": "mean_thickness",
    "mean_defect_density_cm2": "mean_defect_density",
}


@dataclass
class SPCPoint:
    index: int
    timestamp: datetime
    value: float
    cl: float
    ucl: float
    lcl: float
    ewma: float
    ewma_ucl: float
    ewma_lcl: float
    violation_ids: list[int]


@dataclass
class SPCViolation:
    rule_id: int
    rule_name: str
    severity: Literal["low", "medium", "high"]
    index: int
    timestamp: datetime
    value: float
    message: str
    evidence_indices: list[int]


RULE_NAMES = {
    1: "One point beyond 3σ",
    2: "Nine points on same side",
    3: "Six points trending one direction",
    4: "Fourteen points alternating",
    5: "Two of three beyond 2σ",
    6: "Four of five beyond 1σ",
    7: "Fifteen points within 1σ",
    8: "Eight points outside 1σ",
}


def _control_limits(values: np.ndarray, baseline_n: int, sigma_floor: float) -> tuple[float, float, float, float]:
    n = min(len(values), max(2, baseline_n))
    base = values[:n]
    cl = float(np.mean(base))
    sigma = float(np.std(base, ddof=1)) if len(base) > 1 else 0.0
    sigma = max(sigma, sigma_floor)
    return cl, cl + 3.0 * sigma, cl - 3.0 * sigma, sigma


def _ewma(values: np.ndarray, cl: float, sigma: float, lam: float, L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.zeros_like(values, dtype=np.float64)
    ucls = np.zeros_like(values, dtype=np.float64)
    lcls = np.zeros_like(values, dtype=np.float64)
    if len(values) == 0:
        return z, ucls, lcls
    z[0] = cl
    sigma = max(1e-9, sigma)
    for i, x in enumerate(values):
        if i > 0:
            z[i] = lam * x + (1.0 - lam) * z[i - 1]
        t = i + 1
        fac = math.sqrt((lam / (2.0 - lam)) * (1.0 - (1.0 - lam) ** (2 * t)))
        d = L * sigma * fac
        ucls[i] = cl + d
        lcls[i] = cl - d
    return z, ucls, lcls


def _nelson_hits(values: np.ndarray, cl: float, sigma: float) -> dict[int, list[list[int]]]:
    sigma = max(1e-9, sigma)
    z = (values - cl) / sigma
    n = len(z)
    hits: dict[int, list[list[int]]] = {i: [] for i in range(1, 9)}

    for i in range(n):
        if abs(z[i]) > 3:
            hits[1].append([i])
    for i in range(8, n):
        w = z[i - 8 : i + 1]
        if np.all(w > 0) or np.all(w < 0):
            hits[2].append(list(range(i - 8, i + 1)))
    for i in range(5, n):
        w = z[i - 5 : i + 1]
        if np.all(np.diff(w) > 0) or np.all(np.diff(w) < 0):
            hits[3].append(list(range(i - 5, i + 1)))
    for i in range(13, n):
        w = z[i - 13 : i + 1]
        if np.all(np.sign(w[1:]) * np.sign(w[:-1]) < 0):
            hits[4].append(list(range(i - 13, i + 1)))
    for i in range(2, n):
        w = z[i - 2 : i + 1]
        if np.sum(w > 2) >= 2 or np.sum(w < -2) >= 2:
            hits[5].append(list(range(i - 2, i + 1)))
    for i in range(4, n):
        w = z[i - 4 : i + 1]
        if np.sum(w > 1) >= 4 or np.sum(w < -1) >= 4:
            hits[6].append(list(range(i - 4, i + 1)))
    for i in range(14, n):
        if np.all(np.abs(z[i - 14 : i + 1]) < 1):
            hits[7].append(list(range(i - 14, i + 1)))
    for i in range(7, n):
        if np.all(np.abs(z[i - 7 : i + 1]) > 1):
            hits[8].append(list(range(i - 7, i + 1)))
    return hits


def analyze_series(
    values: np.ndarray,
    timestamps: list[datetime],
    *,
    baseline_n: int,
    ewma_lambda: float,
    ewma_L: float,
    sigma_floor: float,
    ewma_enabled: bool = True,
) -> tuple[list[SPCPoint], list[SPCViolation]]:
    if len(values) != len(timestamps):
        raise ValueError("values/timestamps length mismatch")
    if len(values) == 0:
        return [], []

    cl, ucl, lcl, sigma = _control_limits(values, baseline_n, sigma_floor)
    if ewma_enabled:
        ewma_vals, ewma_ucl, ewma_lcl = _ewma(values, cl, sigma, ewma_lambda, ewma_L)
    else:
        ewma_vals = np.asarray([cl] * len(values), dtype=np.float64)
        ewma_ucl = np.asarray([ucl] * len(values), dtype=np.float64)
        ewma_lcl = np.asarray([lcl] * len(values), dtype=np.float64)

    hits = _nelson_hits(values, cl, sigma)
    by_idx: dict[int, list[int]] = {}
    violations: list[SPCViolation] = []
    for rid, windows in hits.items():
        sev: Literal["low", "medium", "high"] = "medium" if rid in {2, 3, 6, 8} else "high" if rid in {1, 5} else "low"
        for window in windows:
            idx = window[-1]
            by_idx.setdefault(idx, []).append(rid)
            violations.append(
                SPCViolation(
                    rule_id=rid,
                    rule_name=RULE_NAMES[rid],
                    severity=sev,
                    index=idx,
                    timestamp=timestamps[idx],
                    value=float(values[idx]),
                    message=f"{RULE_NAMES[rid]} detected",
                    evidence_indices=window,
                )
            )

    if ewma_enabled:
        for i in range(len(values)):
            if ewma_vals[i] > ewma_ucl[i] or ewma_vals[i] < ewma_lcl[i]:
                by_idx.setdefault(i, []).append(100)
                violations.append(
                    SPCViolation(
                        rule_id=100,
                        rule_name="EWMA limit breach",
                        severity="high",
                        index=i,
                        timestamp=timestamps[i],
                        value=float(values[i]),
                        message="EWMA crossed control limit",
                        evidence_indices=[i],
                    )
                )

    points = [
        SPCPoint(
            index=i,
            timestamp=timestamps[i],
            value=float(values[i]),
            cl=cl,
            ucl=ucl,
            lcl=lcl,
            ewma=float(ewma_vals[i]),
            ewma_ucl=float(ewma_ucl[i]),
            ewma_lcl=float(ewma_lcl[i]),
            violation_ids=sorted(by_idx.get(i, [])),
        )
        for i in range(len(values))
    ]
    return points, violations


def compute_label_metrics(points: list[SPCPoint], labels: list[bool]) -> dict[str, float]:
    if len(points) != len(labels):
        raise ValueError("points/labels length mismatch")
    if not points:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "lead_time_runs": 0.0}

    pred = np.array([len(p.violation_ids) > 0 for p in points], dtype=bool)
    y = np.array(labels, dtype=bool)
    tp = float(np.sum(pred & y))
    fp = float(np.sum(pred & ~y))
    fn = float(np.sum(~pred & y))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    first_alert = next((i for i, p in enumerate(pred) if p), None)
    first_label = next((i for i, l in enumerate(y) if l), None)
    lead = float((first_alert - first_label) if first_alert is not None and first_label is not None else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, "lead_time_runs": lead}
