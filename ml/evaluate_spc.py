"""Evaluate SPC detection quality against anomaly labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spc.engine import analyze_series, compute_label_metrics

METRIC_COLUMN_MAP: dict[str, str] = {
    "wafer_yield": "wafer_yield",
    "mean_film_thickness_nm": "mean_thickness",
    "mean_defect_density_cm2": "mean_defect_density",
}


def evaluate_spc(data_path: Path, metric_col: str = "wafer_yield", baseline_n: int = 50) -> dict[str, float]:
    df = pd.read_parquet(data_path).sort_values("timestamp").reset_index(drop=True)
    resolved_col = METRIC_COLUMN_MAP.get(metric_col, metric_col)
    if resolved_col not in df.columns:
        raise ValueError(f"metric column not found: {metric_col} -> {resolved_col}")
    values = df[resolved_col].to_numpy(dtype=np.float64)
    timestamps = list(pd.to_datetime(df["timestamp"]).dt.to_pydatetime())
    labels = df["anomaly_type"].notna().tolist() if "anomaly_type" in df.columns else [False] * len(df)
    points, _ = analyze_series(
        values=values,
        timestamps=timestamps,
        baseline_n=baseline_n,
        ewma_lambda=0.2,
        ewma_L=3.0,
        sigma_floor=1e-6,
    )
    metrics = compute_label_metrics(points, labels)
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate SPC alerts against anomaly labels.")
    p.add_argument("--data", type=Path, required=True, help="Path to wafer_summary parquet")
    p.add_argument("--metric-col", type=str, default="wafer_yield", help="Metric name (API alias or raw parquet column).")
    p.add_argument("--baseline-n", type=int, default=50)
    args = p.parse_args()
    evaluate_spc(args.data, metric_col=args.metric_col, baseline_n=args.baseline_n)


if __name__ == "__main__":
    main()
