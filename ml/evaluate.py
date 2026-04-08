"""Evaluate a saved yield ensemble on a wafer_summary parquet file."""
import os
plots_dir = os.path.join("ml", "plots")
os.makedirs(plots_dir, exist_ok=True)

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from ml.feature_engineering import FeatureEngineeringPipeline
from ml.yield_model import YieldEnsemble, load_ensemble
import numpy as np
import argparse
from pathlib import Path

def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, p)))


def evaluate(artifacts_dir: Path, data_path: Path, yield_threshold: float = 0.85) -> dict:
    ens = load_ensemble(artifacts_dir)
    df = pd.read_parquet(data_path).sort_values("timestamp").reset_index(drop=True)
    y = df["wafer_yield"].to_numpy(dtype=np.float64)

    fe: FeatureEngineeringPipeline = ens.feature_pipe
    X = fe.transform(df)
    Xg = fe.transform_gp(df)
    pred = ens.predict_point(X, Xg)

    metrics = {
        "rmse": _rmse(y, pred),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }
    if len(np.unique((y > yield_threshold).astype(int))) > 1:
        metrics["auc_roc"] = float(roc_auc_score((y > yield_threshold).astype(int), pred))
    else:
        metrics["auc_roc"] = float("nan")

    nom = df["anomaly_type"].isna().to_numpy()
    if nom.any() and (~nom).any():
        metrics["mean_pred_nominal"] = float(np.mean(pred[nom]))
        metrics["mean_pred_anomaly"] = float(np.mean(pred[~nom]))

    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate saved yield ensemble.")
    p.add_argument("--artifacts", type=Path, default=Path("ml/artifacts"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--yield-threshold", type=float, default=0.85)
    args = p.parse_args()
    evaluate(args.artifacts, args.data, args.yield_threshold)


if __name__ == "__main__":
    main()
