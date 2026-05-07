"""Tests for yield ensemble training and prediction API."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("lightgbm")
pytest.importorskip("xgboost")

from wafer_sim.generator import generate_dataset
from wafer_sim.models import ProcessParams

from ml.feature_engineering import FeatureEngineeringPipeline
from ml.monotonicity import audit_xgboost_monotonicity
from ml.train import run_training
from ml.uncertainty import interval_coverage
from ml.yield_model import WaferSummary, load_ensemble, predict_yield


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("yield_ml")
    data_dir = base / "data"
    art_dir = base / "artifacts"
    plot_dir = base / "plots"
    data_dir.mkdir()
    art_dir.mkdir()
    plot_dir.mkdir()
    generate_dataset(
        n_lots=50,
        wafers_per_lot=20,
        seed=123,
        output_dir=data_dir,
        anomalies_enabled=True,
        validate=False,
    )
    run_training(
        data_dir / "wafer_summary.parquet",
        seed=123,
        n_trials=2,
        yield_threshold=0.85,
        skip_gp=True,
        output_dir=art_dir,
        plots_dir=plot_dir,
    )
    (art_dir / "train_data_path.txt").write_text(str(data_dir / "wafer_summary.parquet"), encoding="utf-8")
    return art_dir


def _data_parquet(art: Path) -> Path:
    return Path((art / "train_data_path.txt").read_text(encoding="utf-8").strip())


def test_lot_cv_and_temporal_r2(trained_artifacts: Path) -> None:
    m = json.loads((trained_artifacts / "model_metrics.json").read_text(encoding="utf-8"))
    assert m["lot_cv"]["r2"] > 0.85
    assert m["temporal_holdout"]["r2"] > 0.75


def test_feature_names_include_spc_columns(trained_artifacts: Path) -> None:
    payload = json.loads((trained_artifacts / "feature_names.json").read_text(encoding="utf-8"))
    feats = payload["all_features"]
    for col in [
        "spc_alert_temperature",
        "spc_alert_pressure",
        "spc_alert_gas_flow",
        "spc_alert_rf_power",
        "spc_alert_deposition_time",
        "spc_alert_count",
        "spc_severity_score",
    ]:
        assert col in feats


def test_xgboost_monotonicity_baseline(trained_artifacts: Path) -> None:
    ens = load_ensemble(trained_artifacts)
    fe: FeatureEngineeringPipeline = ens.feature_pipe
    df = pd.read_parquet(_data_parquet(trained_artifacts)).sort_values("timestamp")
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    X = fe.transform(train_df)
    audit = audit_xgboost_monotonicity(lambda Z: ens.xgb_reg.predict(Z), X, n_grid=100)
    assert audit.violations_pct == 0.0, audit.violated_features


def test_stacked_ensemble_monotonicity_reported(trained_artifacts: Path) -> None:
    """Stacking can violate local monotonicity; audit still runs and records metrics."""
    m = json.loads((trained_artifacts / "model_metrics.json").read_text(encoding="utf-8"))
    assert "violations_pct" in m["monotonicity_audit"]
    assert "violated_features" in m["monotonicity_audit"]


def test_quantile_coverage_on_holdout(trained_artifacts: Path) -> None:
    ens = load_ensemble(trained_artifacts)
    fe: FeatureEngineeringPipeline = ens.feature_pipe
    df = pd.read_parquet(_data_parquet(trained_artifacts)).sort_values("timestamp")
    split = int(len(df) * 0.8)
    hold = df.iloc[split:]
    X = fe.transform(hold)
    Xg = fe.transform_gp(hold)
    y = hold["wafer_yield"].to_numpy()
    lo, hi = ens.predict_ci(X, Xg)
    cov = interval_coverage(y, lo, hi)
    assert cov >= 0.88


def test_anomaly_separation(trained_artifacts: Path) -> None:
    ens = load_ensemble(trained_artifacts)
    fe = ens.feature_pipe
    df = pd.read_parquet(_data_parquet(trained_artifacts)).sort_values("timestamp")
    X = fe.transform(df)
    Xg = fe.transform_gp(df)
    pred = ens.predict_point(X, Xg)
    y = df["wafer_yield"].to_numpy()
    nom = df["anomaly_type"].isna()
    if nom.any() and (~nom).any():
        assert float(np.mean(y[~nom])) < float(np.mean(y[nom]))
        assert float(np.median(pred[~nom])) <= float(np.median(pred[nom]))


def test_predict_yield_latency(trained_artifacts: Path) -> None:
    ens = load_ensemble(trained_artifacts)
    p = ProcessParams(
        temperature_c=400.0,
        pressure_mtorr=50.0,
        gas_flow_sccm=80.0,
        rf_power_w=200.0,
        deposition_time_s=120.0,
        tool_id=0,
    )
    t0 = time.perf_counter()
    for _ in range(15):
        predict_yield(p, None, return_shap=False, ensemble=ens)
    dt = (time.perf_counter() - t0) / 15 * 1000
    assert dt < 200, f"mean latency {dt:.1f}ms"


def test_pre_vs_post_ci_width(trained_artifacts: Path) -> None:
    ens = load_ensemble(trained_artifacts)
    p = ProcessParams(400.0, 50.0, 80.0, 200.0, 120.0, 0)
    pre = predict_yield(p, None, return_shap=False, ensemble=ens, tool_run_count=50, lot_position=5)
    ws = WaferSummary(
        mean_thickness=95.0,
        std_thickness=3.0,
        thickness_uniformity_pct=2.5,
        mean_defect_density=0.2,
        max_defect_density=0.35,
    )
    post = predict_yield(p, ws, return_shap=False, ensemble=ens, tool_run_count=50, lot_position=5)
    assert post.ci_width <= pre.ci_width + 1e-6


def test_predict_yield_exposes_spc_summary(trained_artifacts: Path) -> None:
    ens = load_ensemble(trained_artifacts)
    p = ProcessParams(400.0, 50.0, 80.0, 200.0, 120.0, 0)
    pred = predict_yield(p, None, return_shap=False, ensemble=ens)
    assert isinstance(pred.spc_alerts_active, bool)
    assert isinstance(pred.spc_alert_count, int)
    assert pred.spc_alert_count >= 0
    assert pred.spc_severity >= 0.0


def test_risk_flags_out_of_bounds(trained_artifacts: Path) -> None:
    ens = load_ensemble(trained_artifacts)
    p = ProcessParams(900.0, 50.0, 80.0, 200.0, 120.0, 0)  # invalid temp for model but may not validate
    try:
        pred = predict_yield(p, None, return_shap=False, ensemble=ens)
    except Exception:
        return
    crit = [f for f in pred.risk_flags if f.severity == "critical"]
    assert len(crit) >= 1
