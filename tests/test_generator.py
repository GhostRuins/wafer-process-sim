"""Tests for synthetic dataset generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wafer_sim.generator import generate_dataset

DIE_COLS = [
    "run_id",
    "wafer_id",
    "lot_id",
    "tool_id",
    "recipe_id",
    "temperature",
    "pressure",
    "gas_flow",
    "rf_power",
    "deposition_time",
    "x_pos",
    "y_pos",
    "r_pos",
    "film_thickness",
    "thickness_delta_from_target",
    "defect_density",
    "charging_damage_contribution",
    "particle_contamination_contribution",
    "die_yield",
    "pass_fail",
    "uniformity_factor",
    "window_factor",
    "timestamp",
    "shift",
]

SUM_COLS = [
    "run_id",
    "wafer_id",
    "lot_id",
    "tool_id",
    "recipe_id",
    "temperature",
    "pressure",
    "gas_flow",
    "rf_power",
    "deposition_time",
    "mean_thickness",
    "std_thickness",
    "range_thickness",
    "thickness_uniformity_pct",
    "mean_defect_density",
    "max_defect_density",
    "wafer_yield",
    "die_count",
    "passing_die_count",
    "yield_loss_uniformity",
    "yield_loss_window",
    "yield_loss_defects",
    "anomaly_type",
    "anomaly_severity",
    "tool_run_count",
    "timestamp",
    "shift",
    "lot_position",
]


def _out(tmp_path: Path) -> Path:
    d = tmp_path / "out"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_parquet_schema_and_counts(tmp_path: Path) -> None:
    tmp_out = _out(tmp_path)
    n_lots, wpl, gn = 3, 10, 12
    generate_dataset(
        n_lots=n_lots,
        wafers_per_lot=wpl,
        seed=123,
        output_dir=tmp_out,
        validate=False,
        grid_n=gn,
    )
    dies = pd.read_parquet(tmp_out / "wafer_dies.parquet")
    summ = pd.read_parquet(tmp_out / "wafer_summary.parquet")
    lots = pd.read_parquet(tmp_out / "lot_summary.parquet")
    assert set(DIE_COLS) <= set(dies.columns)
    assert set(SUM_COLS) <= set(summ.columns)
    n_wafers = n_lots * wpl
    assert len(summ) == n_wafers
    assert len(lots) == n_lots
    # approximate die count from circular grid
    assert len(dies) == summ["die_count"].sum()
    assert len(dies) == n_wafers * summ["die_count"].iloc[0]


def test_no_nulls_required(tmp_path: Path) -> None:
    tmp_out = _out(tmp_path)
    generate_dataset(
        n_lots=2,
        wafers_per_lot=8,
        seed=7,
        output_dir=tmp_out,
        validate=False,
        grid_n=14,
    )
    dies = pd.read_parquet(tmp_out / "wafer_dies.parquet")
    summ = pd.read_parquet(tmp_out / "wafer_summary.parquet")
    for c in DIE_COLS:
        assert dies[c].notna().all(), c
    for c in SUM_COLS:
        if c in ("anomaly_type", "anomaly_severity"):
            continue
        assert summ[c].notna().all(), c


def test_yield_range_nominal(tmp_path: Path) -> None:
    tmp_out = _out(tmp_path)
    generate_dataset(
        n_lots=8,
        wafers_per_lot=20,
        seed=99,
        output_dir=tmp_out,
        anomalies_enabled=False,
        validate=False,
        grid_n=16,
    )
    summ = pd.read_parquet(tmp_out / "wafer_summary.parquet")
    y = summ["wafer_yield"]
    assert 0.55 < float(y.mean()) < 0.99
    assert float(y.std()) > 0.005


def test_anomaly_fraction(tmp_path: Path) -> None:
    tmp_out = _out(tmp_path)
    generate_dataset(
        n_lots=15,
        wafers_per_lot=25,
        seed=42,
        output_dir=tmp_out,
        anomalies_enabled=True,
        validate=False,
        grid_n=10,
    )
    summ = pd.read_parquet(tmp_out / "wafer_summary.parquet")
    frac = float(summ["anomaly_type"].notna().mean())
    assert 0.03 < frac < 0.22


def test_deterministic_seed(tmp_path: Path) -> None:
    p = tmp_path / "a"
    p.mkdir()
    generate_dataset(n_lots=2, wafers_per_lot=5, seed=555, output_dir=p, grid_n=12)
    d1 = pd.read_parquet(p / "wafer_dies.parquet")
    p2 = tmp_path / "b"
    p2.mkdir()
    generate_dataset(n_lots=2, wafers_per_lot=5, seed=555, output_dir=p2, grid_n=12)
    d2 = pd.read_parquet(p2 / "wafer_dies.parquet")
    pd.testing.assert_frame_equal(d1, d2)


def test_correlation_direction(tmp_path: Path) -> None:
    tmp_out = _out(tmp_path)
    generate_dataset(
        n_lots=12,
        wafers_per_lot=25,
        seed=1000,
        output_dir=tmp_out,
        anomalies_enabled=False,
        validate=False,
        grid_n=10,
    )
    summ = pd.read_parquet(tmp_out / "wafer_summary.parquet")
    r = summ[["temperature", "mean_thickness"]].corr().iloc[0, 1]
    assert r > 0.15
    # Arrhenius thickness vs T should be visible in quintiles (robust to masking by uniformity)
    summ = summ.sort_values("temperature")
    k = max(2, len(summ) // 5)
    low = summ.iloc[:k]["mean_thickness"].mean()
    high = summ.iloc[-k:]["mean_thickness"].mean()
    assert float(high) > float(low)


def test_tool_aging_defect_correlation(tmp_path: Path) -> None:
    tmp_out = _out(tmp_path)
    generate_dataset(
        n_lots=10,
        wafers_per_lot=25,
        seed=2000,
        output_dir=tmp_out,
        anomalies_enabled=False,
        validate=False,
        grid_n=10,
    )
    summ = pd.read_parquet(tmp_out / "wafer_summary.parquet")
    r = summ[["tool_run_count", "mean_defect_density"]].corr().iloc[0, 1]
    assert r > 0.05


def test_lot_position_one_higher_defect(tmp_path: Path) -> None:
    tmp_out = _out(tmp_path)
    generate_dataset(
        n_lots=10,
        wafers_per_lot=25,
        seed=3000,
        output_dir=tmp_out,
        anomalies_enabled=False,
        validate=False,
        grid_n=10,
    )
    dies = pd.read_parquet(tmp_out / "wafer_dies.parquet")
    summ = pd.read_parquet(tmp_out / "wafer_summary.parquet")
    m1 = dies.merge(summ[["run_id", "lot_position"]], on="run_id", how="left")
    a = float(m1.loc[m1["lot_position"] == 1, "defect_density"].mean())
    b = float(m1.loc[m1["lot_position"] > 1, "defect_density"].mean())
    assert a > b


def test_metadata_json(tmp_path: Path) -> None:
    tmp_out = _out(tmp_path)
    meta = generate_dataset(
        n_lots=2,
        wafers_per_lot=5,
        seed=1,
        output_dir=tmp_out,
        validate=False,
        grid_n=10,
    )
    with (tmp_out / "metadata.json").open(encoding="utf-8") as f:
        disk = json.load(f)
    assert disk["schema_version"]
    assert disk["seed"] == 1
    assert "physics_model_snapshot" in disk
    assert meta["row_counts"]["wafer_summary"] == 10
