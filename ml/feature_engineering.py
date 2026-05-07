"""Feature construction for yield models (fab-aligned semantics)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from wafer_sim.models import ProcessParams

NUMERIC_FEATURE_NAMES: list[str] = [
    "temperature",
    "pressure",
    "gas_flow",
    "rf_power",
    "deposition_time",
    "tool_run_count",
    "lot_position",
    "mean_thickness",
    "std_thickness",
    "thickness_uniformity_pct",
    "mean_defect_density",
    "max_defect_density",
    "inv_temperature",
    "log_defect_density",
    "DA_product",
    "rf_power_deviation",
    "rf_power_deviation_sq",
    "rf_energy_density",
    "temp_window_proximity",
    "pressure_window_proximity",
    "temp_pressure_coupling",
    "flow_pressure_ratio",
    "rf_per_flow",
    "rf_per_flow_in_safe_range",
    "thickness_cv",
    "defect_spatial_range",
    "runs_since_pm",
    "is_post_pm",
    "spc_alert_temperature",
    "spc_alert_pressure",
    "spc_alert_gas_flow",
    "spc_alert_rf_power",
    "spc_alert_deposition_time",
    "spc_alert_count",
    "spc_severity_score",
]

CATEGORICAL_FEATURE_NAMES: list[str] = ["tool_id", "recipe_id", "shift"]

ALL_FEATURE_NAMES: list[str] = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES

GP_FEATURE_NAMES: list[str] = [
    "inv_temperature",
    "log_defect_density",
    "DA_product",
    "rf_power_deviation_sq",
    "flow_pressure_ratio",
    "temp_pressure_coupling",
    "thickness_cv",
    "runs_since_pm",
]

META_PASSTHROUGH: list[str] = [
    "inv_temperature",
    "log_defect_density",
    "DA_product",
    "rf_power_deviation_sq",
    "runs_since_pm",
]

MONO_INCREASE: list[str] = [
    "temp_window_proximity",
    "pressure_window_proximity",
    "flow_pressure_ratio",
    "rf_per_flow_in_safe_range",
]
MONO_DECREASE: list[str] = [
    "log_defect_density",
    "DA_product",
    "rf_power_deviation_sq",
    "rf_energy_density",
    "runs_since_pm",
    "thickness_cv",
]


def _tool_str_from_int(tool_id: int) -> str:
    return ("tool_A", "tool_B", "tool_C")[int(tool_id)]


@dataclass
class TrainingStats:
    param_means: dict[str, float] = field(default_factory=dict)
    param_stds: dict[str, float] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)
    # Median log defect (pre-imputation engineered column) for "high defect" monotonicity checks at inference.
    median_log_defect_density: float | None = None


class FeatureEngineeringPipeline:
    def __init__(self) -> None:
        self._numeric_pipe: Pipeline | None = None
        self._cat_encoder: OrdinalEncoder | None = None
        self._gp_scaler: StandardScaler | None = None
        self.training_stats: TrainingStats = TrainingStats()
        self.category_vocab_: dict[str, list[Any]] = {}

    def fit(self, df: pd.DataFrame) -> FeatureEngineeringPipeline:
        wide = build_engineered_frame(df)
        X_num = wide[NUMERIC_FEATURE_NAMES].astype(np.float64)
        self._numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
        self._numeric_pipe.fit(X_num)

        cats = wide[CATEGORICAL_FEATURE_NAMES].astype(str)
        self.category_vocab_ = {c: sorted(cats[c].unique().tolist()) for c in CATEGORICAL_FEATURE_NAMES}
        self._cat_encoder = OrdinalEncoder(
            categories=[self.category_vocab_[c] for c in CATEGORICAL_FEATURE_NAMES],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        self._cat_encoder.fit(cats)

        Xtr = self._to_matrix(wide)
        gp_idx = [ALL_FEATURE_NAMES.index(n) for n in GP_FEATURE_NAMES]
        self._gp_scaler = StandardScaler().fit(Xtr[:, gp_idx])

        raw_cols = ["temperature", "pressure", "gas_flow", "rf_power", "deposition_time"]
        self.training_stats = TrainingStats(
            columns=raw_cols,
            param_means={c: float(wide[c].mean()) for c in raw_cols},
            param_stds={c: float(wide[c].std(ddof=0)) or 1e-9 for c in raw_cols},
            median_log_defect_density=float(wide["log_defect_density"].median()),
        )
        return self

    def _to_matrix(self, wide: pd.DataFrame) -> np.ndarray:
        assert self._numeric_pipe is not None and self._cat_encoder is not None
        X_num = wide[NUMERIC_FEATURE_NAMES].astype(np.float64)
        X_num_t = self._numeric_pipe.transform(X_num)
        cats = wide[CATEGORICAL_FEATURE_NAMES].astype(str)
        X_cat = self._cat_encoder.transform(cats)
        return np.hstack([X_num_t, X_cat.astype(np.float64)])

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self._numeric_pipe is None or self._cat_encoder is None:
            raise RuntimeError("Call fit before transform.")
        wide = build_engineered_frame(df)
        return self._to_matrix(wide)

    def gp_from_full_X(self, X: np.ndarray) -> np.ndarray:
        assert self._gp_scaler is not None
        gp_idx = [ALL_FEATURE_NAMES.index(n) for n in GP_FEATURE_NAMES]
        gp = X[:, gp_idx].astype(np.float64)
        return self._gp_scaler.transform(gp)

    def transform_gp(self, df: pd.DataFrame) -> np.ndarray:
        return self.gp_from_full_X(self.transform(df))

    def get_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        wide = build_engineered_frame(df)
        assert self._numeric_pipe is not None
        wide[NUMERIC_FEATURE_NAMES] = self._numeric_pipe.transform(wide[NUMERIC_FEATURE_NAMES].astype(np.float64))
        return wide


def build_engineered_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    t = out["temperature"].astype(np.float64)
    p = out["pressure"].astype(np.float64)
    g = out["gas_flow"].astype(np.float64)
    rf = out["rf_power"].astype(np.float64)
    dep = out["deposition_time"].astype(np.float64)

    out["inv_temperature"] = 1.0 / (t + 273.15)
    md = out["mean_defect_density"].astype(np.float64)
    out["log_defect_density"] = np.log(md + 1e-6)
    out["DA_product"] = md * 1.0

    out["rf_power_deviation"] = rf - 200.0
    out["rf_power_deviation_sq"] = out["rf_power_deviation"] ** 2
    out["rf_energy_density"] = rf * dep

    out["temp_window_proximity"] = np.clip(1.0 - np.abs(t - 400.0) / 50.0, 0.0, 1.0)
    out["pressure_window_proximity"] = np.clip(1.0 - np.abs(p - 50.0) / 30.0, 0.0, 1.0)

    out["temp_pressure_coupling"] = out["inv_temperature"] * np.log(np.clip(p, 1e-6, None))
    out["flow_pressure_ratio"] = g / np.maximum(p, 1e-6)
    out["rf_per_flow"] = rf / np.maximum(g, 1e-6)
    out["rf_per_flow_in_safe_range"] = g / (rf + 1.0)

    mt = out["mean_thickness"].astype(np.float64)
    st = out["std_thickness"].astype(np.float64)
    out["thickness_cv"] = st / np.maximum(mt, 1e-6)
    out["defect_spatial_range"] = out["max_defect_density"].astype(np.float64) - md

    trc = out["tool_run_count"].astype(np.int64)
    out["runs_since_pm"] = trc % 150
    out["is_post_pm"] = (out["runs_since_pm"] < 3).astype(np.float64)

    for c in [
        "spc_alert_temperature",
        "spc_alert_pressure",
        "spc_alert_gas_flow",
        "spc_alert_rf_power",
        "spc_alert_deposition_time",
        "spc_alert_count",
        "spc_severity_score",
    ]:
        if c not in out.columns:
            out[c] = 0.0

    for c in CATEGORICAL_FEATURE_NAMES:
        out[c] = out[c].astype(str)

    return out


def params_and_summary_to_dataframe(
    params: ProcessParams,
    wafer_summary: dict[str, Any] | None,
    *,
    tool_run_count: int | None = None,
    lot_position: int | None = None,
    recipe_id: str = "standard",
    shift: str = "day",
) -> pd.DataFrame:
    tool = _tool_str_from_int(params.tool_id)
    trc = int(tool_run_count) if tool_run_count is not None else 75
    lp = int(lot_position) if lot_position is not None else 13

    row: dict[str, Any] = {
        "temperature": params.temperature_c,
        "pressure": params.pressure_mtorr,
        "gas_flow": params.gas_flow_sccm,
        "rf_power": params.rf_power_w,
        "deposition_time": params.deposition_time_s,
        "tool_id": tool,
        "recipe_id": recipe_id,
        "shift": shift,
        "tool_run_count": trc,
        "lot_position": lp,
    }
    if wafer_summary:
        row.update(wafer_summary)
    else:
        row["mean_thickness"] = np.nan
        row["std_thickness"] = np.nan
        row["thickness_uniformity_pct"] = np.nan
        row["mean_defect_density"] = np.nan
        row["max_defect_density"] = np.nan
    return pd.DataFrame([row])


def xgboost_monotone_tuple() -> tuple[int, ...]:
    d: dict[str, int] = {}
    for name in MONO_INCREASE:
        d[name] = 1
    for name in MONO_DECREASE:
        d[name] = -1
    return tuple(d.get(n, 0) for n in ALL_FEATURE_NAMES)


def lightgbm_categorical_indices() -> list[int]:
    return [ALL_FEATURE_NAMES.index(c) for c in CATEGORICAL_FEATURE_NAMES]


def save_feature_names(output_dir: Path) -> None:
    payload = {"all_features": ALL_FEATURE_NAMES, "meta_passthrough": META_PASSTHROUGH}
    (output_dir / "feature_names.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
