"""Stacked ensemble for wafer yield: LightGBM quantiles, monotone XGBoost, GP, Ridge meta."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import pydantic
from sklearn.linear_model import Ridge

from wafer_sim.models import ProcessParams

from ml.feature_engineering import (
    ALL_FEATURE_NAMES,
    META_PASSTHROUGH,
    FeatureEngineeringPipeline,
    lightgbm_categorical_indices,
    params_and_summary_to_dataframe,
)
from ml.monotonicity import MonotonicityViolationWarning
from ml.uncertainty import combine_confidence_intervals, pass_probability_gaussian
from wafer_sim.fdc import FDCFeatureAugmentor


def _lgb_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64).copy()
    for j in lightgbm_categorical_indices():
        X[:, j] = np.round(X[:, j])
    return X


__all__ = [
    "MonotonicityViolationWarning",
    "RiskFlag",
    "WaferSummary",
    "YieldPrediction",
    "YieldEnsemble",
    "predict_yield",
]


class WaferSummary(pydantic.BaseModel):
    """Post-deposition measurements (optional fields filled when known)."""

    mean_thickness: float | None = None
    std_thickness: float | None = None
    thickness_uniformity_pct: float | None = None
    mean_defect_density: float | None = None
    max_defect_density: float | None = None


class RiskFlag(pydantic.BaseModel):
    parameter: str
    message: str
    severity: Literal["info", "warning", "critical"]
    shap_contribution: float


class YieldPrediction(pydantic.BaseModel):
    predicted_yield: float
    yield_pct: float
    confidence_interval: tuple[float, float]
    ci_width: float
    pass_probability: float
    shap_breakdown: dict[str, float] | None
    top_risk_factor: str
    risk_flags: list[RiskFlag]
    model_agreement: float
    spc_alerts_active: bool = False
    spc_severity: float = 0.0
    spc_alert_count: int = 0


@dataclass
class YieldEnsemble:
    """Full production artifact: preprocessors + base learners + meta + SHAP config."""

    feature_pipe: FeatureEngineeringPipeline
    lgbm_q025: Any
    lgbm_q50: Any
    lgbm_q975: Any
    xgb_reg: Any
    gp_reg: Any | None
    meta: Ridge
    yield_threshold: float = 0.85
    meta_feature_names_: list[str] = field(default_factory=list)
    shap_feature_names_: list[str] = field(default_factory=list)
    _shap_explainer: Any = None

    def _meta_block(self, X: np.ndarray, X_gp: np.ndarray) -> np.ndarray:
        Xl = _lgb_matrix(X)
        lt = np.asarray(self.lgbm_q025.predict(Xl), dtype=np.float64).ravel()
        lm = np.asarray(self.lgbm_q50.predict(Xl), dtype=np.float64).ravel()
        lh = np.asarray(self.lgbm_q975.predict(Xl), dtype=np.float64).ravel()
        xr = np.asarray(self.xgb_reg.predict(X), dtype=np.float64).ravel()
        if self.gp_reg is not None:
            gm, gs = self.gp_reg.predict(X_gp, return_std=True)
            gm = np.asarray(gm, dtype=np.float64).ravel()
            gs = np.asarray(gs, dtype=np.float64).ravel()
        else:
            gm = lm.copy()
            gs = np.maximum((lh - lt) / 3.92, 1e-4)

        idx = [ALL_FEATURE_NAMES.index(n) for n in META_PASSTHROUGH]
        if X.shape[0] == 0:
            return np.zeros((0, 6 + len(idx)))
        P = X[:, idx].astype(np.float64)
        return np.column_stack([lt, lm, lh, xr, gm, gs, P])

    def predict_point(self, X: np.ndarray, X_gp: np.ndarray) -> np.ndarray:
        M = self._meta_block(X, X_gp)
        return np.asarray(self.meta.predict(M), dtype=np.float64).ravel()

    def predict_ci(
        self,
        X: np.ndarray,
        X_gp: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        M = self._meta_block(X, X_gp)
        point = np.asarray(self.meta.predict(M), dtype=np.float64).ravel()
        coef = np.asarray(self.meta.coef_.ravel(), dtype=np.float64)
        # Pad coef if needed
        n_m = M.shape[1]
        if len(coef) < n_m:
            coef = np.pad(coef, (0, n_m - len(coef)))
        los = []
        his = []
        for i in range(M.shape[0]):
            w = coef[:6]
            lt, lm, lh, xr, gm, gs = M[i, :6]
            lo, hi = combine_confidence_intervals(
                float(lt), float(lh), float(gm), float(gs), w
            )
            los.append(lo)
            his.append(hi)
        return np.array(los), np.array(his)

    def compute_shap_dict(self, X: np.ndarray) -> dict[str, float]:
        if self._shap_explainer is None:
            return {}
        try:
            sv = self._shap_explainer.shap_values(X[:1])
            vals = np.asarray(sv, dtype=np.float64)
            if vals.ndim == 2:
                vals = vals[0]
            names = self.shap_feature_names_
            return {names[j]: float(vals[j]) for j in range(min(len(names), len(vals)))}
        except Exception:
            return {}

    def attach_shap(self) -> None:
        try:
            import shap

            self._shap_explainer = shap.TreeExplainer(self.lgbm_q50)
        except Exception:
            self._shap_explainer = None


_GLOBAL_ENSEMBLE: YieldEnsemble | None = None
_GLOBAL_FDC_AUGMENTOR = FDCFeatureAugmentor()


def load_ensemble(artifacts_dir: Path | str) -> YieldEnsemble:
    path = Path(artifacts_dir) / "yield_ensemble.joblib"
    obj = joblib.load(path)
    if not isinstance(obj, YieldEnsemble):
        raise TypeError(f"Expected YieldEnsemble at {path}")
    return obj


def _enforce_monotonicity(
    prediction: YieldPrediction,
    params: ProcessParams,
    wide_row: dict[str, Any],
    reference_predictions: dict[str, float],
    *,
    median_log_defect_density: float | None,
) -> YieldPrediction:
    """
    If ensemble point prediction violates a known monotone regime vs. the monotone XGBoost
    baseline, fall back to the XGBoost-only prediction (0% monotonicity violations on that model).
    """
    xgb_pred = float(reference_predictions["xgb"])
    ensemble_pred = float(prediction.predicted_yield)
    violations: list[str] = []

    log_dd = float(wide_row.get("log_defect_density", 0.0))
    log_defect_density_high = (
        median_log_defect_density is not None and log_dd > median_log_defect_density
    )

    if log_defect_density_high and ensemble_pred > xgb_pred + 0.02:
        violations.append("log_defect_density")

    if abs(params.rf_power_w - 200.0) > 30.0 and ensemble_pred > xgb_pred + 0.02:
        violations.append("rf_power_deviation_sq")

    if not violations:
        return prediction

    xgb_clipped = float(np.clip(xgb_pred, 0.0, 1.0))
    new_flags = [
        *prediction.risk_flags,
        RiskFlag(
            parameter="ensemble_monotonicity",
            message=(
                "Ensemble overridden by XGBoost due to constraint violation: "
                f"{violations}"
            ),
            severity="info",
            shap_contribution=0.0,
        ),
    ]
    return prediction.model_copy(
        update={
            "predicted_yield": xgb_clipped,
            "yield_pct": xgb_clipped * 100.0,
            "risk_flags": new_flags,
        }
    )


def predict_yield(
    params: ProcessParams,
    wafer_summary: WaferSummary | None = None,
    *,
    return_shap: bool = True,
    ensemble: YieldEnsemble | None = None,
    tool_run_count: int | None = None,
    lot_position: int | None = None,
    recipe_id: str = "standard",
    shift: str = "day",
) -> YieldPrediction:
    """Predict wafer yield (pre- or post-deposition)."""
    ens = ensemble or _GLOBAL_ENSEMBLE
    if ens is None:
        art = Path(__file__).resolve().parent / "artifacts"
        if (art / "yield_ensemble.joblib").exists():
            ens = load_ensemble(art)
        else:
            raise RuntimeError("No trained YieldEnsemble loaded; train models or pass ensemble=.")

    ws = wafer_summary.model_dump(exclude_none=True) if wafer_summary else None
    pre_only = ws is None or len(ws) == 0
    df = params_and_summary_to_dataframe(
        params,
        ws,
        tool_run_count=tool_run_count,
        lot_position=lot_position,
        recipe_id=recipe_id,
        shift=shift,
    )
    spc_inputs = {
        "temperature": float(params.temperature_c),
        "pressure": float(params.pressure_mtorr),
        "gas_flow": float(params.gas_flow_sccm),
        "rf_power": float(params.rf_power_w),
        "deposition_time": float(params.deposition_time_s),
    }
    spc_augmented = _GLOBAL_FDC_AUGMENTOR.augment(spc_inputs)
    for k, v in spc_augmented.items():
        if k.startswith("spc_"):
            df[k] = float(v)
    X = ens.feature_pipe.transform(df)
    wide = ens.feature_pipe.get_dataframe(df)
    X_gp = ens.feature_pipe.transform_gp(df)

    t0 = perf_counter()
    pred = float(ens.predict_point(X, X_gp)[0])
    lo, hi = ens.predict_ci(X, X_gp)
    lo_f, hi_f = float(lo[0]), float(hi[0])
    if pre_only:
        mid = 0.5 * (lo_f + hi_f)
        half = 0.5 * (hi_f - lo_f) * 1.55
        lo_f, hi_f = max(0.0, mid - half), min(1.0, mid + half)

    lt = float(np.asarray(ens.lgbm_q50.predict(_lgb_matrix(X)), dtype=np.float64).ravel()[0])
    xr = float(np.asarray(ens.xgb_reg.predict(X), dtype=np.float64).ravel()[0])
    if ens.gp_reg is not None:
        gm, gs = ens.gp_reg.predict(X_gp, return_std=True)
        gm_f = float(np.asarray(gm, dtype=np.float64).ravel()[0])
        gs_f = float(np.asarray(gs, dtype=np.float64).ravel()[0])
    else:
        gm_f, gs_f = lt, max(
            (float(ens.lgbm_q975.predict(_lgb_matrix(X))[0]) - float(ens.lgbm_q025.predict(_lgb_matrix(X))[0]))
            / 3.92,
            1e-4,
        )

    agreement = float(np.std([lt, xr, gm_f]))

    shap_d = ens.compute_shap_dict(X) if return_shap else {}
    top_risk = "none"
    if shap_d:
        neg = sorted(shap_d.items(), key=lambda kv: kv[1])
        top_risk = neg[0][0] if neg[0][1] < 0 else "none"

    flags = _risk_flags(
        params,
        wide.iloc[0].to_dict(),
        ens,
        agreement,
        shap_d,
        pre_only,
    )

    pp = pass_probability_gaussian(gm_f, gs_f, ens.yield_threshold)

    elapsed_ms = (perf_counter() - t0) * 1000
    if elapsed_ms > 200:
        warnings.warn(f"predict_yield latency {elapsed_ms:.1f}ms exceeds 200ms target", stacklevel=2)

    out = YieldPrediction(
        predicted_yield=float(np.clip(pred, 0.0, 1.0)),
        yield_pct=float(np.clip(pred, 0.0, 1.0)) * 100.0,
        confidence_interval=(float(np.clip(lo_f, 0, 1)), float(np.clip(hi_f, 0, 1))),
        ci_width=float(max(hi_f - lo_f, 0.0)),
        pass_probability=pp,
        shap_breakdown=shap_d if return_shap else None,
        top_risk_factor=top_risk,
        risk_flags=flags,
        model_agreement=agreement,
        spc_alerts_active=bool(_GLOBAL_FDC_AUGMENTOR.last_summary["spc_alerts_active"]),
        spc_severity=float(_GLOBAL_FDC_AUGMENTOR.last_summary["spc_severity"]),
        spc_alert_count=int(_GLOBAL_FDC_AUGMENTOR.last_summary["spc_alert_count"]),
    )
    med_log_dd = getattr(
        ens.feature_pipe.training_stats,
        "median_log_defect_density",
        None,
    )
    return _enforce_monotonicity(
        out,
        params,
        wide.iloc[0].to_dict(),
        {"xgb": xr},
        median_log_defect_density=med_log_dd,
    )


def _risk_flags(
    params: ProcessParams,
    row: dict[str, Any],
    ens: YieldEnsemble,
    agreement: float,
    shap_d: dict[str, float],
    pre_only: bool,
) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    stats = ens.feature_pipe.training_stats
    for param, val in [
        ("temperature", params.temperature_c),
        ("pressure", params.pressure_mtorr),
        ("gas_flow", params.gas_flow_sccm),
        ("rf_power", params.rf_power_w),
        ("deposition_time", params.deposition_time_s),
    ]:
        if param in stats.param_means:
            mu = stats.param_means[param]
            sd = max(stats.param_stds[param], 1e-9)
            if abs(val - mu) > 2 * sd:
                sh = float(shap_d.get(param, shap_d.get("rf_power_deviation_sq", 0.0)))
                flags.append(
                    RiskFlag(
                        parameter=param,
                        message=f"{param} far outside training band (>{2}σ)",
                        severity="critical",
                        shap_contribution=sh,
                    )
                )

    tw = float(row.get("temp_window_proximity", 1.0))
    pw = float(row.get("pressure_window_proximity", 1.0))
    if tw < 0.2 or pw < 0.2:
        flags.append(
            RiskFlag(
                parameter="process_window",
                message="Near process window edge (low proximity)",
                severity="warning",
                shap_contribution=float(shap_d.get("temp_window_proximity", 0.0)),
            )
        )

    rsp = float(row.get("runs_since_pm", 0))
    if rsp > 120:
        flags.append(
            RiskFlag(
                parameter="tool_age",
                message="runs_since_pm approaching PM interval",
                severity="warning",
                shap_contribution=float(shap_d.get("runs_since_pm", 0.0)),
            )
        )

    if agreement > 0.05:
        flags.append(
            RiskFlag(
                parameter="ensemble",
                message="Base learners disagree — elevated epistemic uncertainty",
                severity="warning",
                shap_contribution=0.0,
            )
        )

    if int(row.get("is_post_pm", 0)) == 1 and not pre_only:
        flags.append(
            RiskFlag(
                parameter="qualification",
                message="Post-PM qualification window — expect elevated variability",
                severity="info",
                shap_contribution=0.0,
            )
        )

    return flags
