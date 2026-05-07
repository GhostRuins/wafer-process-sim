"""CLI: train stacked yield ensemble with lot-stratified CV, Optuna, and artifacts."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from collections.abc import Callable
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.pruners import MedianPruner
import pandas as pd
import shap
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
import lightgbm as lgb
import xgboost as xgb

from ml.feature_engineering import (
    ALL_FEATURE_NAMES,
    GP_FEATURE_NAMES,
    META_PASSTHROUGH,
    FeatureEngineeringPipeline,
    lightgbm_categorical_indices,
    save_feature_names,
    xgboost_monotone_tuple,
)
from ml.monotonicity import audit_monotonicity_arrays, audit_xgboost_monotonicity, warn_if_audit_failed
from ml.uncertainty import (
    expected_calibration_error,
    interval_coverage,
    reliability_diagram_data,
)
from ml.yield_model import YieldEnsemble, _lgb_matrix
from wafer_sim.fdc import FDCFeatureAugmentor


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, p)))


def _metrics_block(y_true: np.ndarray, y_pred: np.ndarray, thr: float) -> dict[str, float]:
    out: dict[str, float] = {
        "rmse": _rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    if len(np.unique((y_true > thr).astype(int))) > 1:
        out["auc_roc"] = float(roc_auc_score((y_true > thr).astype(int), y_pred))
    else:
        out["auc_roc"] = float("nan")
    return out


def _cast_cat_int(X: np.ndarray) -> np.ndarray:
    return _lgb_matrix(X)


def _stratified_subsample_ids(
    df: pd.DataFrame,
    y: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    tool = df["tool_id"].astype(str)
    try:
        q = pd.qcut(y, q=4, labels=False, duplicates="drop")
    except ValueError:
        q = np.zeros(len(y), dtype=int)
    parts: list[np.ndarray] = []
    for t in sorted(tool.unique()):
        for k in np.unique(q):
            m = (tool.values == t) & (q == k)
            idx = np.where(m)[0]
            if len(idx) == 0:
                continue
            take = max(1, int(n / (len(tool.unique()) * max(1, len(np.unique(q))))))
            take = min(len(idx), take)
            parts.append(rng.choice(idx, size=take, replace=False))
    idx = np.unique(np.concatenate(parts)) if parts else np.arange(min(max(1, n), len(df)))
    if len(idx) > n:
        idx = rng.choice(idx, size=n, replace=False)
    return idx


def _optuna_lgb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    n_trials: int,
    seed: int,
    alpha: float,
    out_study: Path,
) -> dict[str, Any]:
    cat = lightgbm_categorical_indices()

    def objective(trial: optuna.Trial) -> float:
        n_est = trial.suggest_int("n_estimators", 500, 2000)
        nl = trial.suggest_int("num_leaves", 31, 127)
        lr = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        mcs = trial.suggest_int("min_child_samples", 10, 50)
        params: dict[str, Any] = {
            "objective": "quantile",
            "alpha": alpha,
            "learning_rate": lr,
            "num_leaves": nl,
            "min_child_samples": mcs,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbosity": -1,
            "n_estimators": n_est,
        }
        dtr = lgb.Dataset(
            _cast_cat_int(X_tr),
            label=y_tr,
            feature_name=ALL_FEATURE_NAMES,
            categorical_feature=cat,
        )
        dva = lgb.Dataset(
            _cast_cat_int(X_va),
            label=y_va,
            feature_name=ALL_FEATURE_NAMES,
            categorical_feature=cat,
        )
        booster = lgb.train(
            params,
            dtr,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred = booster.predict(_cast_cat_int(X_va))
        return _rmse(y_va, pred)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    study.optimize(objective, n_trials=max(1, n_trials), show_progress_bar=False)
    out_study.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(study, out_study)
    return study.best_params


def _optuna_xgb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    n_trials: int,
    seed: int,
    out_study: Path,
) -> dict[str, Any]:
    mono = xgboost_monotone_tuple()

    def objective(trial: optuna.Trial) -> float:
        ne = trial.suggest_int("n_estimators", 500, 1500)
        md = trial.suggest_int("max_depth", 4, 8)
        lr = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        ss = trial.suggest_float("subsample", 0.6, 1.0)
        reg = xgb.XGBRegressor(
            n_estimators=ne,
            max_depth=md,
            learning_rate=lr,
            subsample=ss,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=2.0,
            monotone_constraints=mono,
            random_state=seed,
            tree_method="hist",
            n_jobs=0,
        )
        reg.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return _rmse(y_va, reg.predict(X_va))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed + 1),
        pruner=MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=max(1, n_trials), show_progress_bar=False)
    out_study.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(study, out_study)
    return study.best_params


def _optuna_kernel(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    n_trials: int,
    seed: int,
    out_study: Path,
) -> Any:
    d = X_tr.shape[1]

    def objective(trial: optuna.Trial) -> float:
        ls = [trial.suggest_float(f"ls_{j}", 0.1, 10.0, log=True) for j in range(d)]
        noise = trial.suggest_float("noise", 1e-4, 0.1, log=True)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=ls, length_scale_bounds=(1e-2, 100.0)) + WhiteKernel(
            noise_level=noise,
            noise_level_bounds=(1e-6, 1.0),
        )
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, random_state=seed)
        gp.fit(X_tr, y_tr)
        return _rmse(y_va, gp.predict(X_va))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed + 2),
        pruner=MedianPruner(n_startup_trials=4),
    )
    study.optimize(objective, n_trials=max(3, n_trials // 2), show_progress_bar=False)
    out_study.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(study, out_study)
    best = study.best_params
    ls = [best[f"ls_{j}"] for j in range(d)]
    noise = best["noise"]
    return C(1.0) * RBF(length_scale=ls) + WhiteKernel(noise_level=noise)


def _train_lgb_quantile(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    alpha: float,
    best: dict[str, Any],
) -> lgb.Booster:
    cat = lightgbm_categorical_indices()
    params: dict[str, Any] = {
        "objective": "quantile",
        "alpha": alpha,
        "learning_rate": float(best.get("learning_rate", 0.05)),
        "num_leaves": int(best.get("num_leaves", 63)),
        "min_child_samples": int(best.get("min_child_samples", 20)),
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbosity": -1,
        "n_estimators": int(best.get("n_estimators", 1000)),
    }
    dtr = lgb.Dataset(
        _cast_cat_int(X_tr),
        label=y_tr,
        feature_name=ALL_FEATURE_NAMES,
        categorical_feature=cat,
    )
    dva = lgb.Dataset(
        _cast_cat_int(X_va),
        label=y_va,
        feature_name=ALL_FEATURE_NAMES,
        categorical_feature=cat,
    )
    return lgb.train(params, dtr, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])


def _xgb_from_params(best: dict[str, Any], seed: int) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=int(best.get("n_estimators", 800)),
        max_depth=int(best.get("max_depth", 6)),
        learning_rate=float(best.get("learning_rate", 0.05)),
        subsample=float(best.get("subsample", 0.8)),
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        monotone_constraints=xgboost_monotone_tuple(),
        random_state=seed,
        tree_method="hist",
        n_jobs=0,
    )


def _make_plots(
    ensemble: YieldEnsemble,
    train_df: pd.DataFrame,
    hold_df: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_hold: np.ndarray,
    y_hold: np.ndarray,
    pred_hold: np.ndarray,
    plots_dir: Path,
    thr: float,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    tools = train_df["tool_id"].astype(str)

    fig, ax = plt.subplots(figsize=(6, 5))
    for t in tools.unique():
        m = tools == t
        ax.scatter(y_train[m], ensemble.predict_point(X_train[m], ensemble.feature_pipe.gp_from_full_X(X_train[m])), label=t, alpha=0.35, s=10)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Actual yield")
    ax.set_ylabel("Predicted yield")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    feat_plot = [
        "temperature",
        "pressure",
        "gas_flow",
        "rf_power",
        "mean_defect_density",
        "tool_run_count",
        "lot_position",
        "thickness_uniformity_pct",
        "runs_since_pm",
    ]
    wide = ensemble.feature_pipe.get_dataframe(train_df)
    resid = y_train - ensemble.predict_point(X_train, ensemble.feature_pipe.gp_from_full_X(X_train))
    fig, axes = plt.subplots(3, 3, figsize=(9, 8), layout="constrained")
    axes = axes.ravel()
    for ax, name in zip(axes, feat_plot, strict=False):
        x = wide[name].to_numpy()
        ax.scatter(x, resid, s=6, alpha=0.3)
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_xlabel(name)
        ax.set_ylabel("residual")
    fig.savefig(plots_dir / "residuals_by_feature.png", dpi=120)
    plt.close(fig)

    shap_dir = Path(plots_dir).parent / "artifacts" / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    bg = _lgb_matrix(X_train[: min(200, len(X_train))])
    explainer = shap.TreeExplainer(ensemble.lgbm_q50)
    sv = explainer.shap_values(_lgb_matrix(X_hold[: min(300, len(X_hold))]))
    np.save(shap_dir / "shap_values_test.npy", np.asarray(sv))
    mean_abs = np.mean(np.abs(np.asarray(sv)), axis=0)
    top_idx = np.argsort(-mean_abs)[:15]
    ranked = {ALL_FEATURE_NAMES[i]: float(mean_abs[i]) for i in sorted(top_idx, key=lambda k: -mean_abs[k])}
    (shap_dir / "shap_summary.json").write_text(json.dumps({"mean_abs_shap": ranked}, indent=2), encoding="utf-8")

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(np.asarray(sv)[:, top_idx], features=X_hold[: sv.shape[0], :][:, top_idx], feature_names=[ALL_FEATURE_NAMES[i] for i in top_idx], show=False)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(plots_dir / "shap_beeswarm.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    ts = pd.to_datetime(hold_df["timestamp"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ts, y_hold, ".", label="actual", alpha=0.5)
    ax.plot(ts, pred_hold, ".", label="pred", alpha=0.5)
    ax.legend()
    ax.set_title("Temporal holdout")
    fig.tight_layout()
    fig.savefig(plots_dir / "temporal_holdout_drift.png", dpi=120)
    plt.close(fig)

    rs = (hold_df["tool_run_count"].astype(int) % 150).to_numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(rs, y_hold, s=8, alpha=0.4, label="actual")
    ax.scatter(rs, pred_hold, s=8, alpha=0.4, label="pred")
    ax.set_xlabel("runs_since_pm")
    ax.set_ylabel("yield")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "yield_vs_tool_age.png", dpi=120)
    plt.close(fig)

    nom = hold_df["anomaly_type"].isna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pred_hold[nom], bins=25, alpha=0.6, density=True, label="nominal")
    ax.hist(pred_hold[~nom], bins=25, alpha=0.6, density=True, label="anomaly")
    ax.set_xlabel("predicted yield")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "anomaly_separation.png", dpi=120)
    plt.close(fig)

    lo, hi = ensemble.predict_ci(X_hold, ensemble.feature_pipe.gp_from_full_X(X_hold))
    cov = np.mean((y_hold >= lo) & (y_hold <= hi))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["coverage"], [cov])
    ax.axhline(0.95, color="r", ls="--")
    ax.set_ylim(0, 1)
    ax.set_title(f"Quantile/GP CI coverage = {cov:.3f}")
    fig.tight_layout()
    fig.savefig(plots_dir / "quantile_coverage.png", dpi=120)
    plt.close(fig)

    try:
        inter = explainer.shap_interaction_values(bg[: min(120, len(bg))])
        np.save(shap_dir / "shap_interaction_values.npy", np.asarray(inter))
    except Exception:
        np.save(shap_dir / "shap_interaction_values.npy", np.zeros((0, 0, 0)))

    try:
        j = ALL_FEATURE_NAMES.index("inv_temperature")
        k = ALL_FEATURE_NAMES.index("rf_power_deviation_sq")
        shap.dependence_plot(
            j,
            np.asarray(sv),
            X_hold[: sv.shape[0], :],
            interaction_index=k,
            feature_names=ALL_FEATURE_NAMES,
            show=False,
        )
        fig = plt.gcf()
        fig.savefig(plots_dir / "shap_interaction_temp_rf.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass

    # Waterfall examples
    try:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
        idxs = [
            int(np.argmax(y_hold)),
            int(np.argmin(y_hold)),
            int(np.where(~hold_df["anomaly_type"].isna().to_numpy())[0][0]) if (~hold_df["anomaly_type"].isna()).any() else 0,
            int(np.where((hold_df["tool_run_count"].astype(int) % 150) < 3)[0][0]),
        ]
        for ax, ix in zip(axes.ravel(), idxs, strict=False):
            shap.waterfall_plot(
                shap.Explanation(
                    values=np.asarray(sv)[ix],
                    base_values=explainer.expected_value,
                    data=X_hold[ix],
                    feature_names=ALL_FEATURE_NAMES,
                ),
                show=False,
            )
            plt.sca(ax)
        fig.savefig(plots_dir / "shap_waterfall_examples.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass


def run_training(
    data_path: Path,
    seed: int,
    n_trials: int,
    yield_threshold: float,
    skip_gp: bool,
    output_dir: Path,
    plots_dir: Path,
) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)
    sort_cols = ["timestamp"] + (["run_id"] if "run_id" in df.columns else [])
    df = df.sort_values(sort_cols).reset_index(drop=True)
    augmentor = FDCFeatureAugmentor()
    spc_source_cols = ["temperature", "pressure", "gas_flow", "rf_power", "deposition_time"]
    for idx in df.index:
        row_params = {k: float(df.at[idx, k]) for k in spc_source_cols}
        augmented = augmentor.augment(row_params)
        for k, v in augmented.items():
            if k.startswith("spc_"):
                df.at[idx, k] = float(v)
    n = len(df)
    split = max(int(n * 0.8), 2)
    train_df = df.iloc[:split].copy()
    hold_df = df.iloc[split:].copy()

    fe = FeatureEngineeringPipeline().fit(train_df)
    X = fe.transform(train_df)
    y = train_df["wafer_yield"].to_numpy(dtype=np.float64)
    groups = train_df["lot_id"].astype(str).values

    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    first_tr, first_va = next(iter(gkf.split(X, y, groups)))
    X_otr, X_ova = X[first_tr], X[first_va]
    y_otr, y_ova = y[first_tr], y[first_va]
    tr_df = train_df.iloc[first_tr]
    va_df = train_df.iloc[first_va]
    Xg_otr = fe.transform_gp(tr_df)
    Xg_ova = fe.transform_gp(va_df)

    studies = output_dir / "optuna_studies"
    studies.mkdir(parents=True, exist_ok=True)
    b025 = _optuna_lgb(X_otr, y_otr, X_ova, y_ova, n_trials, seed, 0.025, studies / "lgb_q025.joblib")
    b50 = _optuna_lgb(X_otr, y_otr, X_ova, y_ova, max(3, n_trials // 2), seed + 1, 0.5, studies / "lgb_q50.joblib")
    b975 = _optuna_lgb(X_otr, y_otr, X_ova, y_ova, n_trials, seed + 2, 0.975, studies / "lgb_q975.joblib")
    bx = _optuna_xgb(X_otr, y_otr, X_ova, y_ova, n_trials, seed, studies / "xgb.joblib")

    kernel_small = None
    if not skip_gp:
        sub_idx = _stratified_subsample_ids(train_df.iloc[first_tr].reset_index(drop=True), y_otr, min(800, len(first_tr)), rng)
        tr_sub = first_tr[sub_idx]
        mask_va = np.ones(len(first_va), dtype=bool)
        kernel_small = _optuna_kernel(
            Xg_otr[sub_idx],
            y_otr[sub_idx],
            Xg_ova,
            y_ova,
            n_trials,
            seed,
            studies / "gp.joblib",
        )

    n_folds = min(5, len(np.unique(groups)))
    oof = np.zeros((len(train_df), 6))
    inner = GroupKFold(n_splits=max(2, n_folds))

    for tr_idx, va_idx in inner.split(X, y, groups):
        Xtr, Xva, ytr, yva = X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]
        dtr = train_df.iloc[tr_idx]
        dva = train_df.iloc[va_idx]
        Xgtr, Xgva = fe.transform_gp(dtr), fe.transform_gp(dva)

        n_iv = max(1, int(len(tr_idx) * 0.92))
        cal_tr, es_tr = tr_idx[:n_iv], tr_idx[n_iv:]
        bos025 = _train_lgb_quantile(X[cal_tr], y[cal_tr], X[es_tr], y[es_tr], 0.025, b025)
        bos50 = _train_lgb_quantile(X[cal_tr], y[cal_tr], X[es_tr], y[es_tr], 0.5, b50)
        bos975 = _train_lgb_quantile(X[cal_tr], y[cal_tr], X[es_tr], y[es_tr], 0.975, b975)
        xg_m = _xgb_from_params(bx, seed)
        xg_m.fit(Xtr, ytr)

        oof[va_idx, 0] = bos025.predict(_cast_cat_int(Xva))
        oof[va_idx, 1] = bos50.predict(_cast_cat_int(Xva))
        oof[va_idx, 2] = bos975.predict(_cast_cat_int(Xva))
        oof[va_idx, 3] = xg_m.predict(Xva)
        if skip_gp or kernel_small is None:
            oof[va_idx, 4] = oof[va_idx, 1]
            oof[va_idx, 5] = np.maximum((oof[va_idx, 2] - oof[va_idx, 0]) / 3.92, 1e-4)
        else:
            sub = _stratified_subsample_ids(dtr.reset_index(drop=True), ytr, min(1500, len(dtr)), rng)
            gp = GaussianProcessRegressor(kernel=kernel_small, normalize_y=True, random_state=seed)
            gp.fit(Xgtr[sub], ytr[sub])
            gm, gs = gp.predict(Xgva, return_std=True)
            oof[va_idx, 4] = gm
            oof[va_idx, 5] = gs

    idx_meta = [ALL_FEATURE_NAMES.index(n) for n in META_PASSTHROUGH]
    M_train = np.column_stack([oof, X[:, idx_meta]])
    meta = RidgeCV(alphas=np.logspace(-4, 3, 30)).fit(M_train, y)

    # Final models on full train
    va_es = np.arange(min(max(2, len(X) // 10), len(X) - 1))
    tr_es = np.setdiff1d(np.arange(len(X)), va_es)
    lgbm_q025 = _train_lgb_quantile(X[tr_es], y[tr_es], X[va_es], y[va_es], 0.025, b025)
    lgbm_q50 = _train_lgb_quantile(X[tr_es], y[tr_es], X[va_es], y[va_es], 0.5, b50)
    lgbm_q975 = _train_lgb_quantile(X[tr_es], y[tr_es], X[va_es], y[va_es], 0.975, b975)
    xgb_reg = _xgb_from_params(bx, seed)
    xgb_reg.fit(X, y)

    gp_reg = None
    gp_ard: dict[str, float] = {n: float("nan") for n in GP_FEATURE_NAMES}
    if not skip_gp and kernel_small is not None:
        sub_all = _stratified_subsample_ids(train_df, y, min(2000, len(train_df)), rng)
        gp_reg = GaussianProcessRegressor(kernel=kernel_small, normalize_y=True, random_state=seed)
        gp_reg.fit(fe.transform_gp(train_df.iloc[sub_all]), y[sub_all])
        try:
            k2 = gp_reg.kernel_
            rbf = k2.k1.k2  # type: ignore[attr-defined]
            ls = np.atleast_1d(np.asarray(rbf.length_scale, dtype=np.float64))
            for i, name in enumerate(GP_FEATURE_NAMES):
                gp_ard[name] = float(ls[i] if len(ls) > i else ls[0])
        except Exception:
            pass

    ensemble = YieldEnsemble(
        feature_pipe=fe,
        lgbm_q025=lgbm_q025,
        lgbm_q50=lgbm_q50,
        lgbm_q975=lgbm_q975,
        xgb_reg=xgb_reg,
        gp_reg=gp_reg,
        meta=meta,
        yield_threshold=yield_threshold,
        meta_feature_names_=[f"lgb_q{a}" for a in ["025", "050", "975"]]
        + ["xgb", "gp_mean", "gp_std"]
        + META_PASSTHROUGH,
        shap_feature_names_=ALL_FEATURE_NAMES,
    )
    ensemble.attach_shap()

    y_hat_oof = meta.predict(M_train)
    lot_cv = _metrics_block(y, y_hat_oof, yield_threshold)

    rkfold = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=seed)
    rand_oof = np.zeros_like(y)
    for tr, va in rkfold.split(X):
        m = RidgeCV(alphas=np.logspace(-4, 3, 20)).fit(M_train[tr], y[tr])
        rand_oof[va] = m.predict(M_train[va])
    random_cv = _metrics_block(y, rand_oof, yield_threshold)

    X_h = fe.transform(hold_df)
    X_gh = fe.transform_gp(hold_df)
    M_h = ensemble._meta_block(X_h, X_gh)
    pred_h = meta.predict(M_h)
    temporal = _metrics_block(hold_df["wafer_yield"].to_numpy(), pred_h, yield_threshold)

    nom = train_df["anomaly_type"].isna().to_numpy()
    ano = ~nom
    rs = (train_df["tool_run_count"].astype(int) % 150).to_numpy()
    pmq = rs < 3

    by_subset: dict[str, Any] = {
        "nominal_runs": _metrics_block(y[nom], y_hat_oof[nom], yield_threshold) if nom.any() else {},
        "anomaly_runs": _metrics_block(y[ano], y_hat_oof[ano], yield_threshold) if ano.any() else {},
        "post_pm_runs": _metrics_block(y[pmq], y_hat_oof[pmq], yield_threshold) if pmq.any() else {},
        "by_tool": {},
    }
    for t in sorted(train_df["tool_id"].unique()):
        m = (train_df["tool_id"] == t).to_numpy()
        by_subset["by_tool"][str(t)] = _metrics_block(y[m], y_hat_oof[m], yield_threshold)

    coef = np.asarray(meta.coef_, dtype=np.float64).ravel()
    meta_w = {
        "lgbm_coef": float(np.mean(np.abs(coef[:3]))),
        "xgb_coef": float(abs(coef[3])),
        "gp_coef": float(abs(coef[4]) + abs(coef[5])),
    }

    lo_ci, hi_ci = ensemble.predict_ci(X, fe.gp_from_full_X(X))
    cov_train = interval_coverage(y, lo_ci, hi_ci)
    gp_mean_oof = oof[:, 4]
    gp_std_oof = oof[:, 5] + 1e-6
    cal_ece = expected_calibration_error(y, gp_mean_oof, gp_std_oof)
    rel = reliability_diagram_data(y, gp_mean_oof, gp_std_oof)

    stack_predict: Callable[[np.ndarray], np.ndarray] = lambda Xs: ensemble.meta.predict(
        ensemble._meta_block(Xs, fe.gp_from_full_X(Xs))
    )
    audit_result = audit_monotonicity_arrays(stack_predict, X, n_grid=80)
    warn_if_audit_failed(audit_result)
    xgb_audit = audit_xgboost_monotonicity(lambda Xs: xgb_reg.predict(Xs), X, n_grid=80)

    metrics = {
        "lot_cv": lot_cv,
        "random_cv": random_cv,
        "temporal_holdout": temporal,
        "by_subset": by_subset,
        "meta_learner_weights": meta_w,
        "meta_learner_coef": [float(c) for c in coef.tolist()],
        "gp_ard_length_scales": gp_ard,
        "monotonicity_audit": {
            "violations_pct": audit_result.violations_pct,
            "violated_features": audit_result.violated_features,
            "xgboost_baseline": {
                "violations_pct": xgb_audit.violations_pct,
                "violated_features": xgb_audit.violated_features,
            },
        },
        "calibration": {
            "expected_calibration_error": cal_ece,
            "reliability_diagram_data": rel,
            "interval_coverage_train": cov_train,
        },
    }
    (output_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    save_feature_names(output_dir)
    joblib.dump(ensemble, output_dir / "yield_ensemble.joblib")
    joblib.dump(lgbm_q025, output_dir / "lgbm_quantile_low.joblib")
    joblib.dump(lgbm_q50, output_dir / "lgbm_quantile_mid.joblib")
    joblib.dump(lgbm_q975, output_dir / "lgbm_quantile_high.joblib")
    joblib.dump(xgb_reg, output_dir / "xgb_monotone.joblib")
    joblib.dump(gp_reg, output_dir / "gp_regressor.joblib")
    joblib.dump(meta, output_dir / "meta_learner.joblib")
    joblib.dump(fe, output_dir / "feature_engineering.joblib")

    _make_plots(ensemble, train_df, hold_df, X, y, X_h, hold_df["wafer_yield"].to_numpy(), pred_h, plots_dir, yield_threshold)

def main() -> None:
    p = argparse.ArgumentParser(description="Train wafer yield ensemble.")
    p.add_argument("--data", type=Path, default=Path("data/wafer_summary.parquet"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-trials", type=int, default=60)
    p.add_argument("--yield-threshold", type=float, default=0.85)
    p.add_argument("--skip-gp", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("ml/artifacts"))
    p.add_argument("--plots-dir", type=Path, default=Path("ml/plots"))
    p.add_argument("--evaluate-only", type=Path, default=None, help="New wafer_summary.parquet; evaluate saved artifacts")
    args = p.parse_args()
    if args.evaluate_only is not None:
        from ml.evaluate import evaluate

        evaluate(args.output_dir, args.evaluate_only, args.yield_threshold)
        return
    if not args.data.exists():
        raise SystemExit(f"Data not found: {args.data}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        run_training(
            args.data,
            args.seed,
            args.n_trials,
            args.yield_threshold,
            args.skip_gp,
            args.output_dir,
            args.plots_dir,
        )


if __name__ == "__main__":
    main()
