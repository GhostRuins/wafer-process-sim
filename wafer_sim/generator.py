"""Synthetic wafer fab dataset generation (parquet + metadata)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

from wafer_sim.anomaly_injector import AnomalyLogEntry, WaferAnomalyContext, build_anomaly_plan
from wafer_sim.lot_scheduler import WaferScheduleSlot, build_chronological_timestamps, lot_timestamp_bounds
from wafer_sim.models import ProcessParams, YieldModelConfig
from wafer_sim.physics import (
    add_tool_age_offset,
    compute_wafer_die_grid,
    get_physics_snapshot,
    get_tool_run_count,
    increment_tool_run_count,
    init_physics,
    reset_tool_age,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore

SCHEMA_VERSION = "1.0.0"


def _to_datetime64_ns(dt: datetime) -> np.datetime64:
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return np.datetime64(dt.isoformat(), "ns")

# Reporting / calibration (physics stays in physical units; training stats match fab bands)
DEFAULT_YIELD_STRESS_FACTOR = 35.0
DEFECT_DISPLAY_SCALE = 28.0
DEFECT_DISPLAY_CLIP = 0.55
DEFAULT_GRID_N = 36


class DataGenerationError(RuntimeError):
    """Raised when statistical validation fails after generation."""


def _tool_name(tid: int) -> str:
    return ("tool_A", "tool_B", "tool_C")[tid]


def _recipe_name(r: str) -> str:
    return r


def _sample_lot_bases(
    n_lots: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lot-level T, RF, P, G, time with correlations."""
    cov_tr = np.array([[8.0**2, 0.3 * 8.0 * 15.0], [0.3 * 8.0 * 15.0, 15.0**2]], dtype=np.float64)
    tr = rng.multivariate_normal(np.array([400.0, 200.0]), cov_tr, size=n_lots)

    cov_pg = np.array([[1.0, 0.4], [0.4, 1.0]], dtype=np.float64)
    zg = rng.multivariate_normal(np.zeros(2), cov_pg, size=n_lots)
    u1 = norm.cdf(zg[:, 0])
    u2 = norm.cdf(zg[:, 1])
    gas = 40.0 + u1 * 80.0
    mu_log = np.log(50.0)
    pressure = np.exp(mu_log + 0.18 * zg[:, 1])
    dep = rng.normal(120.0, 8.0, size=n_lots)

    t = np.clip(tr[:, 0], 350.0, 450.0)
    rf = np.clip(tr[:, 1], 150.0, 300.0)
    pressure = np.clip(pressure, 20.0, 100.0)
    gas = np.clip(gas, 40.0, 120.0)
    dep = np.clip(dep, 90.0, 180.0)
    return t, rf, pressure, gas, dep, zg[:, 1]


def _assign_tools_and_recipes(
    n_lots: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    tools = rng.choice([0, 1, 2], size=n_lots, p=[0.5, 0.3, 0.2])
    recipes = rng.choice(["standard", "high_rate", "low_stress"], size=n_lots, p=[0.6, 0.25, 0.15])
    return tools, recipes


def _recipe_deltas(recipe: str) -> tuple[float, float, float, float]:
    if recipe == "high_rate":
        return 15.0, 0.0, 20.0, 10.0
    if recipe == "low_stress":
        return -10.0, -5.0, -30.0, 0.0
    return 0.0, 0.0, 0.0, 0.0


def _build_process_params(
    lot_t: float,
    lot_rf: float,
    lot_p: float,
    lot_g: float,
    lot_dep: float,
    recipe: str,
    wafer_noise: np.ndarray,
    tool_id: int,
    ctx: WaferAnomalyContext,
    post_pm_noise: float,
) -> ProcessParams:
    dt, dp, drf, dg = _recipe_deltas(recipe)
    temperature_c = float(
        np.clip(lot_t + dt + wafer_noise[0] * post_pm_noise, 350.0, 450.0)
    )
    pressure_mtorr = float(np.clip(lot_p + dp + wafer_noise[1] * post_pm_noise, 20.0, 100.0))
    gas_flow_sccm = float(np.clip(lot_g + dg + wafer_noise[2] * post_pm_noise, 40.0, 120.0))
    rf_power_w = float(np.clip(lot_rf + drf + wafer_noise[3] * post_pm_noise, 150.0, 300.0))
    deposition_time_s = float(np.clip(lot_dep + wafer_noise[4] * post_pm_noise, 90.0, 180.0))
    if ctx.pressure_override is not None:
        pressure_mtorr = float(np.clip(ctx.pressure_override, 20.0, 100.0))
    if ctx.gas_override is not None:
        gas_flow_sccm = float(np.clip(ctx.gas_override, 40.0, 120.0))
    temperature_c = float(np.clip(temperature_c + ctx.temp_delta, 350.0, 450.0))
    rf_power_w = float(np.clip(rf_power_w + ctx.rf_delta, 150.0, 300.0))
    return ProcessParams(
        temperature_c=temperature_c,
        pressure_mtorr=pressure_mtorr,
        gas_flow_sccm=gas_flow_sccm,
        rf_power_w=rf_power_w,
        deposition_time_s=deposition_time_s,
        tool_id=tool_id,
    )


def generate_dataset(
    n_lots: int = 20,
    wafers_per_lot: int = 25,
    seed: int = 42,
    output_dir: Path | str = Path("data"),
    yield_model: Literal["murphy", "seeds", "negative_binomial"] = "murphy",
    anomalies_enabled: bool = True,
    pm_interval: int = 150,
    verbose: bool = False,
    validate: bool = False,
    yield_stress_factor: float = DEFAULT_YIELD_STRESS_FACTOR,
    grid_n: int = DEFAULT_GRID_N,
) -> dict[str, Any]:
    """Generate parquet outputs and metadata; return metadata dict."""
    rng = np.random.default_rng(seed)
    init_physics(seed)
    for tid in (0, 1, 2):
        reset_tool_age(tid)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_wafers = n_lots * wafers_per_lot
    lot_t, lot_rf, lot_p, lot_g, lot_dep, _ = _sample_lot_bases(n_lots, rng)
    lot_tools, lot_recipes = _assign_tools_and_recipes(n_lots, rng)

    # Chronological wafer order: interleave lots (lot-major order is typical)
    run_idx = 0
    lot_ids: list[str] = []
    wafer_ids: list[str] = []
    run_ids: list[str] = []
    tool_ids: list[int] = []
    recipe_ids: list[str] = []
    lot_positions: list[int] = []
    timestamps: list[datetime] = []
    shifts: list[str] = []

    slots = build_chronological_timestamps(n_wafers, rng)

    for lot in range(n_lots):
        tool = int(lot_tools[lot])
        recipe = str(lot_recipes[lot])
        for pos in range(1, wafers_per_lot + 1):
            run_ids.append(f"RUN_{run_idx:06d}")
            wafer_ids.append(f"WAF_L{lot:03d}_{run_idx:06d}")
            lot_ids.append(f"LOT_{lot:03d}")
            tool_ids.append(tool)
            recipe_ids.append(recipe)
            lot_positions.append(pos)
            timestamps.append(slots[run_idx].timestamp)
            shifts.append(slots[run_idx].shift)
            run_idx += 1

    tool_ids_arr = np.array(tool_ids, dtype=np.int32)
    ctxs, anomaly_log = build_anomaly_plan(tool_ids_arr, rng, enabled=anomalies_enabled)

    # PM / post-PM state
    wafers_since_pm = {0: 0, 1: 0, 2: 0}
    post_pm_left = {0: 0, 1: 0, 2: 0}
    aging_followon = {0: 0, 1: 0, 2: 0}
    aging_factor = {0: 1.0, 1: 1.0, 2: 1.0}

    die_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    lot_indices = np.repeat(np.arange(n_lots, dtype=np.int32), wafers_per_lot)

    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=n_wafers,
            desc="wafers",
            unit="wafer",
            disable=not sys.stdout.isatty(),
        )

    t0 = time.perf_counter()
    anomaly_count = 0

    for wi in range(n_wafers):
        lot = int(lot_indices[wi])
        tool = int(tool_ids_arr[wi])
        recipe = recipe_ids[wi]
        ctx = ctxs[wi]

        # PM scheduling
        unsched = rng.random() < 0.02
        if wafers_since_pm[tool] >= pm_interval or unsched:
            reset_tool_age(tool)
            wafers_since_pm[tool] = 0
            post_pm_left[tool] = 3

        post_pm_noise = 1.5 if post_pm_left[tool] > 0 else 1.0
        if post_pm_left[tool] > 0:
            post_pm_left[tool] -= 1

        wafer_noise = np.array(
            [
                rng.normal(0.0, 2.0),
                rng.normal(0.0, 1.5),
                rng.normal(0.0, 1.0),
                rng.normal(0.0, 5.0),
                rng.normal(0.0, 2.0),
            ]
        )

        if ctx.virtual_age_add:
            add_tool_age_offset(tool, ctx.virtual_age_add)
            aging_followon[tool] = 20

        if aging_followon[tool] > 0:
            aging_followon[tool] -= 1
            aging_factor[tool] = 1.0 + 0.006 * (20 - aging_followon[tool])
        else:
            aging_factor[tool] = 1.0

        params = _build_process_params(
            float(lot_t[lot]),
            float(lot_rf[lot]),
            float(lot_p[lot]),
            float(lot_g[lot]),
            float(lot_dep[lot]),
            recipe,
            wafer_noise,
            tool,
            ctx,
            post_pm_noise,
        )

        tool_age = get_tool_run_count(tool)

        defect_scale = 1.0 * aging_factor[tool]
        particle_scale = float(ctx.particle_scale)
        pboost: tuple[float, int] | None = None
        if ctx.anomaly_type == "particle_shower" and ctx.particle_quadrant is not None:
            pboost = (particle_scale, int(ctx.particle_quadrant))
            particle_scale = 1.0

        lot_pos = lot_positions[wi]
        lot_boost = 1.12 if lot_pos == 1 else 1.0

        seed_w = int(rng.integers(0, 2**31 - 1))
        grid = compute_wafer_die_grid(
            params,
            seed=seed_w,
            grid_nx=grid_n,
            grid_ny=grid_n,
            yield_model=yield_model,
            a_die_cm2=YieldModelConfig().a_die_cm2,
            yield_stress_factor=yield_stress_factor,
            defect_density_scale=defect_scale * lot_boost,
            particle_scale=particle_scale,
            particle_quadrant_boost=pboost,
        )

        increment_tool_run_count(tool)
        wafers_since_pm[tool] += 1

        if ctx.anomaly_type is not None:
            anomaly_count += 1

        raw_scale = DEFECT_DISPLAY_SCALE
        raw_tot = grid.defect_density_cm2 * raw_scale
        dd = np.clip(raw_tot, 0.0, DEFECT_DISPLAY_CLIP)
        tot_phys = grid.charging_damage_cm2 + grid.particle_contamination_cm2
        safe = np.maximum(tot_phys, 1e-15)
        frac_c = grid.charging_damage_cm2 / safe
        frac_p = grid.particle_contamination_cm2 / safe
        ch = frac_c * dd
        pc = frac_p * dd

        n_die = int(grid.x_pos.shape[0])
        pass_fail = grid.die_yield > 0.85
        passing = int(np.sum(pass_fail))

        rid = run_ids[wi]
        wid = wafer_ids[wi]
        lid = lot_ids[wi]

        summary_rows.append(
            {
                "run_id": rid,
                "wafer_id": wid,
                "lot_id": lid,
                "tool_id": _tool_name(tool),
                "recipe_id": _recipe_name(recipe),
                "temperature": params.temperature_c,
                "pressure": params.pressure_mtorr,
                "gas_flow": params.gas_flow_sccm,
                "rf_power": params.rf_power_w,
                "deposition_time": params.deposition_time_s,
                "mean_thickness": grid.mean_film_thickness_nm,
                "std_thickness": grid.std_film_thickness_nm,
                "range_thickness": grid.range_thickness_nm,
                "thickness_uniformity_pct": min(4.0, max(1.5, grid.thickness_uniformity_pct * 5.0)),
                "mean_defect_density": float(np.mean(dd)),
                "max_defect_density": float(np.max(dd)),
                "wafer_yield": grid.wafer_yield,
                "die_count": n_die,
                "passing_die_count": passing,
                "yield_loss_uniformity": grid.yield_loss_uniformity,
                "yield_loss_window": grid.yield_loss_window,
                "yield_loss_defects": grid.yield_loss_defects,
                "anomaly_type": ctx.anomaly_type,
                "anomaly_severity": ctx.anomaly_severity,
                "tool_run_count": tool_age,
                "timestamp": timestamps[wi],
                "shift": shifts[wi],
                "lot_position": lot_pos,
            }
        )

        die_block = {
            "run_id": np.full(n_die, rid),
            "wafer_id": np.full(n_die, wid),
            "lot_id": np.full(n_die, lid),
            "tool_id": np.full(n_die, _tool_name(tool)),
            "recipe_id": np.full(n_die, _recipe_name(recipe)),
            "temperature": np.full(n_die, params.temperature_c),
            "pressure": np.full(n_die, params.pressure_mtorr),
            "gas_flow": np.full(n_die, params.gas_flow_sccm),
            "rf_power": np.full(n_die, params.rf_power_w),
            "deposition_time": np.full(n_die, params.deposition_time_s),
            "x_pos": grid.x_pos,
            "y_pos": grid.y_pos,
            "r_pos": grid.r_pos,
            "film_thickness": grid.film_thickness_nm,
            "thickness_delta_from_target": grid.thickness_delta_nm,
            "defect_density": dd,
            "charging_damage_contribution": ch,
            "particle_contamination_contribution": pc,
            "die_yield": grid.die_yield,
            "pass_fail": pass_fail,
            "uniformity_factor": grid.uniformity_factor_die,
            "window_factor": grid.window_factor_die,
            "timestamp": np.full(n_die, _to_datetime64_ns(timestamps[wi])),
            "shift": np.full(n_die, shifts[wi]),
        }
        die_rows.append(die_block)

        if pbar is not None:
            pbar.set_postfix(
                lot=f"{lot+1}/{n_lots}",
                tool=_tool_name(tool),
                wy=f"{grid.wafer_yield:.3f}",
                anom=anomaly_count,
                refresh=False,
            )
            pbar.update(1)
        elif verbose and wi % wafers_per_lot == 0:
            print(f"lot {lot} tool {_tool_name(tool)} wafer_yield={grid.wafer_yield:.3f}")

    if pbar is not None:
        pbar.close()

    elapsed = time.perf_counter() - t0
    if elapsed > 30 and n_wafers <= 500 and DEFAULT_GRID_N == 36:
        pass  # perf warning optional

    if not die_rows:
        raise DataGenerationError("No die rows generated.")
    die_df = pd.DataFrame(
        {k: np.concatenate([d[k] for d in die_rows]) for k in die_rows[0].keys()}
    )
    sum_df = pd.DataFrame(summary_rows)

    # Fix log entries run_id
    for e in anomaly_log:
        if 0 <= e.window_start < n_wafers:
            e.run_id = run_ids[e.window_start]

    # Lot summary
    lot_summaries: list[dict[str, Any]] = []
    starts, ends = lot_timestamp_bounds(
        [WaferScheduleSlot(ts, sh) for ts, sh in zip(timestamps, shifts, strict=True)],
        lot_indices,
        n_lots,
    )
    for lot in range(n_lots):
        mask = sum_df["lot_id"] == f"LOT_{lot:03d}"
        sub = sum_df.loc[mask]
        anoms = int(sub["anomaly_type"].notna().sum())
        modes = sub["anomaly_type"].dropna()
        dom = str(modes.mode().iloc[0]) if len(modes) else "none"
        lot_summaries.append(
            {
                "lot_id": f"LOT_{lot:03d}",
                "tool_id": sub["tool_id"].iloc[0],
                "recipe_id": sub["recipe_id"].iloc[0],
                "lot_yield": float(np.prod(sub["wafer_yield"].values)),
                "mean_wafer_yield": float(sub["wafer_yield"].mean()),
                "std_wafer_yield": float(sub["wafer_yield"].std(ddof=1) if len(sub) > 1 else 0.0),
                "start_timestamp": starts[lot],
                "end_timestamp": ends[lot],
                "cycle_time_hours": float(
                    (ends[lot] - starts[lot]) / np.timedelta64(1, "h")
                )
                if not np.isnat(ends[lot])
                else 0.0,
                "anomaly_count": anoms,
                "dominant_failure_mode": dom,
            }
        )
    lot_df = pd.DataFrame(lot_summaries)

    die_path = output_dir / "wafer_dies.parquet"
    sum_path = output_dir / "wafer_summary.parquet"
    lot_path = output_dir / "lot_summary.parquet"
    die_df.to_parquet(die_path, compression="snappy", index=False)
    sum_df.to_parquet(sum_path, compression="snappy", index=False)
    lot_df.to_parquet(lot_path, compression="snappy", index=False)

    meta = _build_metadata(
        seed=seed,
        n_lots=n_lots,
        wafers_per_lot=wafers_per_lot,
        yield_model=yield_model,
        anomalies_enabled=anomalies_enabled,
        pm_interval=pm_interval,
        sum_df=sum_df,
        die_df=die_df,
        anomaly_log=anomaly_log,
        elapsed_s=elapsed,
        yield_stress_factor=yield_stress_factor,
        grid_n=grid_n,
    )

    meta_path = output_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=_json_default)

    if validate:
        _validate_statistics(sum_df, die_df, meta)

    return meta


def _json_default(o: Any) -> Any:
    if isinstance(o, (datetime, np.datetime64)):
        return str(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError


def _build_metadata(
    seed: int,
    n_lots: int,
    wafers_per_lot: int,
    yield_model: str,
    anomalies_enabled: bool,
    pm_interval: int,
    sum_df: pd.DataFrame,
    die_df: pd.DataFrame,
    anomaly_log: list[AnomalyLogEntry],
    elapsed_s: float,
    yield_stress_factor: float,
    grid_n: int,
) -> dict[str, Any]:
    snap = get_physics_snapshot()
    nominal = sum_df["anomaly_type"].isna()
    per_tool = {}
    for t in ["tool_A", "tool_B", "tool_C"]:
        m = sum_df["tool_id"] == t
        if m.any():
            per_tool[t] = {
                "mean_yield": float(sum_df.loc[m, "wafer_yield"].mean()),
                "std_yield": float(sum_df.loc[m, "wafer_yield"].std(ddof=1)),
            }

    params_desc: dict[str, Any] = {}
    for col in ["temperature", "pressure", "gas_flow", "rf_power", "deposition_time"]:
        params_desc[col] = {
            "mean": float(sum_df[col].mean()),
            "std": float(sum_df[col].std(ddof=1)),
            "min": float(sum_df[col].min()),
            "max": float(sum_df[col].max()),
        }

    inj = [
        {
            "run_id": e.run_id,
            "type": e.anomaly_type,
            "severity": e.severity,
            "affected_parameter": e.affected_parameter,
            "magnitude": e.magnitude,
            "window_start": e.window_start,
            "window_end": e.window_end,
            **e.extra,
        }
        for e in anomaly_log
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "config": {
            "n_lots": n_lots,
            "wafers_per_lot": wafers_per_lot,
            "yield_model": yield_model,
            "anomalies_enabled": anomalies_enabled,
            "pm_interval": pm_interval,
            "grid_n": grid_n,
            "yield_stress_factor": yield_stress_factor,
            "defect_display_scale": DEFECT_DISPLAY_SCALE,
        },
        "parameter_distributions": params_desc,
        "yield_statistics": {
            "overall_mean": float(sum_df["wafer_yield"].mean()),
            "overall_std": float(sum_df["wafer_yield"].std(ddof=1)),
            "nominal_mean": float(sum_df.loc[nominal, "wafer_yield"].mean()),
            "nominal_std": float(sum_df.loc[nominal, "wafer_yield"].std(ddof=1)),
            "per_tool": per_tool,
        },
        "anomaly_injection_log": inj,
        "tool_age_end": {
            "tool_A": int(get_tool_run_count(0)),
            "tool_B": int(get_tool_run_count(1)),
            "tool_C": int(get_tool_run_count(2)),
        },
        "physics_model_snapshot": {
            "yield_model": yield_model,
            **snap,
        },
        "generation_seconds": float(elapsed_s),
        "row_counts": {
            "wafer_dies": int(len(die_df)),
            "wafer_summary": int(len(sum_df)),
            "lot_summary": int(n_lots),
        },
    }


def _validate_statistics(sum_df: pd.DataFrame, die_df: pd.DataFrame, meta: dict[str, Any]) -> None:
    nominal = sum_df["anomaly_type"].isna()
    ny = sum_df.loc[nominal, "wafer_yield"]
    if len(ny) < 10:
        return
    m = float(ny.mean())
    s = float(ny.std(ddof=1))
    if not (0.70 <= m <= 0.92):
        raise DataGenerationError(f"Nominal yield mean {m} outside [0.70, 0.92]")
    if not (0.02 <= s <= 0.18):
        raise DataGenerationError(f"Nominal yield std {s} outside [0.02, 0.18]")

    nu = sum_df.loc[nominal, "thickness_uniformity_pct"]
    if float(nu.mean()) < 1.2 or float(nu.mean()) > 4.5:
        raise DataGenerationError("Thickness uniformity (reported) out of expected band")

    nom_sum = sum_df.loc[nominal]
    inner = nom_sum["lot_position"] > 1
    if inner.any():
        rid_ok = nom_sum.loc[inner, "run_id"]
        md = die_df.loc[die_df["run_id"].isin(rid_ok), "defect_density"]
        if float(md.mean()) < 0.04 or float(md.mean()) > 0.55:
            raise DataGenerationError("Defect density (display) out of band for nominal dies")

    # Correlations (wafer-level)
    sub = sum_df.loc[nominal]
    if len(sub) > 20:
        r_tm = sub[["temperature", "mean_thickness"]].corr().iloc[0, 1]
        if r_tm <= 0.18:
            raise DataGenerationError(f"Temperature–thickness correlation too low: {r_tm}")
        r_dy = sub[["mean_defect_density", "wafer_yield"]].corr().iloc[0, 1]
        if r_dy >= -0.30:
            raise DataGenerationError(f"Defect–yield correlation not negative enough: {r_dy}")
        rf_mu = float(sub["rf_power"].mean())
        sub = sub.copy()
        sub["rf_dev"] = sub["rf_power"] - rf_mu
        r_rd = sub[["rf_dev", "mean_defect_density"]].corr().iloc[0, 1]
        if r_rd <= 0.12:
            raise DataGenerationError(f"RF deviation–defect correlation too low: {r_rd}")

    # Tool spread
    yt = [meta["yield_statistics"]["per_tool"][k]["mean_yield"] for k in meta["yield_statistics"]["per_tool"]]
    if len(yt) >= 2 and (max(yt) - min(yt)) >= 0.30:
        raise DataGenerationError("Tool yield gap unexpectedly large")

    # Anomaly impact
    an = sum_df["anomaly_type"].notna()
    if an.any() and nominal.any():
        a = float(sum_df.loc[an, "wafer_yield"].mean())
        b = float(sum_df.loc[nominal, "wafer_yield"].mean())
        if a > b - 0.04:
            raise DataGenerationError("Anomaly runs not sufficiently lower yield than nominal")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate synthetic wafer fab datasets.")
    p.add_argument("--n-lots", type=int, default=20)
    p.add_argument("--wafers-per-lot", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("data"))
    p.add_argument("--yield-model", choices=["murphy", "seeds", "negative_binomial"], default="murphy")
    p.add_argument("--no-anomalies", action="store_true")
    p.add_argument("--pm-interval", type=int, default=150)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--grid-n", type=int, default=DEFAULT_GRID_N)
    args = p.parse_args(argv)

    generate_dataset(
        n_lots=args.n_lots,
        wafers_per_lot=args.wafers_per_lot,
        seed=args.seed,
        output_dir=args.output_dir,
        yield_model=args.yield_model,
        anomalies_enabled=not args.no_anomalies,
        pm_interval=args.pm_interval,
        verbose=args.verbose,
        validate=args.validate,
        grid_n=args.grid_n,
    )


if __name__ == "__main__":
    main()
