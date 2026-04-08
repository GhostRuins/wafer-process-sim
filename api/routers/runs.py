"""Historical run and analytics routes backed by parquet."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import polars as pl
from fastapi import APIRouter, HTTPException, Query

from api.deps import AppStateDep
from api.schemas import CorrelationMatrix, PaginatedRuns, ProcessRunItem, RunDiesPage, ToolPerformanceRow, WaferMapJSON, YieldTrendPoint
from api.util import wafermap_from_lazy_dies

router = APIRouter(prefix="/runs", tags=["data"])

yield_router = APIRouter(prefix="/yield", tags=["data"])
tools_router = APIRouter(prefix="/tools", tags=["data"])
process_router = APIRouter(prefix="/process-runs", tags=["data"])


def _parse_iso_dt(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _require_data(state: AppStateDep) -> None:
    if not state.data_loaded or state.lf_wafer_summary is None:
        raise HTTPException(
            status_code=503,
            detail="Dataset not loaded. Generate parquet under DATA_DIR (wafer_sim.generator.generate_dataset).",
        )


@process_router.get("", response_model=list[ProcessRunItem], summary="List process runs in time range")
async def list_process_runs(
    state: AppStateDep,
    start: str = Query(..., description="ISO8601 range start (inclusive)"),
    end: str = Query(..., description="ISO8601 range end (inclusive)"),
) -> list[ProcessRunItem]:
    """Dashboard-friendly run list with process parameters and anomaly flag."""
    _require_data(state)
    assert state.lf_wafer_summary is not None
    t0 = _parse_iso_dt(start)
    t1 = _parse_iso_dt(end)
    df = (
        state.lf_wafer_summary.filter(
            (pl.col("timestamp") >= pl.lit(t0)) & (pl.col("timestamp") <= pl.lit(t1)),
        )
        .sort("timestamp")
        .collect()
    )
    out: list[dict[str, Any]] = []
    for r in df.to_dicts():
        ts = r.get("timestamp")
        if ts is not None:
            ts = str(ts)
        at = r.get("anomaly_type")
        out.append(
            {
                "run_id": r.get("run_id"),
                "wafer_id": r.get("wafer_id"),
                "lot_id": r.get("lot_id"),
                "lot_position": int(r.get("lot_position") or 0),
                "tool_id": r.get("tool_id"),
                "wafer_yield": float(r.get("wafer_yield") or 0.0),
                "timestamp": ts,
                "anomaly_injected": at is not None,
                "anomaly_type": at,
                "temperature": float(r.get("temperature") or 0.0),
                "pressure": float(r.get("pressure") or 0.0),
                "gas_flow": float(r.get("gas_flow") or 0.0),
                "rf_power": float(r.get("rf_power") or 0.0),
                "deposition_time": float(r.get("deposition_time") or 0.0),
            },
        )
    return out


@router.get("", response_model=PaginatedRuns, summary="List paginated run summaries")
async def list_runs(
    state: AppStateDep,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> PaginatedRuns:
    _require_data(state)
    assert state.lf_wafer_summary is not None
    total = state.lf_wafer_summary.select(pl.len()).collect().item()
    rows = (
        state.lf_wafer_summary
        .sort("timestamp")
        .slice(offset, limit)
        .collect()
        .to_dicts()
    )
    for r in rows:
        if "timestamp" in r and r["timestamp"] is not None:
            r["timestamp"] = str(r["timestamp"])
    summaries: list[dict[str, Any]] = []
    for r in rows:
        summaries.append(
            {
                "run_id": r.get("run_id"),
                "wafer_id": r.get("wafer_id"),
                "lot_id": r.get("lot_id"),
                "tool_id": r.get("tool_id"),
                "wafer_yield": r.get("wafer_yield"),
                "timestamp": r.get("timestamp"),
                "mean_thickness": r.get("mean_thickness"),
                "anomaly_type": r.get("anomaly_type"),
            }
        )
    return PaginatedRuns(items=summaries, total=int(total), limit=limit, offset=offset)


@router.get("/{run_id}", response_model=dict[str, Any], summary="Get run details by run ID")
async def get_run(run_id: str, state: AppStateDep) -> dict[str, Any]:
    _require_data(state)
    assert state.lf_wafer_summary is not None
    row = (
        state.lf_wafer_summary.filter(pl.col("run_id") == run_id)
        .collect()
        .to_dicts()
    )
    if not row:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    r = row[0]
    if r.get("timestamp") is not None:
        r["timestamp"] = str(r["timestamp"])
    return r


@router.get("/{run_id}/dies", response_model=RunDiesPage, summary="Get die-level data for a run")
async def get_run_dies(
    run_id: str,
    state: AppStateDep,
    limit: int = Query(10_000, ge=1, le=500_000),
    offset: int = Query(0, ge=0),
) -> RunDiesPage:
    _require_data(state)
    assert state.lf_wafer_dies is not None
    q = state.lf_wafer_dies.filter(pl.col("run_id") == run_id)
    total = q.select(pl.len()).collect().item()
    chunk = q.slice(offset, limit).collect().to_dicts()
    for r in chunk:
        if "timestamp" in r and r["timestamp"] is not None:
            r["timestamp"] = str(r["timestamp"])
    return RunDiesPage(run_id=run_id, total=int(total), limit=limit, offset=offset, items=chunk)


@router.get("/{run_id}/wafermap", response_model=WaferMapJSON, summary="Build wafer map tensors for a run")
async def get_run_wafermap(run_id: str, state: AppStateDep) -> WaferMapJSON:
    _require_data(state)
    assert state.lf_wafer_dies is not None
    cfg = state.metadata.get("config") or {}
    nx = ny = int(cfg.get("grid_n", state.grid_n))
    rows = (
        state.lf_wafer_dies.filter(pl.col("run_id") == run_id)
        .collect()
        .to_dicts()
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"No die data for run_id: {run_id}")
    wm = wafermap_from_lazy_dies(rows, nx, ny)
    return WaferMapJSON.model_validate(wm)


@tools_router.get("", response_model=list[ToolPerformanceRow], summary="Get tool-level performance aggregates")
async def tool_performance(state: AppStateDep) -> list[ToolPerformanceRow]:
    _require_data(state)
    assert state.lf_wafer_summary is not None
    df = (
        state.lf_wafer_summary.group_by("tool_id")
        .agg(
            pl.len().alias("run_count"),
            pl.mean("wafer_yield").alias("mean_yield"),
            pl.std("wafer_yield").alias("std_yield"),
            pl.mean("mean_thickness").alias("mean_thickness_nm"),
            pl.mean("mean_defect_density").alias("mean_defect_density"),
        )
        .sort("tool_id")
        .collect()
    )
    out: list[ToolPerformanceRow] = []
    for r in df.to_dicts():
        out.append(
            ToolPerformanceRow(
                tool_id=str(r["tool_id"]),
                run_count=int(r["run_count"]),
                mean_yield=float(r["mean_yield"] or 0.0),
                std_yield=float(r["std_yield"] or 0.0),
                mean_thickness_nm=float(r["mean_thickness_nm"] or 0.0),
                mean_defect_density=float(r["mean_defect_density"] or 0.0),
            )
        )
    return out


@yield_router.get("/trend", response_model=list[YieldTrendPoint], summary="Get daily yield trend points")
async def yield_trend(
    state: AppStateDep,
    days: int = Query(30, ge=1, le=3650),
) -> list[YieldTrendPoint]:
    _require_data(state)
    assert state.lf_wafer_summary is not None
    df = state.lf_wafer_summary.collect()
    if df.height == 0:
        return []
    mx = df.select(pl.col("timestamp").max()).item()
    if mx is None:
        return []
    sub = df.filter(pl.col("timestamp") >= pl.lit(mx) - pl.duration(days=days))
    g = (
        sub.with_columns(pl.col("timestamp").dt.date().alias("d"))
        .group_by("d")
        .agg(pl.mean("wafer_yield").alias("mean_yield"), pl.len().alias("run_count"))
        .sort("d")
    )
    points: list[YieldTrendPoint] = []
    for r in g.to_dicts():
        points.append(
            YieldTrendPoint(
                date=str(r["d"]),
                mean_yield=float(r["mean_yield"]),
                run_count=int(r["run_count"]),
            )
        )
    return points


@yield_router.get("/correlation", response_model=CorrelationMatrix, summary="Get process-to-yield correlation matrix")
async def yield_correlation(state: AppStateDep) -> CorrelationMatrix:
    _require_data(state)
    assert state.lf_wafer_summary is not None
    df = state.lf_wafer_summary.collect()
    cols = [
        "temperature",
        "pressure",
        "gas_flow",
        "rf_power",
        "deposition_time",
        "mean_thickness",
        "std_thickness",
        "mean_defect_density",
        "wafer_yield",
    ]
    use = [c for c in cols if c in df.columns]
    if len(use) < 2:
        raise HTTPException(status_code=500, detail="Not enough numeric columns for correlation.")
    sub = df.select([pl.col(c).cast(pl.Float64, strict=False) for c in use]).drop_nulls()
    n_feat = len(use)
    ident = [[1.0 if i == j else 0.0 for j in range(n_feat)] for i in range(n_feat)]
    if sub.height < 3:
        return CorrelationMatrix(features=use, matrix=ident)
    arr = sub.to_numpy()
    c = np.corrcoef(arr, rowvar=False)
    c = np.nan_to_num(c, nan=0.0)
    mat = [[float(c[i, j]) for j in range(c.shape[1])] for i in range(c.shape[0])]
    return CorrelationMatrix(features=use, matrix=mat)
