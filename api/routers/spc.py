"""SPC batch + ingest routes."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
from fastapi import APIRouter, HTTPException

from api.deps import AppStateDep
from api.schemas_spc import (
    SPCBatchRequest,
    SPCBatchResponse,
    SPCIngestRequest,
    SPCIngestResponse,
    SPCPointDTO,
    SPCStateSnapshotItem,
    SPCSummaryDTO,
    SPCViolationDTO,
)
from spc.engine import SPC_METRIC_COLUMN_MAP, analyze_series, compute_label_metrics

router = APIRouter(prefix="/spc", tags=["spc"])


def _require_data(state: AppStateDep) -> None:
    if not state.data_loaded or state.lf_wafer_summary is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")


def _collect_series(
    state: AppStateDep,
    *,
    metric: str,
    start: datetime,
    end: datetime,
    tool_id: str | None = None,
):
    _require_data(state)
    assert state.lf_wafer_summary is not None
    col = SPC_METRIC_COLUMN_MAP[metric]
    q = state.lf_wafer_summary.filter(
        (pl.col("timestamp") >= pl.lit(start)) & (pl.col("timestamp") <= pl.lit(end))
    )
    if tool_id:
        q = q.filter(pl.col("tool_id") == tool_id)
    df = (
        q.select(["timestamp", "tool_id", col, "anomaly_type"])
        .sort("timestamp")
        .drop_nulls(["timestamp", col])
        .collect()
    )
    return df, col


@router.post("/analyze", response_model=SPCBatchResponse, summary="Run batch SPC analysis")
async def analyze_spc(body: SPCBatchRequest, state: AppStateDep) -> SPCBatchResponse:
    if body.start > body.end:
        raise HTTPException(status_code=400, detail="start must be <= end")
    df, col = _collect_series(
        state,
        metric=body.metric,
        start=body.start,
        end=body.end,
        tool_id=body.tool_id,
    )
    if df.height == 0:
        return SPCBatchResponse(
            metric=body.metric,
            series_id=body.tool_id or "all_tools",
            summary=SPCSummaryDTO(total_points=0, total_violations=0, violations_by_rule={}),
            points=[],
            violations=[],
        )

    rows = df.to_dicts()
    timestamps = [r["timestamp"] for r in rows]
    values = np.asarray([float(r[col]) for r in rows], dtype=np.float64)
    labels = [r.get("anomaly_type") is not None for r in rows]
    points, violations = analyze_series(
        values,
        timestamps,
        baseline_n=body.baseline.baseline_n,
        ewma_lambda=body.ewma.ewma_lambda,
        ewma_L=body.ewma.ewma_L,
        sigma_floor=body.baseline.sigma_floor,
        ewma_enabled=body.ewma.enabled,
    )

    if body.nelson.enabled:
        allowed = set(body.nelson.rules + [100])
        violations = [v for v in violations if v.rule_id in allowed]
        for p in points:
            p.violation_ids = [rid for rid in p.violation_ids if rid in allowed]
    else:
        violations = [v for v in violations if v.rule_id == 100]
        for p in points:
            p.violation_ids = [rid for rid in p.violation_ids if rid == 100]

    by_rule: dict[str, int] = {}
    for v in violations:
        k = str(v.rule_id)
        by_rule[k] = by_rule.get(k, 0) + 1
    metrics = compute_label_metrics(points, labels)

    return SPCBatchResponse(
        metric=body.metric,
        series_id=body.tool_id or "all_tools",
        summary=SPCSummaryDTO(
            total_points=len(points),
            total_violations=len(violations),
            violations_by_rule=by_rule,
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            lead_time_runs=metrics["lead_time_runs"],
        ),
        points=[SPCPointDTO(**p.__dict__) for p in points] if body.include_points else [],
        violations=[SPCViolationDTO(**v.__dict__) for v in violations],
    )


@router.post("/stream/ingest", response_model=SPCIngestResponse, summary="Ingest one point into streaming SPC")
async def ingest_spc(body: SPCIngestRequest, state: AppStateDep) -> SPCIngestResponse:
    point, violations = state.spc_processor.ingest(
        metric=body.metric,
        series_id=body.series_id,
        timestamp=body.timestamp,
        value=body.value,
        ewma_lambda=body.ewma_lambda,
        ewma_L=body.ewma_L,
        ewma_enabled=body.ewma_enabled,
    )
    return SPCIngestResponse(
        accepted=True,
        point=SPCPointDTO(**point.__dict__),
        violations=[SPCViolationDTO(**v.__dict__) for v in violations],
    )


@router.get("/stream/state", response_model=list[SPCStateSnapshotItem], summary="Get stream SPC state")
async def stream_state(state: AppStateDep) -> list[SPCStateSnapshotItem]:
    return [SPCStateSnapshotItem(**r) for r in state.spc_processor.snapshot()]
