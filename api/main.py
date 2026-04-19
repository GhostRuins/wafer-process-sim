"""FastAPI entrypoint: wafer data API, simulation, ML prediction, optimization."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import date, datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import numpy as np
import polars as pl

from api.routers import optimization, prediction, runs, simulation, spc
from api.schemas import HealthResponse
from api.state import init_app_state
from spc.engine import SPC_METRIC_COLUMN_MAP
from spc.stream import SPCStreamProcessor


class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if np.isnan(v) else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _to_json_safe(val):
    if val is None:
        return None
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        v = float(val)
        return None if np.isnan(v) else v
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, (bool, int, float, str)):
        return val
    if hasattr(val, "item"):
        return _to_json_safe(val.item())
    return str(val)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO"))
    app.state.app_state = init_app_state()
    logger.info("Wafer API started")
    yield
    logger.info("Wafer API shutdown")


app = FastAPI(
    title="Wafer Process Intelligence API",
    description=(
        "Physics-informed yield prediction for CVD processes. "
        "Ensemble ML model (LightGBM + XGBoost + GP) trained on "
        "2000 wafer runs with lot-stratified cross-validation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

_origins = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs.router, prefix="/api")
app.include_router(runs.tools_router, prefix="/api")
app.include_router(runs.yield_router, prefix="/api")
app.include_router(runs.process_router, prefix="/api")
app.include_router(simulation.router, prefix="/api")
app.include_router(prediction.router, prefix="/api")
app.include_router(optimization.router, prefix="/api")
app.include_router(spc.router, prefix="/api")


@app.get("/health", response_model=HealthResponse, summary="Health check", tags=["data"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.websocket("/ws/spc")
async def spc_websocket(websocket: WebSocket) -> None:
    allowed_origins = [o.strip() for o in _origins if o.strip()]
    origin = websocket.headers.get("origin", "")
    if origin and origin not in allowed_origins:
        await websocket.close(code=1008)
        return

    state = websocket.app.state.app_state
    if not state.data_loaded or state.lf_wafer_summary is None:
        await websocket.close(code=1011)
        return
    await websocket.accept()

    speed_delays = {
        "0.5x": 0.8,
        "1x": 0.4,
        "5x": 0.08,
        "10x": 0.04,
        "max": 0.0,
        "Max": 0.0,
    }

    def _parse_iso(v: str | None) -> datetime | None:
        if not v:
            return None
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None

    try:
        init = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
        metric = str(init.get("metric", "wafer_yield"))
        if metric not in SPC_METRIC_COLUMN_MAP:
            metric = "wafer_yield"
        tool_id = init.get("tool_id")
        start = _parse_iso(init.get("start"))
        end = _parse_iso(init.get("end"))
        speed = str(init.get("speed", "1x"))

        q = state.lf_wafer_summary
        assert q is not None
        col = SPC_METRIC_COLUMN_MAP[metric]
        if start is not None:
            q = q.filter(pl.col("timestamp") >= pl.lit(start))
        if end is not None:
            q = q.filter(pl.col("timestamp") <= pl.lit(end))
        if isinstance(tool_id, str) and tool_id:
            q = q.filter(pl.col("tool_id") == tool_id)
        rows = (
            q.select(["timestamp", "tool_id", "wafer_id", "anomaly_type", col])
            .sort("timestamp")
            .drop_nulls(["timestamp", col])
            .collect()
            .to_dicts()
        )

        await websocket.send_text(
            json.dumps(
                {"type": "spc.connected", "metric": metric, "total_runs": len(rows)},
                cls=SafeJSONEncoder,
            )
        )
        if not rows:
            await websocket.send_text(
                json.dumps({"type": "complete", "total": 0}, cls=SafeJSONEncoder)
            )
            return

        processor = SPCStreamProcessor()
        series_id = str(tool_id) if isinstance(tool_id, str) and tool_id else "all_tools"
        paused = False
        stopped = False
        idx = 0
        total = len(rows)

        while idx < total and not stopped:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                msg = json.loads(raw)
                cmd = str(msg.get("command", "")).lower()
                if cmd == "pause":
                    paused = True
                elif cmd == "resume":
                    paused = False
                elif cmd == "stop":
                    stopped = True
                    break
                elif cmd == "set_speed":
                    speed = str(msg.get("speed", "1x"))
                elif cmd == "ping":
                    pass
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                return
            except Exception:
                pass

            if paused:
                await asyncio.sleep(0.1)
                continue

            row = rows[idx]
            point, violations = processor.ingest(
                metric=metric,
                series_id=series_id,
                timestamp=row["timestamp"],
                value=float(row[col]),
            )
            safe_rules = [f"R{int(v.rule_id)}" for v in violations]
            safe_point = {
                "index": int(point.index),
                "timestamp": _to_json_safe(point.timestamp),
                "value": _to_json_safe(point.value),
                "cl": _to_json_safe(point.cl),
                "ucl": _to_json_safe(point.ucl),
                "lcl": _to_json_safe(point.lcl),
                "ewma": _to_json_safe(point.ewma),
                "ewma_ucl": _to_json_safe(point.ewma_ucl),
                "ewma_lcl": _to_json_safe(point.ewma_lcl),
                "violation_ids": [_to_json_safe(v) for v in point.violation_ids],
                "run_index": int(idx),
                "lot_label": _to_json_safe(row.get("wafer_id")),
                "tool_id": _to_json_safe(row.get("tool_id")),
                "is_anomaly": bool(row.get("anomaly_type") is not None),
                "anomaly_type": _to_json_safe(row.get("anomaly_type")),
                "is_violation": len(violations) > 0,
                "violated_rules": safe_rules,
            }
            payload = {
                "type": "spc.point",
                "metric": metric,
                "series_id": series_id,
                "point": safe_point,
                "violations": [
                    {
                        "rule_id": int(v.rule_id),
                        "rule_name": _to_json_safe(v.rule_name),
                        "severity": _to_json_safe(v.severity),
                        "index": int(v.index),
                        "timestamp": _to_json_safe(v.timestamp),
                        "value": _to_json_safe(v.value),
                        "message": _to_json_safe(v.message),
                        "evidence_indices": [int(i) for i in v.evidence_indices],
                    }
                    for v in violations
                ],
                "run_index": idx + 1,
                "total_runs": total,
            }
            try:
                await websocket.send_text(json.dumps(payload, cls=SafeJSONEncoder))
            except WebSocketDisconnect:
                return

            idx += 1
            delay = float(speed_delays.get(speed, 0.4))
            if delay > 0:
                await asyncio.sleep(delay)
            elif idx % 10 == 0:
                await asyncio.sleep(0)

        if not stopped:
            try:
                await websocket.send_text(
                    json.dumps({"type": "complete", "total": total}, cls=SafeJSONEncoder)
                )
            except Exception:
                pass
    except asyncio.TimeoutError:
        try:
            await websocket.close()
        except Exception:
            pass
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
