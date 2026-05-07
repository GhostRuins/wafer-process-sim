"""Pydantic v2 models exposed by the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ProcessParams(BaseModel):
    temperature: float = Field(..., ge=350, le=450, description="°C")
    pressure: float = Field(..., ge=20, le=100, description="mTorr")
    gas_flow: float = Field(..., ge=40, le=120, description="sccm")
    rf_power: float = Field(..., ge=150, le=300, description="W")
    deposition_time: float = Field(..., ge=90, le=180, description="s")
    tool_id: Literal["tool_A", "tool_B", "tool_C"]

    class Config:
        json_schema_extra = {
            "example": {
                "temperature": 400.0,
                "pressure": 50.0,
                "gas_flow": 80.0,
                "rf_power": 225.0,
                "deposition_time": 120.0,
                "tool_id": "tool_A",
            }
        }


class YieldPrediction(BaseModel):
    predicted_yield: float
    confidence_interval: tuple[float, float]
    shap_breakdown: dict[str, float]
    risk_flags: list[str]
    spc_alerts_active: bool = False
    spc_severity: float = 0.0
    spc_alert_count: int = 0

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_yield": 0.923,
                "confidence_interval": [0.901, 0.941],
                "shap_breakdown": {"temperature": 0.012, "gas_flow": -0.018},
                "risk_flags": ["gas_flow_high"],
                "spc_alerts_active": True,
                "spc_severity": 1.0,
                "spc_alert_count": 2,
            }
        }


class DiePoint(BaseModel):
    x: float
    y: float
    r: float
    film_thickness_nm: float
    defect_density_cm2: float
    die_yield: float

    class Config:
        json_schema_extra = {
            "example": {
                "x": 12,
                "y": 8,
                "r": 0.52,
                "film_thickness_nm": 701.2,
                "defect_density_cm2": 0.11,
                "die_yield": 0.94,
            }
        }


class WaferSimulateResult(BaseModel):
    wafer_yield: float = Field(serialization_alias="yield")
    mean_film_thickness_nm: float
    std_film_thickness_nm: float
    mean_defect_density_cm2: float
    thickness_map: list[list[float | None]]
    defect_map: list[list[float | None]]
    yield_map: list[list[float | None]]
    dies: list[DiePoint]

    class Config:
        json_schema_extra = {
            "example": {
                "yield": 0.918,
                "mean_film_thickness_nm": 700.5,
                "std_film_thickness_nm": 3.1,
                "mean_defect_density_cm2": 0.12,
                "thickness_map": [[700.1, 699.8], [701.0, 700.9]],
                "defect_map": [[0.1, 0.13], [0.09, 0.14]],
                "yield_map": [[0.95, 0.91], [0.97, 0.9]],
                "dies": [
                    {
                        "x": 0,
                        "y": 0,
                        "r": 0.2,
                        "film_thickness_nm": 700.1,
                        "defect_density_cm2": 0.1,
                        "die_yield": 0.95,
                    }
                ],
            }
        }


class PaginatedRuns(BaseModel):
    items: list[dict[str, Any]]
    total: int
    limit: int
    offset: int

    class Config:
        json_schema_extra = {
            "example": {
                "items": [{"run_id": "RUN_000001", "wafer_id": "WAF_L001_000001"}],
                "total": 500,
                "limit": 50,
                "offset": 0,
            }
        }


class ProcessRunItem(BaseModel):
    run_id: str
    wafer_id: str
    lot_id: str | None = None
    lot_position: int | None = None
    tool_id: str
    wafer_yield: float
    timestamp: str
    anomaly_injected: bool = False
    anomaly_type: str | None = None
    temperature: float
    pressure: float
    gas_flow: float
    rf_power: float
    deposition_time: float

    class Config:
        json_schema_extra = {
            "example": {
                "run_id": "RUN_000123",
                "wafer_id": "WAF_L004_000123",
                "lot_id": "LOT_004",
                "lot_position": 7,
                "tool_id": "tool_A",
                "wafer_yield": 0.918,
                "timestamp": "2026-01-20 13:02:00",
                "anomaly_injected": False,
                "anomaly_type": None,
                "temperature": 402.1,
                "pressure": 48.7,
                "gas_flow": 81.4,
                "rf_power": 223.0,
                "deposition_time": 119.4,
            }
        }


class RunDiesPage(BaseModel):
    run_id: str
    total: int
    limit: int
    offset: int
    items: list[dict[str, Any]]

    class Config:
        json_schema_extra = {
            "example": {
                "run_id": "RUN_000123",
                "total": 1296,
                "limit": 500,
                "offset": 0,
                "items": [{"x_pos": 0.0, "y_pos": 0.0, "die_yield": 0.95}],
            }
        }


class BatchSimulateResponse(BaseModel):
    status: str
    message: str
    output_dir: str

    class Config:
        json_schema_extra = {
            "example": {
                "status": "scheduled",
                "message": "Regenerating dataset under data in background.",
                "output_dir": "data",
            }
        }


class HealthResponse(BaseModel):
    status: str

    class Config:
        json_schema_extra = {"example": {"status": "ok"}}


class WaferMapJSON(BaseModel):
    grid_nx: int
    grid_ny: int
    x_edges: list[float]
    y_edges: list[float]
    thickness: list[list[float | None]]
    defect_density: list[list[float | None]]
    die_yield: list[list[float | None]]

    class Config:
        json_schema_extra = {
            "example": {
                "grid_nx": 2,
                "grid_ny": 2,
                "x_edges": [0.0, 0.5, 1.0],
                "y_edges": [0.0, 0.5, 1.0],
                "thickness": [[700.0, 701.1], [699.9, 700.8]],
                "defect_density": [[0.11, 0.13], [0.1, 0.14]],
                "die_yield": [[0.96, 0.92], [0.97, 0.9]],
            }
        }


class ToolPerformanceRow(BaseModel):
    tool_id: str
    run_count: int
    mean_yield: float
    std_yield: float
    mean_thickness_nm: float
    mean_defect_density: float

    class Config:
        json_schema_extra = {
            "example": {
                "tool_id": "tool_A",
                "run_count": 180,
                "mean_yield": 0.917,
                "std_yield": 0.028,
                "mean_thickness_nm": 700.3,
                "mean_defect_density": 0.12,
            }
        }


class YieldTrendPoint(BaseModel):
    date: str
    mean_yield: float
    run_count: int

    class Config:
        json_schema_extra = {
            "example": {
                "date": "2026-01-20",
                "mean_yield": 0.914,
                "run_count": 24,
            }
        }


class CorrelationMatrix(BaseModel):
    features: list[str]
    matrix: list[list[float]]

    class Config:
        json_schema_extra = {
            "example": {
                "features": ["temperature", "pressure", "wafer_yield"],
                "matrix": [[1.0, -0.21, 0.31], [-0.21, 1.0, -0.44], [0.31, -0.44, 1.0]],
            }
        }


class OptimizeRequest(BaseModel):
    target_yield: float = Field(..., ge=0.0, le=1.0)
    tool_id: Literal["tool_A", "tool_B", "tool_C"]
    constraints: dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "target_yield": 0.95,
                "tool_id": "tool_B",
                "constraints": {"temperature": [390, 420], "pressure": [35, 60]},
            }
        }


class OptimizeResponse(BaseModel):
    suggested_params: ProcessParams
    achieved_yield: float
    target_yield: float
    optimizer_success: bool
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "suggested_params": {
                    "temperature": 404.0,
                    "pressure": 48.0,
                    "gas_flow": 76.0,
                    "rf_power": 221.0,
                    "deposition_time": 118.0,
                    "tool_id": "tool_B",
                },
                "achieved_yield": 0.948,
                "target_yield": 0.95,
                "optimizer_success": True,
                "message": "Converged within tolerance.",
            }
        }


class BatchSimulateParams(BaseModel):
    n_lots: int = Field(20, ge=1, le=500)
    wafers_per_lot: int = Field(25, ge=1, le=200)
    seed: int = Field(42, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "n_lots": 20,
                "wafers_per_lot": 25,
                "seed": 42,
            }
        }
