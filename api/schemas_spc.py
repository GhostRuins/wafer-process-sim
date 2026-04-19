"""SPC API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SPCMetric = Literal["wafer_yield", "mean_film_thickness_nm", "mean_defect_density_cm2"]
SPCMode = Literal["batch", "stream"]


class SPCBaselineConfig(BaseModel):
    baseline_n: int = Field(50, ge=10, le=500)
    sigma_floor: float = Field(1e-6, gt=0)


class SPCEWMAConfig(BaseModel):
    enabled: bool = True
    ewma_lambda: float = Field(0.2, gt=0, lt=1)
    ewma_L: float = Field(3.0, gt=0)


class SPCNelsonConfig(BaseModel):
    enabled: bool = True
    rules: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])


class SPCBatchRequest(BaseModel):
    metric: SPCMetric = "wafer_yield"
    start: datetime
    end: datetime
    tool_id: str | None = None
    mode: SPCMode = "batch"
    include_points: bool = True
    baseline: SPCBaselineConfig = Field(default_factory=SPCBaselineConfig)
    ewma: SPCEWMAConfig = Field(default_factory=SPCEWMAConfig)
    nelson: SPCNelsonConfig = Field(default_factory=SPCNelsonConfig)


class SPCPointDTO(BaseModel):
    index: int
    timestamp: datetime
    value: float
    cl: float
    ucl: float
    lcl: float
    ewma: float
    ewma_ucl: float
    ewma_lcl: float
    violation_ids: list[int]


class SPCViolationDTO(BaseModel):
    rule_id: int
    rule_name: str
    severity: Literal["low", "medium", "high"]
    index: int
    timestamp: datetime
    value: float
    message: str
    evidence_indices: list[int]


class SPCSummaryDTO(BaseModel):
    total_points: int
    total_violations: int
    violations_by_rule: dict[str, int]
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    lead_time_runs: float | None = None


class SPCBatchResponse(BaseModel):
    metric: SPCMetric
    series_id: str
    summary: SPCSummaryDTO
    points: list[SPCPointDTO] = Field(default_factory=list)
    violations: list[SPCViolationDTO] = Field(default_factory=list)


class SPCIngestRequest(BaseModel):
    metric: SPCMetric
    series_id: str
    timestamp: datetime
    value: float
    ewma_lambda: float = Field(0.2, gt=0, lt=1)
    ewma_L: float = Field(3.0, gt=0)
    ewma_enabled: bool = True


class SPCIngestResponse(BaseModel):
    accepted: bool
    point: SPCPointDTO
    violations: list[SPCViolationDTO] = Field(default_factory=list)


class SPCReplayRequest(BaseModel):
    metric: SPCMetric
    start: datetime
    end: datetime
    tool_id: str | None = None
    replay_speed: Literal["1x", "10x", "100x"] = "10x"
    reset_state: bool = True
    ewma_enabled: bool = True


class SPCReplayResponse(BaseModel):
    replayed_points: int
    emitted_violations: int
    replay_speed: str


class SPCStateSnapshotItem(BaseModel):
    metric: str
    series_id: str
    buffer_size: int
    last_timestamp: datetime | None = None
    last_value: float | None = None
