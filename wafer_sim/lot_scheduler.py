"""Lot, shift, and timestamp scheduling for synthetic fab runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import numpy as np

ShiftName = Literal["day", "swing", "night"]


@dataclass
class WaferScheduleSlot:
    """One wafer run placement in fab time."""

    timestamp: datetime
    shift: ShiftName


def _hour_to_shift(hour: float) -> ShiftName:
    h = hour % 24.0
    if 6.0 <= h < 14.0:
        return "day"
    if 14.0 <= h < 22.0:
        return "swing"
    return "night"


def build_chronological_timestamps(
    n_wafers: int,
    rng: np.random.Generator,
    start: datetime | None = None,
    span_days: float = 30.0,
) -> list[WaferScheduleSlot]:
    """Produce monotonic timestamps spread across ``span_days`` with realistic jitter.

    Wafers are ordered in processing sequence; spacing is random but keeps the
    series within ``span_days``.
    """
    if start is None:
        start = datetime(2026, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
    total_hours = max(span_days * 24.0, float(n_wafers) * 0.15)
    # Random gaps that sum to approximately total_hours
    gaps = rng.uniform(0.25, 1.4, size=n_wafers)
    gaps = gaps / gaps.sum() * total_hours
    t = float(start.timestamp())
    out: list[WaferScheduleSlot] = []
    for g in gaps:
        t += g * 3600.0
        dt = datetime.fromtimestamp(t, tz=timezone.utc)
        hour = (t / 3600.0) % 24.0
        out.append(WaferScheduleSlot(timestamp=dt, shift=_hour_to_shift(hour)))
    return out


def lot_timestamp_bounds(
    wafer_slots: list[WaferScheduleSlot],
    lot_indices: np.ndarray,
    n_lots: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-lot min/max timestamps from wafer slots (same length as wafer count)."""
    def _to_ns(dt: datetime) -> np.datetime64:
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return np.datetime64(dt.isoformat(), "ns")

    ts_ns = np.array([_to_ns(s.timestamp) for s in wafer_slots], dtype="datetime64[ns]")
    starts = np.full(n_lots, np.datetime64("NaT"), dtype="datetime64[ns]")
    ends = np.full(n_lots, np.datetime64("NaT"), dtype="datetime64[ns]")
    for li in range(n_lots):
        mask = lot_indices == li
        if not np.any(mask):
            continue
        sub = ts_ns[mask]
        starts[li] = sub.min()
        ends[li] = sub.max()
    return starts, ends
