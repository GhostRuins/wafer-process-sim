from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any

from spc.engine import SPCPoint, SPCViolation, analyze_series


def _np_array(vals: list[float]):
    import numpy as np

    return np.asarray(vals, dtype=np.float64)


class SPCStreamProcessor:
    def __init__(self, baseline_n: int = 50, max_points: int = 512) -> None:
        self.baseline_n = baseline_n
        self.max_points = max_points
        self._buffers: dict[tuple[str, str], deque[tuple[datetime, float]]] = {}

    def _get_buffer(self, metric: str, series_id: str) -> deque[tuple[datetime, float]]:
        key = (metric, series_id)
        if key not in self._buffers:
            self._buffers[key] = deque(maxlen=self.max_points)
        return self._buffers[key]

    def ingest(
        self,
        *,
        metric: str,
        series_id: str,
        timestamp: datetime,
        value: float,
        ewma_lambda: float = 0.2,
        ewma_L: float = 3.0,
        sigma_floor: float = 1e-6,
        ewma_enabled: bool = True,
    ) -> tuple[SPCPoint, list[SPCViolation]]:
        buf = self._get_buffer(metric, series_id)
        buf.append((timestamp, float(value)))
        vals = [v for _, v in buf]
        ts = [t for t, _ in buf]
        points, violations = analyze_series(
            values=_np_array(vals),
            timestamps=ts,
            baseline_n=self.baseline_n,
            ewma_lambda=ewma_lambda,
            ewma_L=ewma_L,
            sigma_floor=sigma_floor,
            ewma_enabled=ewma_enabled,
        )
        last_idx = len(points) - 1
        return points[last_idx], [v for v in violations if v.index == last_idx]

    def snapshot(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for (metric, series_id), buf in self._buffers.items():
            out.append(
                {
                    "metric": metric,
                    "series_id": series_id,
                    "buffer_size": len(buf),
                    "last_timestamp": buf[-1][0] if buf else None,
                    "last_value": buf[-1][1] if buf else None,
                }
            )
        return out

    def reset(self) -> None:
        self._buffers.clear()
