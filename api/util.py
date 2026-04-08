"""Shared helpers for gridding and tool ID mapping."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from wafer_sim.physics import WaferDieGridResult


TOOL_NAME_TO_INT = {"tool_A": 0, "tool_B": 1, "tool_C": 2}
TOOL_INT_TO_NAME = {v: k for k, v in TOOL_NAME_TO_INT.items()}


def tool_name_to_int(name: str) -> int:
    if name not in TOOL_NAME_TO_INT:
        raise ValueError(f"Invalid tool_id: {name}")
    return TOOL_NAME_TO_INT[name]


def tool_int_to_name(tid: int) -> str:
    return TOOL_INT_TO_NAME[int(tid)]


def flat_to_2d_maps(
    grid_nx: int,
    grid_ny: int,
    grid: WaferDieGridResult,
) -> tuple[list[list[float | None]], list[list[float | None]], list[list[float | None]]]:
    xs = np.linspace(-1.0, 1.0, grid_nx)
    ys = np.linspace(-1.0, 1.0, grid_ny)
    thick = [[None if (xv * xv + yv * yv > 1.0) else math.nan for xv in xs] for yv in ys]
    defects = [[None if (xv * xv + yv * yv > 1.0) else math.nan for xv in xs] for yv in ys]
    yields = [[None if (xv * xv + yv * yv > 1.0) else math.nan for xv in xs] for yv in ys]

    x_flat = np.asarray(grid.x_pos)
    y_flat = np.asarray(grid.y_pos)
    ti = np.asarray(grid.film_thickness_nm)
    di = np.asarray(grid.defect_density_cm2)
    yi = np.asarray(grid.die_yield)

    for k in range(len(x_flat)):
        i = int(np.argmin(np.abs(xs - x_flat[k])))
        j = int(np.argmin(np.abs(ys - y_flat[k])))
        thick[j][i] = float(ti[k])
        defects[j][i] = float(di[k])
        yields[j][i] = float(yi[k])

    def _clean_nan(m: list[list[float | None]]) -> list[list[float | None]]:
        out: list[list[float | None]] = []
        for row in m:
            out.append([None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v) for v in row])
        return out

    return _clean_nan(thick), _clean_nan(defects), _clean_nan(yields)


def wafermap_from_lazy_dies(
    rows: list[dict[str, Any]],
    grid_nx: int,
    grid_ny: int,
) -> dict[str, Any]:
    """Pivot pre-fetched die rows into dense rectangular maps (NaN outside disk)."""
    xs = np.linspace(-1.0, 1.0, grid_nx)
    ys = np.linspace(-1.0, 1.0, grid_ny)
    thick = np.full((grid_ny, grid_nx), np.nan, dtype=np.float64)
    dd = np.full((grid_ny, grid_nx), np.nan, dtype=np.float64)
    dy = np.full((grid_ny, grid_nx), np.nan, dtype=np.float64)

    for row in rows:
        xv = float(row["x_pos"])
        yv = float(row["y_pos"])
        if xv * xv + yv * yv > 1.0:
            continue
        i = int(np.argmin(np.abs(xs - xv)))
        j = int(np.argmin(np.abs(ys - yv)))
        thick[j, i] = float(row.get("film_thickness", float("nan")))
        dd[j, i] = float(row.get("defect_density", float("nan")))
        dy[j, i] = float(row.get("die_yield", float("nan")))

    def to_jsonable(arr: np.ndarray) -> list[list[float | None]]:
        out: list[list[float | None]] = []
        for j, yv in enumerate(ys):
            row: list[float | None] = []
            for i, xv in enumerate(xs):
                if xv * xv + yv * yv > 1.0:
                    row.append(None)
                else:
                    v = arr[j, i]
                    row.append(None if math.isnan(v) else float(v))
            out.append(row)
        return out

    return {
        "grid_nx": grid_nx,
        "grid_ny": grid_ny,
        "x_edges": xs.tolist(),
        "y_edges": ys.tolist(),
        "thickness": to_jsonable(thick),
        "defect_density": to_jsonable(dd),
        "die_yield": to_jsonable(dy),
    }
