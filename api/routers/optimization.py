"""Physics-only recipe optimization (bonus endpoint)."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter
from scipy.optimize import minimize

from api.convert import api_to_sim_params
from api.deps import AppStateDep
from api.schemas import OptimizeRequest, OptimizeResponse, ProcessParams
from api.util import tool_name_to_int
from wafer_sim.physics import compute_wafer_die_grid

router = APIRouter(prefix="/optimize", tags=["optimization"])

_DEFAULT_BOUNDS = [
    (350.0, 450.0),
    (20.0, 100.0),
    (40.0, 120.0),
    (150.0, 300.0),
    (90.0, 180.0),
]
_LABELS = ["temperature", "pressure", "gas_flow", "rf_power", "deposition_time"]


def _tighten_bounds(constraints: dict[str, Any]) -> list[tuple[float, float]]:
    bounds = [list(b) for b in _DEFAULT_BOUNDS]
    for key, spec in constraints.items():
        if key in _LABELS and isinstance(spec, dict):
            i = _LABELS.index(key)
            if "min" in spec:
                bounds[i][0] = max(bounds[i][0], float(spec["min"]))
            if "max" in spec:
                bounds[i][1] = min(bounds[i][1], float(spec["max"]))
    return [(float(lo), float(hi)) for lo, hi in bounds]


@router.post("", response_model=OptimizeResponse, summary="Optimize recipe parameters for target yield")
async def optimize_recipe(body: OptimizeRequest, state: AppStateDep) -> OptimizeResponse:
    tool_idx = tool_name_to_int(body.tool_id)
    bounds = _tighten_bounds(body.constraints)
    n = state.grid_n
    ym = state.yield_model
    ys = state.yield_stress_factor

    def objective(x: np.ndarray) -> float:
        t, p, g, rf, dep = (float(v) for v in x)
        from wafer_sim.models import ProcessParams as Sim

        sim = Sim(
            temperature_c=t,
            pressure_mtorr=p,
            gas_flow_sccm=g,
            rf_power_w=rf,
            deposition_time_s=dep,
            tool_id=tool_idx,
        )
        grid = compute_wafer_die_grid(
            sim,
            seed=state.physics_seed,
            grid_nx=n,
            grid_ny=n,
            yield_model=ym,  # type: ignore[arg-type]
            yield_stress_factor=ys,
        )
        pen = 0.0
        if grid.wafer_yield < body.target_yield:
            pen = 50.0 * (body.target_yield - grid.wafer_yield) ** 2
        return float(-grid.wafer_yield + pen)

    x0 = np.array(
        [
            0.5 * (bounds[0][0] + bounds[0][1]),
            0.5 * (bounds[1][0] + bounds[1][1]),
            0.5 * (bounds[2][0] + bounds[2][1]),
            0.5 * (bounds[3][0] + bounds[3][1]),
            0.5 * (bounds[4][0] + bounds[4][1]),
        ],
        dtype=np.float64,
    )
    res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 80, "ftol": 1e-9},
    )
    x_opt = res.x
    sim_f = api_to_sim_params(
        ProcessParams(
            temperature=float(x_opt[0]),
            pressure=float(x_opt[1]),
            gas_flow=float(x_opt[2]),
            rf_power=float(x_opt[3]),
            deposition_time=float(x_opt[4]),
            tool_id=body.tool_id,
        )
    )
    grid_f = compute_wafer_die_grid(
        sim_f,
        seed=state.physics_seed,
        grid_nx=n,
        grid_ny=n,
        yield_model=ym,  # type: ignore[arg-type]
        yield_stress_factor=ys,
    )
    achieved = grid_f.wafer_yield
    ok = bool(res.success) and achieved >= body.target_yield - 0.02
    msg = res.message or ("ok" if ok else "optimizer stopped without meeting target")
    return OptimizeResponse(
        suggested_params=ProcessParams(
            temperature=float(x_opt[0]),
            pressure=float(x_opt[1]),
            gas_flow=float(x_opt[2]),
            rf_power=float(x_opt[3]),
            deposition_time=float(x_opt[4]),
            tool_id=body.tool_id,
        ),
        achieved_yield=achieved,
        target_yield=body.target_yield,
        optimizer_success=bool(res.success),
        message=msg,
    )
