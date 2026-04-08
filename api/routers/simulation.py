"""On-demand wafer simulation and batch regeneration."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from loguru import logger

from api.convert import api_to_sim_params
from api.deps import AppStateDep
from api.schemas import BatchSimulateParams, BatchSimulateResponse, DiePoint, ProcessParams, WaferSimulateResult
from api.util import flat_to_2d_maps
from wafer_sim.physics import compute_wafer_die_grid

router = APIRouter(prefix="/simulate", tags=["simulation"])


@router.post(
    "",
    response_model=WaferSimulateResult,
    response_model_by_alias=True,
    summary="Run single-wafer process simulation",
)
async def simulate_one(
    body: ProcessParams,
    state: AppStateDep,
    seed: int = Query(42, ge=0),
) -> WaferSimulateResult:
    p = api_to_sim_params(body)
    n = state.grid_n
    grid = compute_wafer_die_grid(
        p,
        seed=seed,
        grid_nx=n,
        grid_ny=n,
        yield_model=state.yield_model,  # type: ignore[arg-type]
        yield_stress_factor=state.yield_stress_factor,
    )
    tmap, dmap, ymap = flat_to_2d_maps(n, n, grid)
    dies = []
    for k in range(len(grid.x_pos)):
        dies.append(
            {
                "x": float(grid.x_pos[k]),
                "y": float(grid.y_pos[k]),
                "r": float(grid.r_pos[k]),
                "film_thickness_nm": float(grid.film_thickness_nm[k]),
                "defect_density_cm2": float(grid.defect_density_cm2[k]),
                "die_yield": float(grid.die_yield[k]),
            }
        )
    return WaferSimulateResult(
        wafer_yield=grid.wafer_yield,
        mean_film_thickness_nm=grid.mean_film_thickness_nm,
        std_film_thickness_nm=grid.std_film_thickness_nm,
        mean_defect_density_cm2=grid.mean_defect_density_cm2,
        thickness_map=tmap,
        defect_map=dmap,
        yield_map=ymap,
        dies=[DiePoint.model_validate(d) for d in dies],
    )


def _run_batch_job(params: BatchSimulateParams, output_dir: Path) -> None:
    from wafer_sim.generator import generate_dataset

    logger.info("Starting batch dataset generation: {}", params.model_dump())
    meta = generate_dataset(
        n_lots=params.n_lots,
        wafers_per_lot=params.wafers_per_lot,
        seed=params.seed,
        output_dir=output_dir,
        verbose=True,
    )
    logger.info("Batch generation finished: rows={}", meta.get("row_counts"))


@router.post("/batch", response_model=BatchSimulateResponse, summary="Schedule batch dataset regeneration")
async def simulate_batch(
    state: AppStateDep,
    background_tasks: BackgroundTasks,
    body: BatchSimulateParams | None = None,
) -> BatchSimulateResponse:
    body = body or BatchSimulateParams()
    out = state.data_dir
    if not os.access(out, os.W_OK):
        raise HTTPException(status_code=500, detail=f"DATA_DIR not writable: {out}")
    background_tasks.add_task(_run_batch_job, body, out)
    return BatchSimulateResponse(
        status="scheduled",
        message=f"Regenerating dataset under {out} in background.",
        output_dir=str(out),
    )
