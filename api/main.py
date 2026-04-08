"""FastAPI entrypoint: wafer data API, simulation, ML prediction, optimization."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routers import optimization, prediction, runs, simulation
from api.schemas import HealthResponse
from api.state import init_app_state


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


@app.get("/health", response_model=HealthResponse, summary="Health check", tags=["data"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok")
