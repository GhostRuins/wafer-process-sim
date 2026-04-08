"""Application singletons: parquet lazy frames, metadata, optional ML ensemble."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from wafer_sim.physics import init_physics


@dataclass
class AppState:
    data_dir: Path
    artifacts_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    lf_wafer_summary: pl.LazyFrame | None = None
    lf_wafer_dies: pl.LazyFrame | None = None
    lf_lot_summary: pl.LazyFrame | None = None
    data_loaded: bool = False
    ensemble: Any = None  # YieldEnsemble when ml optional + artifact present
    grid_n: int = 36
    yield_model: str = "murphy"
    yield_stress_factor: float = 35.0
    physics_seed: int = 42


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def init_app_state() -> AppState:
    root = Path(__file__).resolve().parents[1]
    data_dir = Path(os.environ.get("DATA_DIR", root / "data")).resolve()
    artifacts_dir = Path(os.environ.get("ML_ARTIFACTS_DIR", root / "ml" / "artifacts")).resolve()
    meta = _load_json(data_dir / "metadata.json")

    cfg = meta.get("config") or {}
    grid_n = int(cfg.get("grid_n", 36))
    yield_model = str(cfg.get("yield_model", "murphy"))
    yield_stress = float(cfg.get("yield_stress_factor", 35.0))
    seed = int(meta.get("seed") or os.environ.get("PHYSICS_SEED", 42))
    init_physics(seed)

    state = AppState(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        metadata=meta,
        grid_n=grid_n,
        yield_model=yield_model,
        yield_stress_factor=yield_stress,
        physics_seed=seed,
    )

    sum_path = data_dir / "wafer_summary.parquet"
    die_path = data_dir / "wafer_dies.parquet"
    lot_path = data_dir / "lot_summary.parquet"
    if sum_path.is_file() and die_path.is_file() and lot_path.is_file():
        state.lf_wafer_summary = pl.scan_parquet(sum_path)
        state.lf_wafer_dies = pl.scan_parquet(die_path)
        state.lf_lot_summary = pl.scan_parquet(lot_path)
        state.data_loaded = True
        logger.info("Loaded parquet datasets from {}", data_dir)
    else:
        logger.warning(
            "Parquet datasets missing under {}; data endpoints will return errors until generation.",
            data_dir,
        )

    ensemble_path = artifacts_dir / "yield_ensemble.joblib"
    if ensemble_path.is_file():
        try:
            from ml.yield_model import YieldEnsemble, load_ensemble

            ens = load_ensemble(artifacts_dir)
            if isinstance(ens, YieldEnsemble):
                ens.attach_shap()
                state.ensemble = ens
                logger.info("Loaded yield ensemble from {}", ensemble_path)
        except Exception as e:
            logger.exception("Failed to load ML ensemble: {}", e)
    else:
        logger.warning("No yield_ensemble.joblib at {}; /api/predict unavailable.", artifacts_dir)

    return state
