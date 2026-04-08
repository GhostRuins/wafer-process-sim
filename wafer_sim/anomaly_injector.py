"""Anomaly injection plans for synthetic wafer runs (isolated from nominal physics)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

AnomalyType = Literal[
    "temp_drift",
    "particle_shower",
    "tool_drift",
    "recipe_excursion",
    "tool_aging",
]


@dataclass
class WaferAnomalyContext:
    """Per-wafer modifiers and labels (at most one primary anomaly per wafer)."""

    anomaly_type: str | None = None
    anomaly_severity: str | None = None
    temp_delta: float = 0.0
    rf_delta: float = 0.0
    particle_scale: float = 1.0
    particle_quadrant: int | None = None
    pressure_override: float | None = None
    gas_override: float | None = None
    virtual_age_add: int = 0
    # recipe_excursion uses overrides; tool_aging handled via virtual_age_add + follow-up state in generator


@dataclass
class AnomalyLogEntry:
    run_id: str
    anomaly_type: str
    severity: str | None
    affected_parameter: str
    magnitude: float
    window_start: int
    window_end: int
    extra: dict[str, Any] = field(default_factory=dict)


def _tool_chains(tool_ids: np.ndarray) -> dict[int, list[int]]:
    chains: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for idx, tid in enumerate(tool_ids.astype(int)):
        chains[int(tid)].append(idx)
    return chains


def _pick_consecutive_subchain(
    chain: list[int],
    length: int,
    rng: np.random.Generator,
) -> list[int] | None:
    if len(chain) < length:
        return None
    start = int(rng.integers(0, len(chain) - length + 1))
    return chain[start : start + length]


def _severity_temp(max_ramp_c: float) -> str:
    if max_ramp_c < 10.0:
        return "minor"
    if max_ramp_c < 20.0:
        return "major"
    return "critical"


def build_anomaly_plan(
    tool_ids: np.ndarray,
    rng: np.random.Generator,
    enabled: bool = True,
) -> tuple[list[WaferAnomalyContext], list[AnomalyLogEntry]]:
    """Create one anomaly context per wafer run and a metadata log.

    Ensures at most one primary injected anomaly type per wafer. Multi-run
    anomalies reserve all participating run indices first.
    """
    n = int(tool_ids.shape[0])
    ctxs = [WaferAnomalyContext() for _ in range(n)]
    log: list[AnomalyLogEntry] = []
    if not enabled or n == 0:
        return ctxs, log

    occupied: set[int] = set()
    chains = _tool_chains(tool_ids)

    def occupy(idxs: list[int]) -> bool:
        if any(i in occupied for i in idxs):
            return False
        occupied.update(idxs)
        return True

    # --- Multi-run anomalies first ---
    # temp_drift: ~2% of runs touched — approximate via number of windows
    n_temp_windows = max(1, int(round(0.02 * n / 6.0)))
    for _ in range(n_temp_windows):
        tid = int(rng.choice([0, 1, 2]))
        length = int(rng.integers(5, 11))
        seg = _pick_consecutive_subchain(chains[tid], length, rng)
        if seg is None or not occupy(seg):
            continue
        end_ramp = float(rng.uniform(15.0, 28.0))
        sev = _severity_temp(end_ramp)
        for k, run_i in enumerate(seg):
            t = k / max(1, len(seg) - 1)
            ctxs[run_i].anomaly_type = "temp_drift"
            ctxs[run_i].anomaly_severity = sev
            ctxs[run_i].temp_delta = end_ramp * t
        log.append(
            AnomalyLogEntry(
                run_id="",
                anomaly_type="temp_drift",
                severity=sev,
                affected_parameter="temperature_c",
                magnitude=float(end_ramp),
                window_start=int(seg[0]),
                window_end=int(seg[-1]),
                extra={"tool_id": tid, "length": len(seg)},
            )
        )

    # tool_drift: rf bias over 8–15 runs
    n_tool_drift = max(1, int(round(0.02 * n / 10.0)))
    for _ in range(n_tool_drift):
        tid = int(rng.choice([0, 1, 2]))
        length = int(rng.integers(8, 16))
        seg = _pick_consecutive_subchain(chains[tid], length, rng)
        if seg is None or not occupy(seg):
            continue
        drift_total = float(rng.uniform(-30.0, 30.0))
        sev = "major" if abs(drift_total) > 18.0 else "minor"
        for k, run_i in enumerate(seg):
            t = (k + 1) / len(seg)
            ctxs[run_i].anomaly_type = "tool_drift"
            ctxs[run_i].anomaly_severity = sev
            ctxs[run_i].rf_delta = drift_total * t
        log.append(
            AnomalyLogEntry(
                run_id="",
                anomaly_type="tool_drift",
                severity=sev,
                affected_parameter="rf_power_w",
                magnitude=float(drift_total),
                window_start=int(seg[0]),
                window_end=int(seg[-1]),
                extra={"tool_id": tid, "length": len(seg)},
            )
        )

    # recipe_excursion: 1–3 wafers, gas high or pressure high
    n_recipe = max(1, int(round(0.015 * n / 2.0)))
    for _ in range(n_recipe):
        tid = int(rng.choice([0, 1, 2]))
        length = int(rng.integers(1, 4))
        seg = _pick_consecutive_subchain(chains[tid], length, rng)
        if seg is None or not occupy(seg):
            continue
        mode = rng.choice(["gas", "pressure"])
        sev = "major"
        for run_i in seg:
            ctxs[run_i].anomaly_type = "recipe_excursion"
            ctxs[run_i].anomaly_severity = sev
            if mode == "gas":
                ctxs[run_i].gas_override = float(rng.uniform(115.0, 120.0))
                ap = "gas_flow_sccm"
                mag = float(ctxs[run_i].gas_override)
            else:
                ctxs[run_i].pressure_override = float(rng.uniform(90.0, 100.0))
                ap = "pressure_mtorr"
                mag = float(ctxs[run_i].pressure_override)
        log.append(
            AnomalyLogEntry(
                run_id="",
                anomaly_type="recipe_excursion",
                severity=sev,
                affected_parameter=ap,
                magnitude=mag,
                window_start=int(seg[0]),
                window_end=int(seg[-1]),
                extra={"tool_id": tid, "length": len(seg), "mode": mode},
            )
        )

    # particle_shower: ~1.5% single wafers
    target_particle = int(round(0.015 * n))
    placed_p = 0
    tries = 0
    while placed_p < target_particle and tries < n * 5:
        tries += 1
        i = int(rng.integers(0, n))
        if i in occupied:
            continue
        if not occupy([i]):
            continue
        scale = float(rng.uniform(5.0, 15.0))
        q = int(rng.integers(0, 4))
        sev = "critical" if scale > 10.0 else "major"
        ctxs[i].anomaly_type = "particle_shower"
        ctxs[i].anomaly_severity = sev
        ctxs[i].particle_scale = scale
        ctxs[i].particle_quadrant = q
        log.append(
            AnomalyLogEntry(
                run_id="",
                anomaly_type="particle_shower",
                severity=sev,
                affected_parameter="particle_contamination",
                magnitude=scale,
                window_start=i,
                window_end=i,
                extra={"quadrant": q},
            )
        )
        placed_p += 1

    # tool_aging: ~1% — virtual age jump +50 affecting subsequent runs handled in generator state
    target_age = max(1, int(round(0.01 * n)))
    placed_a = 0
    tries = 0
    while placed_a < target_age and tries < n * 5:
        tries += 1
        i = int(rng.integers(0, n))
        if i in occupied:
            continue
        if not occupy([i]):
            continue
        ctxs[i].anomaly_type = "tool_aging"
        ctxs[i].anomaly_severity = "major"
        ctxs[i].virtual_age_add = 50
        log.append(
            AnomalyLogEntry(
                run_id="",
                anomaly_type="tool_aging",
                severity="major",
                affected_parameter="tool_run_count",
                magnitude=50.0,
                window_start=i,
                window_end=i,
                extra={"follow_on_runs": 20},
            )
        )
        placed_a += 1

    return ctxs, log


def particle_quadrant_mask(
    x: np.ndarray,
    y: np.ndarray,
    quadrant: int,
) -> np.ndarray:
    """Multiplier mask (1 outside quadrant, 0 inside) for particle hot spots — invert for boost.

    Returns a factor that is **low** inside the affected quadrant so we multiply
    particle by ``scale`` inside quadrant via: ``particle * (1 + (scale-1)*inside)``.
    """
    # quadrant 0: x>0,y>0; 1: x<0,y>0; 2: x<0,y<0; 3: x>0,y<0
    if quadrant == 0:
        inside = (x > 0) & (y > 0)
    elif quadrant == 1:
        inside = (x <= 0) & (y > 0)
    elif quadrant == 2:
        inside = (x <= 0) & (y <= 0)
    else:
        inside = (x > 0) & (y <= 0)
    return np.where(inside, 1.0, 0.0)
