"""
Physics-inspired CVD process models linking process parameters to wafer outcomes.

Uses module-level state for per-tool constants (drawn once at initialization).
Per-wafer stochasticity uses the ``seed`` argument to :func:`generate_wafer` and
:func:`generate_die`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy import ndimage

from wafer_sim.models import (
    DieResult,
    ProcessParams,
    WaferResult,
    YieldModelConfig,
    validate_process_params,
)

# --- Physical constants (documented units) ---

# Boltzmann constant in eV/K
K_B_EV: float = 8.617333262e-5

# Reference RF power for defect U-curve (W)
RF_NOMINAL_W: float = 200.0

# Reference pressure for defect power-law (mTorr); keeps D0 interpretable
PRESSURE_REF_MTORR: float = 500.0

# Nominal conditions used to calibrate pre-exponential thickness factor A (nm scale)
_NOMINAL_T_C: float = 400.0
_NOMINAL_P_MTORR: float = 500.0
_NOMINAL_FLOW_SCCM: float = 50.0
_NOMINAL_TIME_S: float = 60.0
_TARGET_THICKNESS_NM: float = 100.0

# Hotspot pool size per tool (fixed locations; Poisson draws activate subsets per wafer)
_HOTSPOT_POOL: int = 48

# Turbulence random field grid resolution (unitless normalized coords)
_TURB_GRID: int = 64


@dataclass
class _PhysicsState:
    """Frozen per-tool and global physics constants (initialized once)."""

    module_seed: int
    alpha_pressure: float
    beta_flow: float
    ea_ev: np.ndarray  # shape (3,)
    tool_bias: np.ndarray  # shape (3,)
    preexp_a: float
    cluster_factor: np.ndarray  # shape (3,)
    n_defect_pressure: float
    delta_rf: float
    d0_cm2: float
    tool_hotspot_rate: np.ndarray  # shape (3,) Poisson mean per wafer
    hotspot_xy: np.ndarray  # shape (3, _HOTSPOT_POOL, 2)
    a_hotspot: float
    sigma_hotspot: float
    k_turb: float
    k_u: float
    radial_slope: float
    turb_sigma_norm: float


_STATE: Optional[_PhysicsState] = None

# Per-tool wafer run counter (aging / PM semantics); reset by :func:`init_physics` and :func:`reset_tool_age`.
_TOOL_RUN_COUNT: np.ndarray = np.zeros(3, dtype=np.int64)


def reset_tool_age(tool_id: int) -> None:
    """Reset accumulated run count for a tool (e.g., after preventive maintenance).

    Parameters
    ----------
    tool_id : int
        Tool index 0–2 (unitless).
    """
    if tool_id not in (0, 1, 2):
        raise ValueError("tool_id must be 0, 1, or 2 (unitless categorical index).")
    _TOOL_RUN_COUNT[tool_id] = 0


def get_tool_run_count(tool_id: int) -> int:
    """Return the number of wafer runs recorded since last PM reset for ``tool_id``."""
    if tool_id not in (0, 1, 2):
        raise ValueError("tool_id must be 0, 1, or 2 (unitless categorical index).")
    return int(_TOOL_RUN_COUNT[tool_id])


def increment_tool_run_count(tool_id: int) -> None:
    """Increment run count after a wafer completes on ``tool_id``."""
    if tool_id not in (0, 1, 2):
        raise ValueError("tool_id must be 0, 1, or 2 (unitless categorical index).")
    _TOOL_RUN_COUNT[tool_id] += 1


def add_tool_age_offset(tool_id: int, delta: int) -> None:
    """Add a synthetic age offset (e.g., missed-PM anomaly)."""
    if tool_id not in (0, 1, 2):
        raise ValueError("tool_id must be 0, 1, or 2 (unitless categorical index).")
    _TOOL_RUN_COUNT[tool_id] = max(0, int(_TOOL_RUN_COUNT[tool_id]) + int(delta))


def init_physics(module_seed: int = 42) -> None:
    """Initialize module-level physics constants from a deterministic seed.

    Call this to reproduce tool biases, activation energies, and hotspot sites.
    Safe to call multiple times to reset simulation state.

    Parameters
    ----------
    module_seed : int
        Seed for drawing all per-tool constants (unitless).
    """
    global _STATE
    global _TOOL_RUN_COUNT
    _TOOL_RUN_COUNT = np.zeros(3, dtype=np.int64)
    rng = np.random.default_rng(module_seed)

    # Global exponents (ranges per spec)
    alpha_pressure = float(rng.uniform(0.3, 0.6))
    beta_flow = float(rng.uniform(0.2, 0.5))
    n_defect_pressure = float(rng.uniform(0.5, 1.5))
    delta_rf = float(rng.uniform(1e-5, 5e-5))
    d0_cm2 = 0.1
    a_hotspot = float(rng.uniform(0.02, 0.08))
    sigma_hotspot = float(rng.uniform(0.12, 0.22))
    k_turb = float(rng.uniform(0.0008, 0.0025))
    k_u = float(rng.uniform(0.5, 1.0))
    radial_slope = float(rng.uniform(0.02, 0.04)) * (1.0 if rng.random() < 0.5 else -1.0)
    turb_sigma_norm = 0.3

    ea_ev = np.zeros(3, dtype=np.float64)
    tool_bias = np.zeros(3, dtype=np.float64)
    cluster_factor = np.zeros(3, dtype=np.float64)
    tool_hotspot_rate = np.zeros(3, dtype=np.float64)
    hotspot_xy = np.zeros((3, _HOTSPOT_POOL, 2), dtype=np.float64)

    for tid in range(3):
        tr = np.random.default_rng(module_seed + 1000 * (tid + 1))
        ea_ev[tid] = float(tr.uniform(0.3, 0.6))
        tool_bias[tid] = float(tr.uniform(-0.08, 0.08))
        cluster_factor[tid] = float(tr.uniform(0.5, 2.0))
        tool_hotspot_rate[tid] = float(tr.uniform(0.5, 3.0))
        # Hotspot sites fixed in the normalized wafer square (reject outside disk r<=1)
        k = 0
        while k < _HOTSPOT_POOL:
            x, y = tr.uniform(-1.0, 1.0), tr.uniform(-1.0, 1.0)
            if x * x + y * y <= 1.0:
                hotspot_xy[tid, k, 0] = x
                hotspot_xy[tid, k, 1] = y
                k += 1

    # Calibrate A so nominal conditions give ~100 nm at wafer center (r=0), ignoring tool_bias in calibration
    t_k = _NOMINAL_T_C + 273.15
    arr = math.exp(-ea_ev[0] / (K_B_EV * t_k))
    geom = (
        (_NOMINAL_P_MTORR**alpha_pressure)
        * (_NOMINAL_FLOW_SCCM**beta_flow)
        * _NOMINAL_TIME_S
    )
    preexp_a = _TARGET_THICKNESS_NM / (arr * geom)

    _STATE = _PhysicsState(
        module_seed=module_seed,
        alpha_pressure=alpha_pressure,
        beta_flow=beta_flow,
        ea_ev=ea_ev,
        tool_bias=tool_bias,
        preexp_a=preexp_a,
        cluster_factor=cluster_factor,
        n_defect_pressure=n_defect_pressure,
        delta_rf=delta_rf,
        d0_cm2=d0_cm2,
        tool_hotspot_rate=tool_hotspot_rate,
        hotspot_xy=hotspot_xy,
        a_hotspot=a_hotspot,
        sigma_hotspot=sigma_hotspot,
        k_turb=k_turb,
        k_u=k_u,
        radial_slope=radial_slope,
        turb_sigma_norm=turb_sigma_norm,
    )


def _get_state() -> _PhysicsState:
    """Return initialized module physics state (lazy default seed 42 if unset)."""
    if _STATE is None:
        init_physics(42)
    assert _STATE is not None
    return _STATE


def get_physics_snapshot() -> dict:
    """Serializable snapshot of key physics constants (for dataset metadata)."""
    s = _get_state()
    return {
        "module_seed": int(s.module_seed),
        "alpha_pressure": float(s.alpha_pressure),
        "beta_flow": float(s.beta_flow),
        "Ea_ev_per_tool": [float(x) for x in s.ea_ev.tolist()],
        "k_u": float(s.k_u),
        "preexp_a": float(s.preexp_a),
        "d0_cm2": float(s.d0_cm2),
    }


def radial_gradient(r: float, state: _PhysicsState) -> float:
    """Center-to-edge radial thickness modulation (fractional, unitless).

    For ``r <= 0.85``, a linear gradient; for ``r > 0.85``, a smooth rolloff
    toward the edge exclusion zone.

    Parameters
    ----------
    r : float
        Radial distance from wafer center in normalized coordinates (unitless).
    state : _PhysicsState
        Module physics state.

    Returns
    -------
    float
        Additive fractional contribution to ``nonuniformity_factor`` (unitless).
    """
    if r <= 0.0:
        r = 0.0
    if r <= 0.85:
        return state.radial_slope * (r / 0.85)
    if r >= 1.0:
        return 0.0
    edge = state.radial_slope * (0.85 / 0.85)
    rolloff = 1.0 - ((r - 0.85) / 0.15) ** 2
    if rolloff < 0.0:
        rolloff = 0.0
    return edge * rolloff


def tool_bias(tool_id: int, state: _PhysicsState) -> float:
    """Fixed per-tool fractional bias (unitless)."""
    return float(state.tool_bias[tool_id])


def nonuniformity_factor(x: float, y: float, tool_id: int, state: _PhysicsState) -> float:
    """Spatial + tool nonuniformity used multiplicatively on thickness.

    Parameters
    ----------
    x : float
        Normalized horizontal wafer coordinate in [-1, 1] (unitless).
    y : float
        Normalized vertical wafer coordinate in [-1, 1] (unitless).
    tool_id : int
        Tool index 0–2 (unitless).
    state : _PhysicsState
        Module physics state.

    Returns
    -------
    float
        Dimensionless value such that ``1 + nonuniformity_factor`` scales thickness.
        The sum is clipped to ``[-0.10, 0.10]`` so local thickness stays within
        ±10% of the uniform baseline (unitless).
    """
    r = math.sqrt(x * x + y * y)
    nu = radial_gradient(r, state) + tool_bias(tool_id, state)
    return float(min(max(nu, -0.10), 0.10))


def film_thickness_nm(
    temperature_c: float,
    pressure_mtorr: float,
    gas_flow_sccm: float,
    deposition_time_s: float,
    tool_id: int,
    x: float,
    y: float,
    state: _PhysicsState,
) -> float:
    """Film thickness (nm) from Arrhenius kinetics and multiplicative nonuniformity."""
    t_k = temperature_c + 273.15
    tid = tool_id
    arr = math.exp(-state.ea_ev[tid] / (K_B_EV * t_k))
    nu = nonuniformity_factor(x, y, tid, state)
    thick = (
        state.preexp_a
        * arr
        * (pressure_mtorr**state.alpha_pressure)
        * (gas_flow_sccm**state.beta_flow)
        * deposition_time_s
        * (1.0 + nu)
    )
    return float(thick)


def _draw_hotspot_indices(tool_id: int, rng: np.random.Generator, state: _PhysicsState) -> np.ndarray:
    """Draw how many hotspots are active and which fixed sites.

    Parameters
    ----------
    tool_id : int
        Tool index 0–2 (unitless).
    rng : numpy.random.Generator
        Call-level RNG for the Poisson draw (unitless seed stream).
    state : _PhysicsState
        Module physics state containing ``tool_hotspot_rate`` (expected active
        count per wafer, unitless mean of Poisson distribution).

    Returns
    -------
    numpy.ndarray
        Integer indices in ``[0, _HOTSPOT_POOL)`` selecting fixed hotspot
        coordinates for this wafer/call (unitless).
    """
    n = int(rng.poisson(state.tool_hotspot_rate[tool_id]))
    if n <= 0:
        return np.array([], dtype=np.int64)
    k = min(n, _HOTSPOT_POOL)
    idx = rng.choice(_HOTSPOT_POOL, size=k, replace=False)
    return idx


def _hotspot_contamination(
    x: float,
    y: float,
    tool_id: int,
    state: _PhysicsState,
    hotspot_indices: np.ndarray,
) -> float:
    """Particle contamination from fixed hotspot sites selected for this wafer/call (cm⁻²)."""
    if hotspot_indices.size == 0:
        return 0.0
    pool = state.hotspot_xy[tool_id]
    total = 0.0
    sigma2 = state.sigma_hotspot**2
    for i in hotspot_indices:
        hx, hy = float(pool[i, 0]), float(pool[i, 1])
        dx = x - hx
        dy = y - hy
        dist2 = dx * dx + dy * dy
        total += state.a_hotspot * math.exp(-dist2 / sigma2)
    return float(total)


def _build_turbulence_field(rng: np.random.Generator, state: _PhysicsState) -> np.ndarray:
    """Build a 2D Gaussian-smoothed random field on a normalized wafer grid.

    The field has zero mean and unit standard deviation after normalization; it
    is scaled later by :func:`_turbulence_term` (unitless carrier field).
    """
    raw = rng.standard_normal((_TURB_GRID, _TURB_GRID))
    # sigma in pixels: FWHM-related scale ~ 0.3 in normalized coords over span 2.0
    sigma_pix = (state.turb_sigma_norm / 2.0) * _TURB_GRID
    smooth = ndimage.gaussian_filter(raw, sigma=sigma_pix, mode="reflect")
    smooth -= float(np.mean(smooth))
    std = float(np.std(smooth)) + 1e-12
    smooth /= std
    return smooth


def _turbulence_term(
    x: float,
    y: float,
    gas_flow_sccm: float,
    field: Optional[np.ndarray],
    state: _PhysicsState,
) -> float:
    """Spatially correlated turbulence contamination contribution (cm⁻²).

    Active only when ``gas_flow_sccm > 80`` sccm and ``field`` is precomputed.
    """
    if gas_flow_sccm <= 80.0 or field is None:
        return 0.0
    # Map (x,y) in [-1,1] to grid indices
    gx = (x + 1.0) / 2.0 * (_TURB_GRID - 1)
    gy = (y + 1.0) / 2.0 * (_TURB_GRID - 1)
    ix = int(np.clip(round(gx), 0, _TURB_GRID - 1))
    iy = int(np.clip(round(gy), 0, _TURB_GRID - 1))
    amp = state.k_turb * ((gas_flow_sccm - 80.0) ** 1.5)
    return float(amp * field[ix, iy])


def defect_density_cm2(
    params: ProcessParams,
    x: float,
    y: float,
    state: _PhysicsState,
    turb_field: Optional[np.ndarray],
    hotspot_indices: np.ndarray,
) -> float:
    """Defect density (cm⁻²) with RF U-curve and spatial contamination."""
    p = params
    base = (
        state.d0_cm2
        * ((p.pressure_mtorr / PRESSURE_REF_MTORR) ** state.n_defect_pressure)
        * (1.0 + state.delta_rf * (p.rf_power_w - RF_NOMINAL_W) ** 2)
    )
    hot = _hotspot_contamination(x, y, p.tool_id, state, hotspot_indices)
    turb = _turbulence_term(x, y, p.gas_flow_sccm, turb_field, state)
    return float(max(base + hot + turb, 0.0))


def _murphy_yield(d_a: float, cluster_exp: float) -> float:
    """Murphy clustered-yield form; cluster_exp is the exponent (unitless)."""
    if d_a <= 0.0:
        return 1.0
    inner = (1.0 - math.exp(-d_a)) / d_a
    inner = min(max(inner, 0.0), 1.0)
    return float(inner**cluster_exp)


def _seeds_yield(d_a: float) -> float:
    """Poisson (Seeds) yield exp(-D*A)."""
    return float(math.exp(-d_a))


def _negative_binomial_yield(d_a: float, alpha_nb: float) -> float:
    """Negative-binomial yield (1 + D*A/alpha)^(-alpha)."""
    if alpha_nb <= 0.0:
        raise ValueError("alpha_nb must be positive (unitless).")
    return float((1.0 + d_a / alpha_nb) ** (-alpha_nb))


def _yield_from_model(
    d_a: float,
    model: Literal["murphy", "seeds", "negative_binomial"],
    cluster_factor: float,
) -> float:
    if model == "murphy":
        return _murphy_yield(d_a, cluster_factor)
    if model == "seeds":
        return _seeds_yield(d_a)
    return _negative_binomial_yield(d_a, cluster_factor)


def uniformity_factor(sigma_t_nm: float, mu_t_nm: float, k_u: float) -> float:
    """Multiplicative thickness uniformity factor in [0, 1].

    Parameters
    ----------
    sigma_t_nm : float
        Standard deviation of thickness across dies (nm).
    mu_t_nm : float
        Mean thickness across dies (nm).
    k_u : float
        Sensitivity (unitless), typically 0.5–1.0.

    Returns
    -------
    float
        Factor in [0, 1] (unitless).
    """
    if mu_t_nm <= 0.0:
        return 0.0
    ratio = sigma_t_nm / mu_t_nm
    u = 1.0 - k_u * ratio
    return float(min(max(u, 0.0), 1.0))


def generate_die(
    params: ProcessParams,
    x: float,
    y: float,
    seed: int,
    yield_model: Literal["murphy", "seeds", "negative_binomial"] = "murphy",
    a_die_cm2: float = 1.0,
) -> DieResult:
    """Simulate a single point (die-sized neighborhood) on the wafer.

    Thickness uniformity penalty is **not** applied; ``yield_`` uses only the
    defect-limited model at ``(x, y)``.

    Parameters
    ----------
    params : ProcessParams
        Process parameters; ``wafer_position_*`` are ignored in favor of ``x``, ``y``.
    x : float
        Normalized horizontal coordinate in [-1, 1] (unitless).
    y : float
        Normalized vertical coordinate in [-1, 1] (unitless).
    seed : int
        Per-call seed for Poisson hotspots and turbulence field (unitless).
    yield_model : {'murphy', 'seeds', 'negative_binomial'}, optional
        Which yield closed form to use (unitless selector).
    a_die_cm2 : float
        Die area in cm².

    Returns
    -------
    DieResult
        Local thickness, defect density, and yield.
    """
    validate_process_params(params)
    state = _get_state()
    rng = np.random.default_rng(seed)
    turb_field = _build_turbulence_field(rng, state) if params.gas_flow_sccm > 80.0 else None
    hotspot_indices = _draw_hotspot_indices(params.tool_id, rng, state)

    t_nm = film_thickness_nm(
        params.temperature_c,
        params.pressure_mtorr,
        params.gas_flow_sccm,
        params.deposition_time_s,
        params.tool_id,
        x,
        y,
        state,
    )
    d_cm2 = defect_density_cm2(
        params,
        x,
        y,
        state,
        turb_field,
        hotspot_indices,
    )
    d_a = d_cm2 * a_die_cm2
    yld = _yield_from_model(
        d_a,
        yield_model,
        float(state.cluster_factor[params.tool_id]),
    )
    yld = float(min(max(yld, 0.0), 1.0))
    return DieResult(
        film_thickness_nm=t_nm,
        defect_density_cm2=d_cm2,
        yield_=yld,
    )


def generate_wafer(
    params: ProcessParams,
    seed: int,
    yield_cfg: Optional[YieldModelConfig] = None,
) -> WaferResult:
    """Simulate a full wafer on a rectangular die grid.

    Parameters
    ----------
    params : ProcessParams
        Process parameters shared across the wafer (same run).
    seed : int
        Per-wafer seed for stochastic contamination fields (unitless).
    yield_cfg : YieldModelConfig, optional
        Yield model and grid sizing; defaults to Murphy and 12×12 dies.

    Returns
    -------
    WaferResult
        Mean/std thickness, mean defect density, and combined yield.
    """
    validate_process_params(params)
    state = _get_state()
    cfg = yield_cfg or YieldModelConfig()
    k_u_eff = cfg.k_u if cfg.k_u is not None else state.k_u

    rng = np.random.default_rng(seed)
    turb_field = _build_turbulence_field(rng, state) if params.gas_flow_sccm > 80.0 else None
    hotspot_indices = _draw_hotspot_indices(params.tool_id, rng, state)

    nx, ny = cfg.grid_nx, cfg.grid_ny
    xs = np.linspace(-1.0, 1.0, nx)
    ys = np.linspace(-1.0, 1.0, ny)
    thick = np.zeros((ny, nx), dtype=np.float64)
    defs = np.zeros((ny, nx), dtype=np.float64)

    for j, yv in enumerate(ys):
        for i, xv in enumerate(xs):
            if xv * xv + yv * yv > 1.0:
                thick[j, i] = np.nan
                defs[j, i] = np.nan
                continue
            thick[j, i] = film_thickness_nm(
                params.temperature_c,
                params.pressure_mtorr,
                params.gas_flow_sccm,
                params.deposition_time_s,
                params.tool_id,
                float(xv),
                float(yv),
                state,
            )
            defs[j, i] = defect_density_cm2(
                params,
                float(xv),
                float(yv),
                state,
                turb_field,
                hotspot_indices,
            )

    mask = np.isfinite(thick)
    mu_t = float(np.nanmean(thick))
    sigma_t = float(np.nanstd(thick))
    mean_d = float(np.nanmean(defs))

    d_a = mean_d * cfg.a_die_cm2
    y_m = _yield_from_model(
        d_a,
        cfg.yield_model,
        float(state.cluster_factor[params.tool_id]),
    )
    ufac = uniformity_factor(sigma_t, mu_t, k_u_eff)
    y_tot = float(min(max(y_m * ufac, 0.0), 1.0))

    return WaferResult(
        mean_film_thickness_nm=mu_t,
        std_film_thickness_nm=sigma_t,
        mean_defect_density_cm2=mean_d,
        yield_=y_tot,
    )


@dataclass
class WaferDieGridResult:
    """Vectorized die-level arrays (only points inside the unit wafer disk)."""

    x_pos: np.ndarray
    y_pos: np.ndarray
    r_pos: np.ndarray
    film_thickness_nm: np.ndarray
    thickness_delta_nm: np.ndarray
    charging_damage_cm2: np.ndarray
    particle_contamination_cm2: np.ndarray
    defect_density_cm2: np.ndarray
    die_yield: np.ndarray
    uniformity_factor_die: np.ndarray
    window_factor_die: np.ndarray
    mean_film_thickness_nm: float
    std_film_thickness_nm: float
    range_thickness_nm: float
    thickness_uniformity_pct: float
    mean_defect_density_cm2: float
    max_defect_density_cm2: float
    wafer_yield: float
    yield_loss_defects: float
    yield_loss_uniformity: float
    yield_loss_window: float


def _radial_gradient_arr(r: np.ndarray, state: _PhysicsState) -> np.ndarray:
    """Vectorized :func:`radial_gradient`."""
    r_eff = np.maximum(r, 0.0)
    out = np.zeros_like(r_eff, dtype=np.float64)
    m1 = r_eff <= 0.85
    m2 = (r_eff > 0.85) & (r_eff < 1.0)
    m3 = r_eff >= 1.0
    out[m1] = state.radial_slope * (r_eff[m1] / 0.85)
    edge = state.radial_slope * (0.85 / 0.85)
    rolloff = 1.0 - ((r_eff[m2] - 0.85) / 0.15) ** 2
    rolloff = np.clip(rolloff, 0.0, None)
    out[m2] = edge * rolloff
    out[m3] = 0.0
    return out


def _nonuniformity_factor_arr(
    x: np.ndarray,
    y: np.ndarray,
    tool_id: int,
    state: _PhysicsState,
) -> np.ndarray:
    r = np.sqrt(x * x + y * y)
    nu = _radial_gradient_arr(r, state) + tool_bias(tool_id, state)
    return np.clip(nu, -0.10, 0.10)


def _film_thickness_nm_arr(
    temperature_c: float,
    pressure_mtorr: float,
    gas_flow_sccm: float,
    deposition_time_s: float,
    tool_id: int,
    x: np.ndarray,
    y: np.ndarray,
    state: _PhysicsState,
) -> np.ndarray:
    t_k = temperature_c + 273.15
    tid = tool_id
    arr = np.exp(-state.ea_ev[tid] / (K_B_EV * t_k))
    nu = _nonuniformity_factor_arr(x, y, tid, state)
    return (
        state.preexp_a
        * arr
        * (pressure_mtorr**state.alpha_pressure)
        * (gas_flow_sccm**state.beta_flow)
        * deposition_time_s
        * (1.0 + nu)
    )


def _hotspot_contamination_arr(
    x: np.ndarray,
    y: np.ndarray,
    tool_id: int,
    state: _PhysicsState,
    hotspot_indices: np.ndarray,
) -> np.ndarray:
    if hotspot_indices.size == 0:
        return np.zeros_like(x, dtype=np.float64)
    pool = state.hotspot_xy[tool_id, hotspot_indices]
    dx = x[..., None] - pool[None, None, :, 0]
    dy = y[..., None] - pool[None, None, :, 1]
    dist2 = dx * dx + dy * dy
    sigma2 = state.sigma_hotspot**2
    return np.sum(state.a_hotspot * np.exp(-dist2 / sigma2), axis=-1)


def _turbulence_term_arr(
    x: np.ndarray,
    y: np.ndarray,
    gas_flow_sccm: float,
    field: Optional[np.ndarray],
    state: _PhysicsState,
) -> np.ndarray:
    if gas_flow_sccm <= 80.0 or field is None:
        return np.zeros_like(x, dtype=np.float64)
    gx = (x + 1.0) / 2.0 * (_TURB_GRID - 1)
    gy = (y + 1.0) / 2.0 * (_TURB_GRID - 1)
    ix = np.clip(np.round(gx).astype(np.int64), 0, _TURB_GRID - 1)
    iy = np.clip(np.round(gy).astype(np.int64), 0, _TURB_GRID - 1)
    amp = state.k_turb * ((gas_flow_sccm - 80.0) ** 1.5)
    return amp * field[ix, iy]


def _yield_from_model_arr(
    d_a: np.ndarray,
    model: Literal["murphy", "seeds", "negative_binomial"],
    cluster_factor: float,
) -> np.ndarray:
    d_a = np.maximum(d_a, 0.0)
    if model == "murphy":
        inner = (1.0 - np.exp(-np.maximum(d_a, 1e-15))) / np.maximum(d_a, 1e-15)
        inner = np.clip(inner, 0.0, 1.0)
        return inner**cluster_factor
    if model == "seeds":
        return np.exp(-d_a)
    alpha_nb = cluster_factor
    return (1.0 + d_a / alpha_nb) ** (-alpha_nb)


def compute_wafer_die_grid(
    params: ProcessParams,
    seed: int,
    grid_nx: int,
    grid_ny: int,
    yield_model: Literal["murphy", "seeds", "negative_binomial"] = "murphy",
    a_die_cm2: float = 1.0,
    k_u: Optional[float] = None,
    t_target_nm: float = _TARGET_THICKNESS_NM,
    defect_density_scale: float = 1.0,
    particle_scale: float = 1.0,
    charging_scale: float = 1.0,
    particle_quadrant_mask: Optional[np.ndarray] = None,
    particle_quadrant_boost: Optional[tuple[float, int]] = None,
    yield_stress_factor: float = 1.0,
) -> WaferDieGridResult:
    """Fully vectorized die grid inside the wafer disk (for dataset generation).

    Decomposes defect density into RF/pressure ``charging_damage`` (uniform) and
    spatial ``particle_contamination`` (hotspots + turbulence).
    """
    validate_process_params(params)
    state = _get_state()
    k_u_eff = k_u if k_u is not None else state.k_u

    rng = np.random.default_rng(seed)
    turb_field = _build_turbulence_field(rng, state) if params.gas_flow_sccm > 80.0 else None
    hotspot_indices = _draw_hotspot_indices(params.tool_id, rng, state)

    xs = np.linspace(-1.0, 1.0, grid_nx)
    ys = np.linspace(-1.0, 1.0, grid_ny)
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    r = np.sqrt(xv * xv + yv * yv)
    disk = r <= 1.0

    thick = _film_thickness_nm_arr(
        params.temperature_c,
        params.pressure_mtorr,
        params.gas_flow_sccm,
        params.deposition_time_s,
        params.tool_id,
        xv,
        yv,
        state,
    )
    thick = np.where(disk, thick, np.nan)

    p = params
    base = (
        state.d0_cm2
        * ((p.pressure_mtorr / PRESSURE_REF_MTORR) ** state.n_defect_pressure)
        * (1.0 + state.delta_rf * (p.rf_power_w - RF_NOMINAL_W) ** 2)
    )
    hot = _hotspot_contamination_arr(xv, yv, p.tool_id, state, hotspot_indices)
    turb = _turbulence_term_arr(xv, yv, p.gas_flow_sccm, turb_field, state)

    charging = np.full_like(xv, base * charging_scale, dtype=np.float64)
    particle = (hot + turb) * particle_scale
    if particle_quadrant_boost is not None:
        sc, quad = particle_quadrant_boost
        if quad == 0:
            inside = (xv > 0) & (yv > 0)
        elif quad == 1:
            inside = (xv <= 0) & (yv > 0)
        elif quad == 2:
            inside = (xv <= 0) & (yv <= 0)
        else:
            inside = (xv > 0) & (yv <= 0)
        particle = particle * (1.0 + (sc - 1.0) * inside.astype(np.float64))
    if particle_quadrant_mask is not None:
        particle = particle * particle_quadrant_mask
    defect = np.maximum((charging + particle) * defect_density_scale, 0.0)
    defect = np.where(disk, defect, np.nan)

    mu_t = float(np.nanmean(thick))
    sigma_t = float(np.nanstd(thick))
    tmax = float(np.nanmax(thick))
    tmin = float(np.nanmin(thick))
    mean_d = float(np.nanmean(defect))
    max_d = float(np.nanmax(defect))

    d_a_mean = mean_d * a_die_cm2 * yield_stress_factor
    y_m = _yield_from_model(d_a_mean, yield_model, float(state.cluster_factor[p.tool_id]))
    ufac_w = uniformity_factor(sigma_t, mu_t, k_u_eff)
    y_ref_no_window = float(min(max(y_m * ufac_w, 0.0), 1.0))

    yield_loss_defects = float(min(max(1.0 - y_m, 0.0), 1.0))
    yield_loss_uniformity = float(min(max(y_m * (1.0 - ufac_w), 0.0), 1.0))

    # Per-die window factor (edge exclusion / window loss)
    win = np.ones_like(xv, dtype=np.float64)
    win = np.where(r <= 0.85, 1.0, np.clip(1.0 - 0.12 * ((r - 0.85) / 0.15), 0.88, 1.0))
    win = np.where(disk, win, np.nan)

    d_a_die = defect * a_die_cm2 * yield_stress_factor
    y_def = _yield_from_model_arr(d_a_die, yield_model, float(state.cluster_factor[p.tool_id]))
    uni = 1.0 - k_u_eff * np.abs(thick - mu_t) / np.maximum(mu_t, 1e-12)
    uni = np.clip(uni, 0.0, 1.0)
    uni = np.where(disk, uni, np.nan)

    die_y = y_def * uni * win
    die_y = np.clip(die_y, 0.0, 1.0)
    die_y = np.where(disk, die_y, np.nan)

    die_y_flat = die_y[disk]
    wafer_yield = float(np.mean(die_y_flat))
    yield_loss_window = float(min(max(y_ref_no_window - wafer_yield, 0.0), 1.0))

    delta = thick - t_target_nm

    flat = disk.ravel()
    return WaferDieGridResult(
        x_pos=xv.ravel()[flat],
        y_pos=yv.ravel()[flat],
        r_pos=r.ravel()[flat],
        film_thickness_nm=thick.ravel()[flat],
        thickness_delta_nm=delta.ravel()[flat],
        charging_damage_cm2=charging.ravel()[flat],
        particle_contamination_cm2=particle.ravel()[flat],
        defect_density_cm2=defect.ravel()[flat],
        die_yield=die_y.ravel()[flat],
        uniformity_factor_die=uni.ravel()[flat],
        window_factor_die=win.ravel()[flat],
        mean_film_thickness_nm=mu_t,
        std_film_thickness_nm=sigma_t,
        range_thickness_nm=tmax - tmin,
        thickness_uniformity_pct=(sigma_t / mu_t * 100.0) if mu_t > 0 else 0.0,
        mean_defect_density_cm2=mean_d,
        max_defect_density_cm2=max_d,
        wafer_yield=wafer_yield,
        yield_loss_defects=yield_loss_defects,
        yield_loss_uniformity=yield_loss_uniformity,
        yield_loss_window=yield_loss_window,
    )


# Initialize default module state on import
init_physics(42)
